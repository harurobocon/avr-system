"""ffmpeg サブプロセスによる取得層（RTMP / USB 共通）。

PoC の実測（poc/results/FINDINGS.md F-3, F-4）で決めた構成:

  ffmpeg -i <入力>
    -map 0:v -c:v copy  -f segment ... out_%Y%m%d-%H%M%S.mkv   # 無劣化・常時録画
    -map 0:v -vf scale=WxH -pix_fmt bgr24 -f rawvideo -        # プレビュー(stdout)

1 プロセスで「落としてはいけない録画」と「表示用の軽いプレビュー」を同時に出す。
録画パスにデコードコストが一切かからないのが要点。OpenCV の VideoWriter は
MJPG 指定でも再エンコードしてしまうため、録画には使わない。

入力を差し替えるだけで RTMP と USB の両方を同じ扱いにできる。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .frame_source import Capabilities, Frame, FrameSource, SourceState


@dataclass(frozen=True)
class PreviewSize:
    width: int = 640
    height: int = 360

    @property
    def nbytes(self) -> int:
        return self.width * self.height * 3   # bgr24


class FfmpegSource(FrameSource):
    """ffmpeg の stdout から生 BGR フレームを読み出すソース。

    RTMP は送出端末ごとに独自のエンコーダとバッファを持つため、USB と違って
    カメラ間のずれが数百ms〜秒オーダーになりうる。ここで付けるタイムスタンプは
    あくまで「受信時刻」であり、露光時刻ではないことに注意。
    """

    # 切断しても勝手に諦めない。大会中に1台落ちたまま、が最悪の事態なので。
    RESTART_DELAY_S = 2.0

    def __init__(
        self,
        source_id: str,
        input_args: list[str],
        preview: PreviewSize | None = None,
        record_dir: Optional[Path] = None,
        segment_seconds: int = 10,
        fps: float = 30.0,
        name: str = "",
        auto_restart: bool = True,
        ffmpeg: str = "ffmpeg",
    ) -> None:
        super().__init__(source_id)
        self._input_args = list(input_args)
        self._preview = preview or PreviewSize()
        self._record_dir = Path(record_dir) if record_dir else None
        self._segment_seconds = segment_seconds
        self._fps = fps
        self._name = name or source_id
        self._auto_restart = auto_restart
        self._ffmpeg = ffmpeg

        self._proc: Optional[subprocess.Popen] = None
        self._seq = 0
        self._interval_ns = int(1e9 / fps) if fps > 0 else 0
        self._want_stop = threading.Event()
        self._stderr_tail: list[str] = []
        self._stderr_thread: Optional[threading.Thread] = None
        self._restarts = 0
        self._lock_proc = threading.Lock()

    # ------------------------------------------------------------ 生成用
    @classmethod
    def from_rtmp(cls, source_id: str, url: str, low_latency: bool = True,
                  listen: bool = False, **kw) -> "FfmpegSource":
        """RTMP から取得する。

        既定は「既にある RTMP サーバーを購読する」クライアント動作。
        listen=True にすると自分が受信側になる（送出側を直接受ける場合）。

        実測（1280x720@30 を 26 秒送出、poc/results/FINDINGS.md F-7）:

        | 設定                        | 初フレーム | 定常fps | 受信率 |
        |-----------------------------|-----------|---------|--------|
        | analyzeduration=1s,probe=1M | 2.98s     | 30.0    | 94%    |
        | analyzeduration=0,probe=32k | 1.69s     | 30.0    | 98%    |

        どちらも定常状態は 30.0fps で安定しており、遅延は蓄積しない。
        起動直後だけ溜めたぶんを速く吐くので、その間の受信時刻は
        実際の撮影間隔を表さない点に注意。

        low_latency=False は、解析情報が足りずに解像度を確定できない
        ストリームに当たった場合の逃げ道。
        """
        args: list[str] = ["-fflags", "nobuffer", "-flags", "low_delay"]
        if low_latency:
            args += ["-analyzeduration", "0", "-probesize", "32768"]
        else:
            args += ["-analyzeduration", "2000000", "-probesize", "2000000"]
        if listen:
            args += ["-listen", "1", "-f", "flv"]
        else:
            args += ["-rtmp_live", "live"]
        args += ["-i", url]
        kw.setdefault("name", url)
        return cls(source_id, args, **kw)

    @classmethod
    def from_rtsp(cls, source_id: str, url: str, **kw) -> "FfmpegSource":
        args = [
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-rtsp_transport", "tcp",        # UDP はフレーム欠けの原因になりやすい
            "-i", url,
        ]
        kw.setdefault("name", url)
        return cls(source_id, args, **kw)

    @classmethod
    def from_dshow(cls, source_id: str, device: str, width: int = 1920,
                   height: int = 1080, fps: float = 30.0, **kw) -> "FfmpegSource":
        """USB カメラ。F-2 のとおり FHD は MJPEG でしか出ないので明示する。"""
        args = [
            "-f", "dshow",
            "-vcodec", "mjpeg",
            "-video_size", f"{width}x{height}",
            "-framerate", str(int(fps)),
            "-i", f"video={device}",
        ]
        kw.setdefault("name", device)
        kw.setdefault("fps", fps)
        return cls(source_id, args, **kw)

    # ------------------------------------------------------------ 情報
    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(self._preview.width, self._preview.height,
                            self._fps, "BGR24(preview)", name=self._name)

    @property
    def restarts(self) -> int:
        return self._restarts

    @property
    def last_error(self) -> str:
        return "\n".join(self._stderr_tail[-5:])

    # ------------------------------------------------------------ 起動
    def _build_command(self) -> list[str]:
        cmd = [self._ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin"]
        cmd += self._input_args

        if self._record_dir is not None:
            self._record_dir.mkdir(parents=True, exist_ok=True)
            # セグメント化しておくと、プロセスが落ちても閉じ済みの区間は壊れない。
            # 「試合開始10秒前」はセグメント境界とタイムスタンプから後で切り出す。
            cmd += [
                "-map", "0:v", "-c:v", "copy",
                "-f", "segment",
                "-segment_time", str(self._segment_seconds),
                "-reset_timestamps", "1",
                "-strftime", "1",
                str(self._record_dir / f"{self.source_id}_%Y%m%d-%H%M%S.mkv"),
            ]

        # プレビューは必ず指定どおりの画素数にする。
        # サイズが変わると stdout のバイト境界がずれてフレームが崩れるため、
        # アスペクト比は維持しつつ黒帯で固定サイズに詰める。
        w, h = self._preview.width, self._preview.height
        vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
              f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2")
        cmd += [
            "-map", "0:v", "-an",
            "-vf", vf,
            "-pix_fmt", "bgr24",
            "-f", "rawvideo", "-",
        ]
        return cmd

    def _spawn(self) -> None:
        cmd = self._build_command()
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=self._preview.nbytes * 2,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self._proc,), daemon=True,
            name=f"ffmpeg-err-{self.source_id}")
        self._stderr_thread.start()

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        """stderr を読み捨てないこと。放置するとパイプが詰まって停止する。"""
        if proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self._stderr_tail.append(line)
                del self._stderr_tail[:-50]

    def start(self) -> None:
        if shutil.which(self._ffmpeg) is None:
            self._set_state(SourceState.ERROR)
            raise RuntimeError("ffmpeg が見つかりません")
        self._want_stop.clear()
        self._seq = 0
        self._restarts = 0
        with self._lock_proc:
            self._spawn()
        self._set_state(SourceState.RUNNING)

    def stop(self) -> None:
        self._want_stop.set()
        with self._lock_proc:
            proc, self._proc = self._proc, None
        if proc is not None:
            # segment muxer に最後のファイルを閉じさせてから終了させる
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception:
                pass
        self._set_state(SourceState.STOPPED)

    # ------------------------------------------------------------ 読み出し
    def _read_exact(self, proc: subprocess.Popen, n: int) -> Optional[bytes]:
        """rawvideo は固定長。途中で切れたフレームは捨てる。"""
        if proc.stdout is None:
            return None
        buf = bytearray()
        while len(buf) < n:
            if self._want_stop.is_set():
                return None
            chunk = proc.stdout.read(n - len(buf))
            if not chunk:
                return None          # EOF = プロセス終了 or 切断
            buf += chunk
        return bytes(buf)

    def read(self) -> Optional[Frame]:
        with self._lock_proc:
            proc = self._proc
        if proc is None:
            return None

        data = self._read_exact(proc, self._preview.nbytes)
        ts_ns = time.perf_counter_ns()

        if data is None:
            if self._want_stop.is_set():
                return None
            self.stats.read_failures += 1
            self._set_state(SourceState.DISCONNECTED)
            self._maybe_restart()
            return None

        if self.state is not SourceState.RUNNING:
            self._set_state(SourceState.RUNNING)
        img = np.frombuffer(data, dtype=np.uint8).reshape(
            self._preview.height, self._preview.width, 3)
        self._seq += 1
        self.stats.record(ts_ns, self._interval_ns)
        return Frame(image=img, timestamp_ns=ts_ns, seq=self._seq,
                     source_id=self.source_id)

    def _maybe_restart(self) -> None:
        """切断からの自動復帰。RTMP は送出側の都合でいつでも切れる。"""
        if not self._auto_restart or self._want_stop.is_set():
            return
        with self._lock_proc:
            old, self._proc = self._proc, None
        if old is not None:
            try:
                old.kill()
            except Exception:
                pass
        if self._want_stop.wait(self.RESTART_DELAY_S):
            return
        self._restarts += 1
        try:
            with self._lock_proc:
                self._spawn()
        except Exception:
            self._set_state(SourceState.ERROR)


# ---------------------------------------------------------------- 補助

_SIZE_RE = re.compile(r",\s*(\d{2,5})x(\d{2,5})")


def probe_stream(url: str, timeout_s: float = 10.0,
                 ffprobe: str = "ffprobe") -> dict:
    """接続可否と実際の解像度/fps/コーデックを確認する。

    設定画面やセットアップ時の疎通確認に使う。届かない URL を
    そのまま本番構成に入れてしまうのを防ぐ。
    """
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,pix_fmt",
        "-of", "default=noprint_wrappers=1",
        "-rtmp_live", "live" if url.startswith("rtmp") else "any",
        url,
    ]
    if not url.startswith("rtmp"):
        cmd = [c for c in cmd if c not in ("-rtmp_live", "live", "any")]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"{timeout_s:g} 秒以内に応答がありません"}
    except FileNotFoundError:
        return {"ok": False, "error": "ffprobe が見つかりません"}

    if p.returncode != 0:
        return {"ok": False,
                "error": p.stderr.decode("utf-8", "replace").strip() or "接続失敗"}

    info: dict = {"ok": True}
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        k, _, v = line.partition("=")
        info[k.strip()] = v.strip()
    if "r_frame_rate" in info and "/" in info["r_frame_rate"]:
        num, den = info["r_frame_rate"].split("/")
        try:
            info["fps"] = round(int(num) / int(den), 2) if int(den) else None
        except ValueError:
            info["fps"] = None
    return info
