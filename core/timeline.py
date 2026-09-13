"""リングバッファと共通タイムライン。

VAR の肝は「録画を止めずに過去を見られる」こと。そのために
取得（書き込み）と再生（読み出し）を完全に分離し、両者はこのバッファ
だけを介してやり取りする。取得スレッドは再生の都合で待たされてはならない。

カメラ間の同期もここで行う。UVC にハード同期が無い以上、各フレームの
受信時刻を唯一の共通軸として「指定時刻に最も近いフレーム」を各カメラから
引くのが、現実的に達成できる最良の揃え方になる。
"""

from __future__ import annotations

import bisect
import threading
from typing import Optional

from .frame_source import Frame, FrameSource, SourceState


class FrameBuffer:
    """1カメラ分の時刻付きリングバッファ。

    保持長で古いフレームを捨てる。FHD30 の BGR は 1 フレーム約 6MB なので、
    メモリ常駐はプレビュー用の縮小フレームに限る前提（原寸の長時間保持は
    ディスク上のセグメント録画が担当する）。
    """

    def __init__(self, source_id: str, max_seconds: float = 120.0) -> None:
        self.source_id = source_id
        self.max_ns = int(max_seconds * 1e9)
        self._ts: list[int] = []
        self._frames: list[Frame] = []
        self._lock = threading.RLock()

    def append(self, frame: Frame) -> None:
        with self._lock:
            # 時刻は単調増加する前提だが、万一戻った場合は挿入位置を探す
            if self._ts and frame.timestamp_ns < self._ts[-1]:
                i = bisect.bisect_left(self._ts, frame.timestamp_ns)
                self._ts.insert(i, frame.timestamp_ns)
                self._frames.insert(i, frame)
            else:
                self._ts.append(frame.timestamp_ns)
                self._frames.append(frame)

            cutoff = self._ts[-1] - self.max_ns
            drop = bisect.bisect_left(self._ts, cutoff)
            if drop > 0:
                del self._ts[:drop]
                del self._frames[:drop]

    def nearest(self, ts_ns: int) -> Optional[Frame]:
        """指定時刻に最も近いフレーム。4分割の同期表示はこれで揃える。"""
        with self._lock:
            if not self._ts:
                return None
            i = bisect.bisect_left(self._ts, ts_ns)
            if i == 0:
                return self._frames[0]
            if i >= len(self._ts):
                return self._frames[-1]
            before, after = self._ts[i - 1], self._ts[i]
            return self._frames[i - 1] if (ts_ns - before) <= (after - ts_ns) else self._frames[i]

    def step(self, ts_ns: int, delta: int) -> Optional[Frame]:
        """コマ送り。現在時刻の前後 delta 枚目のフレームを返す。

        「指定時刻 + 1/fps 秒」で引くのではなく実フレームを辿るのが重要。
        フレーム落ちがあっても必ず隣の実フレームに移動でき、
        「送ったのに絵が変わらない」を防げる。
        """
        with self._lock:
            if not self._ts:
                return None
            i = bisect.bisect_left(self._ts, ts_ns)
            if i < len(self._ts) and self._ts[i] == ts_ns:
                j = i + delta
            elif delta > 0:
                j = i + delta - 1
            else:
                j = i + delta
            j = max(0, min(len(self._frames) - 1, j))
            return self._frames[j]

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def span(self) -> Optional[tuple[int, int]]:
        with self._lock:
            return (self._ts[0], self._ts[-1]) if self._ts else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._ts)


class CaptureThread(threading.Thread):
    """取得専用スレッド。再生側の都合で絶対にブロックさせない。"""

    def __init__(self, source: FrameSource, buffer: FrameBuffer,
                 downscale: Optional[tuple[int, int]] = None) -> None:
        super().__init__(name=f"capture-{source.source_id}", daemon=True)
        self.source = source
        self.buffer = buffer
        self.downscale = downscale
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        import cv2

        while not self._stop.is_set():
            frame = self.source.read()
            if frame is None:
                continue
            if self.downscale is not None:
                img = cv2.resize(frame.image, self.downscale, interpolation=cv2.INTER_AREA)
                frame = Frame(img, frame.timestamp_ns, frame.seq, frame.source_id)
            self.buffer.append(frame)


class Timeline:
    """全カメラのバッファ束と、それらに共通の再生ヘッド。"""

    LIVE = None  # 再生ヘッドが LIVE のときは常に最新を追う

    def __init__(self, sources: list[FrameSource], max_seconds: float = 120.0,
                 downscale: Optional[tuple[int, int]] = None) -> None:
        self.sources = sources
        self.buffers = {s.source_id: FrameBuffer(s.source_id, max_seconds) for s in sources}
        self._threads = [
            CaptureThread(s, self.buffers[s.source_id], downscale) for s in sources
        ]
        self.playhead_ns: Optional[int] = self.LIVE
        self.rate: float = 1.0
        self.paused: bool = False
        self.tags: list[dict] = []

    # --- 取得の開始/停止 ---
    def start(self) -> None:
        for s in self.sources:
            if s.state is SourceState.STOPPED:
                s.start()
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        for t in self._threads:
            t.stop()
        for t in self._threads:
            t.join(timeout=2.0)
        for s in self.sources:
            s.stop()

    # --- 時間軸 ---
    def span(self) -> Optional[tuple[int, int]]:
        """全カメラを通じて再生可能な時間範囲。"""
        spans = [b.span() for b in self.buffers.values()]
        spans = [s for s in spans if s]
        if not spans:
            return None
        return (min(s[0] for s in spans), max(s[1] for s in spans))

    @property
    def is_live(self) -> bool:
        return self.playhead_ns is self.LIVE

    def current_ns(self) -> Optional[int]:
        if self.playhead_ns is not self.LIVE:
            return self.playhead_ns
        sp = self.span()
        return sp[1] if sp else None

    def go_live(self) -> None:
        self.playhead_ns = self.LIVE
        self.paused = False

    def seek(self, ts_ns: int) -> None:
        sp = self.span()
        if sp:
            ts_ns = max(sp[0], min(sp[1], ts_ns))
        self.playhead_ns = ts_ns

    def advance(self, elapsed_s: float) -> None:
        """再生の進行。LIVE 中や一時停止中は何もしない。"""
        if self.is_live or self.paused or self.rate == 0.0:
            return
        cur = self.current_ns()
        if cur is None:
            return
        nxt = cur + int(elapsed_s * self.rate * 1e9)
        sp = self.span()
        if sp and nxt >= sp[1]:
            self.go_live()   # 現在に追いついたらライブに復帰する
            return
        self.seek(nxt)

    def step_frames(self, delta: int, ref_source: Optional[str] = None) -> None:
        """コマ送り。基準カメラの実フレームを辿って移動する。"""
        cur = self.current_ns()
        if cur is None:
            return
        ref = ref_source or next(iter(self.buffers))
        frame = self.buffers[ref].step(cur, delta)
        if frame is not None:
            self.paused = True
            self.seek(frame.timestamp_ns)

    # --- 表示 ---
    def frames_at_playhead(self) -> dict[str, Optional[Frame]]:
        """各カメラから再生ヘッドに最も近いフレームを引く（ソフト同期）。"""
        if self.is_live:
            return {sid: b.latest() for sid, b in self.buffers.items()}
        ts = self.current_ns()
        if ts is None:
            return {sid: None for sid in self.buffers}
        return {sid: b.nearest(ts) for sid, b in self.buffers.items()}

    # --- ライブタギング ---
    def add_tag(self, label: str = "") -> dict:
        """録画を止めずに現在時刻へマークを打つ。"""
        ts = self.current_ns()
        tag = {"ts_ns": ts, "label": label or f"TAG {len(self.tags) + 1}"}
        self.tags.append(tag)
        self.tags.sort(key=lambda t: (t["ts_ns"] is None, t["ts_ns"]))
        return tag
