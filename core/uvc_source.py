"""実 UVC カメラからの取得（トラックB）。

UVC にはハードウェア同期トリガが無いため、フレームに付与できる時刻は
「アプリケーションが受け取った時刻」しかない。これが 4 台をそろえる
唯一の共通軸になる。read() は必ず専用スレッドから呼ぶこと
（cv2 の read() はブロッキングで、UI スレッドで呼ぶと固まる）。
"""

from __future__ import annotations

import time
from typing import Optional

import cv2

from .frame_source import Capabilities, Frame, FrameSource, SourceState


class UvcSource(FrameSource):
    def __init__(
        self,
        source_id: str,
        index: int,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        fourcc: str = "MJPG",
        backend: int = cv2.CAP_DSHOW,
        buffer_size: int = 1,
        exposure: Optional[int] = -7,
    ) -> None:
        super().__init__(source_id)
        self._index = index
        self._req = Capabilities(width, height, fps, fourcc, name=f"UVC #{index}")
        self._caps = self._req
        self._backend = backend
        self._buffer_size = buffer_size
        self._exposure = exposure
        self._cap: Optional[cv2.VideoCapture] = None
        self._seq = 0
        self._interval_ns = int(1e9 / fps) if fps > 0 else 0

    @property
    def capabilities(self) -> Capabilities:
        """start() 後は実際にネゴシエートされた値を返す（要求値とは限らない）。"""
        return self._caps

    @property
    def requested(self) -> Capabilities:
        return self._req

    def start(self) -> None:
        cap = cv2.VideoCapture(self._index, self._backend)
        if not cap.isOpened():
            cap.release()
            self._set_state(SourceState.ERROR)
            raise RuntimeError(f"カメラ index={self._index} を開けません")

        # 順序が重要: FOURCC を先に立てないと、非圧縮のまま解像度が確定して
        # FHD が通らない（多くの UVC カメラで YUY2 は VGA 止まり）。
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*self._req.pixel_format))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._req.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._req.height)
        cap.set(cv2.CAP_PROP_FPS, self._req.fps)
        # 内部バッファを最小にする。溜めると「今」より古いフレームが返り、
        # ライブ監視としての遅延になる。
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)
        except cv2.error:
            pass

        # 露出の手動固定。これは任意設定ではなく必須。
        # 実測（ELECOM 2MP Webcam, FHD MJPEG）:
        #   オート露出        10.0 fps  <- 暗所で露光を延ばし fps が 1/3 に落ちる
        #   手動 2^-5 (31ms)  23.3 fps
        #   手動 2^-6 (16ms)  28.8 fps
        #   手動 2^-7 (7.8ms) 29.2 fps
        # 加えて、短い露光はモーションブラーを抑えるため接触判定の精度に直結する。
        # 代償として暗くなるので、会場照度が足りることが前提になる。
        if self._exposure is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DirectShow: 0.25=manual, 0.75=auto
            cap.set(cv2.CAP_PROP_EXPOSURE, self._exposure)

        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS)) or self._req.fps
        self._caps = Capabilities(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=actual_fps,
            pixel_format="".join(chr((fourcc >> (8 * k)) & 0xFF) for k in range(4)).strip(),
            name=self._req.name,
        )
        self._interval_ns = int(1e9 / actual_fps) if actual_fps > 0 else 0
        self._cap = cap
        self._seq = 0
        self._set_state(SourceState.RUNNING)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._set_state(SourceState.STOPPED)

    def read(self) -> Optional[Frame]:
        cap = self._cap
        if cap is None:
            return None
        ok, img = cap.read()
        # タイムスタンプは read() 復帰直後に採る。露光時刻そのものではないが、
        # 全カメラで同じ採り方をする限り相対ずれの指標としては一貫している。
        ts_ns = time.perf_counter_ns()
        if not ok or img is None:
            self.stats.read_failures += 1
            # 連続失敗は切断とみなす（UI の状態表示はこれを見る）
            if self.stats.read_failures >= 10:
                self._set_state(SourceState.DISCONNECTED)
            return None
        if self.state is SourceState.DISCONNECTED:
            self._set_state(SourceState.RUNNING)
        self._seq += 1
        self.stats.record(ts_ns, self._interval_ns)
        return Frame(image=img, timestamp_ns=ts_ns, seq=self._seq, source_id=self.source_id)
