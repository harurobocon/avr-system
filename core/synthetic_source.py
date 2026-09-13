"""カメラ不要で UI を開発するための合成映像ソース。

フレーム番号と時刻を画面に焼き込むので、コマ送りが本当に 1 フレームずつ
進退しているかを目視で検証できる（これが無いと「それっぽく動く」だけの
コマ送りを見抜けない）。ドロップ・フリーズ・切断を注入できるため、
異常系 UI も実機を待たずに詰められる。
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from .frame_source import Capabilities, Frame, FrameSource, SourceState

# カメラごとに色を変え、4分割表示でどれがどれか一目で分かるようにする
_PALETTE = [(40, 60, 160), (40, 140, 60), (150, 90, 30), (120, 40, 130)]


class SyntheticSource(FrameSource):
    """指定 fps で合成フレームを生成する。

    fault injection:
      freeze(sec)     … 同じフレームを返し続ける（映像が固まるカメラの再現）
      disconnect()    … read() が None を返す（ケーブル抜けの再現）
      drop_rate       … 指定確率でフレームを飛ばす（帯域不足の再現）
    """

    def __init__(
        self,
        source_id: str,
        index: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        drop_rate: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(source_id)
        self._caps = Capabilities(width, height, fps, "SYNTH", name=f"Synthetic #{index}")
        self._index = index
        self._drop_rate = drop_rate
        self._rng = np.random.default_rng(seed if seed is not None else index)
        self._seq = 0
        self._start_ns = 0
        self._frozen_until_ns = 0
        self._frozen_frame: Optional[np.ndarray] = None
        self._disconnected = threading.Event()
        self._interval_ns = int(1e9 / fps)
        self._base = self._make_base()

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    def _make_base(self) -> np.ndarray:
        """背景は一度だけ作って使い回す（毎フレーム生成すると CPU 計測が濁る）。"""
        h, w = self._caps.height, self._caps.width
        img = np.zeros((h, w, 3), np.uint8)
        img[:, :] = _PALETTE[self._index % len(_PALETTE)]
        # グリッド線：ズーム/パンの倍率と位置を目視確認するための基準
        step = max(h // 12, 1)
        img[::step, :] = (255, 255, 255)
        img[:, ::step] = (255, 255, 255)
        return img

    def start(self) -> None:
        self._start_ns = time.perf_counter_ns()
        self._seq = 0
        self._disconnected.clear()
        self._set_state(SourceState.RUNNING)

    def stop(self) -> None:
        self._set_state(SourceState.STOPPED)

    def disconnect(self) -> None:
        self._disconnected.set()
        self._set_state(SourceState.DISCONNECTED)

    def reconnect(self) -> None:
        self._disconnected.clear()
        self._set_state(SourceState.RUNNING)

    def freeze(self, seconds: float) -> None:
        self._frozen_until_ns = time.perf_counter_ns() + int(seconds * 1e9)
        self._set_state(SourceState.FROZEN)

    def _render(self, seq: int, ts_ns: int) -> np.ndarray:
        img = self._base.copy()
        h, w = img.shape[:2]
        scale = h / 540.0
        elapsed = (ts_ns - self._start_ns) / 1e9
        lines = [
            f"CAM {self._index}  {self.source_id}",
            f"FRAME {seq:07d}",
            f"T {elapsed:9.3f}s",
        ]
        y = int(60 * scale)
        for text in lines:
            cv2.putText(img, text, (int(40 * scale), y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2 * scale, (255, 255, 255), max(int(3 * scale), 2), cv2.LINE_AA)
            y += int(60 * scale)
        # 一定速で動くマーカー。コマ送りで 1 フレーム分だけ動くことを目視できる
        cx = int((elapsed * 0.25 % 1.0) * w)
        cv2.circle(img, (cx, h - int(80 * scale)), int(25 * scale), (255, 255, 255), -1)
        return img

    def read(self) -> Optional[Frame]:
        if self._disconnected.is_set():
            time.sleep(self._interval_ns / 1e9)
            self.stats.read_failures += 1
            return None

        # 実カメラ同様、次フレームの到来時刻までブロックする
        target_ns = self._start_ns + (self._seq + 1) * self._interval_ns
        now_ns = time.perf_counter_ns()
        if target_ns > now_ns:
            time.sleep((target_ns - now_ns) / 1e9)
        self._seq += 1
        ts_ns = time.perf_counter_ns()

        if self._drop_rate > 0 and self._rng.random() < self._drop_rate:
            return None  # 欠落。stats はドロップとして間隔から推定される

        now_ns = time.perf_counter_ns()
        if now_ns < self._frozen_until_ns and self._frozen_frame is not None:
            img = self._frozen_frame
        else:
            if now_ns >= self._frozen_until_ns and self.state is SourceState.FROZEN:
                self._set_state(SourceState.RUNNING)
            img = self._render(self._seq, ts_ns)
            self._frozen_frame = img

        self.stats.record(ts_ns, self._interval_ns)
        return Frame(image=img, timestamp_ns=ts_ns, seq=self._seq, source_id=self.source_id)
