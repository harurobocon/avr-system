"""フレームソース抽象。

UI（トラックA）とカメラ実証（トラックB）の合流点。
UI は FrameSource にしか依存しないため、実カメラの実力が確定する前でも
SyntheticSource / FileSource を相手に開発を進められる。
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class SourceState(Enum):
    """UI の状態表示（要件5章「カメラ/録画の状態表示」）に直結する。"""

    STOPPED = "stopped"
    RUNNING = "running"
    FROZEN = "frozen"        # フレームは来るが更新が止まっている
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass(frozen=True)
class Capabilities:
    """カメラごとのプロファイル。混在機材が前提なので台数分それぞれ持つ。"""

    width: int
    height: int
    fps: float
    pixel_format: str = "MJPG"
    name: str = ""

    def __str__(self) -> str:
        return f"{self.width}x{self.height}@{self.fps:g} {self.pixel_format}"


@dataclass(frozen=True)
class Frame:
    """1フレームと、それがいつ届いたか。

    timestamp_ns はソフト同期の基準。UVC には外部トリガが無いため、
    受信時刻を唯一の共通軸として 4 台を揃える。
    monotonic 系（time.perf_counter_ns 相当）で採る。
    """

    image: np.ndarray
    timestamp_ns: int
    seq: int
    source_id: str = ""

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ns / 1e9


@dataclass
class SourceStats:
    """ベンチと UI の異常表示で共用する統計。"""

    frames_read: int = 0
    read_failures: int = 0
    estimated_drops: int = 0
    first_timestamp_ns: Optional[int] = None
    last_timestamp_ns: Optional[int] = None
    _intervals_ns: list[int] = field(default_factory=list)

    def record(self, ts_ns: int, nominal_interval_ns: int) -> None:
        if self.first_timestamp_ns is None:
            self.first_timestamp_ns = ts_ns
        elif self.last_timestamp_ns is not None:
            gap = ts_ns - self.last_timestamp_ns
            self._intervals_ns.append(gap)
            # 受信間隔が公称間隔の 1.5 倍を超えたら、その分だけ落ちたとみなす。
            # UVC はフレーム番号を返さないので、間隔からの推定が唯一の手段。
            if nominal_interval_ns > 0 and gap > nominal_interval_ns * 1.5:
                self.estimated_drops += int(round(gap / nominal_interval_ns)) - 1
        self.last_timestamp_ns = ts_ns
        self.frames_read += 1

    @property
    def elapsed_s(self) -> float:
        if self.first_timestamp_ns is None or self.last_timestamp_ns is None:
            return 0.0
        return (self.last_timestamp_ns - self.first_timestamp_ns) / 1e9

    @property
    def effective_fps(self) -> float:
        e = self.elapsed_s
        return (self.frames_read - 1) / e if e > 0 else 0.0

    @property
    def drop_rate(self) -> float:
        total = self.frames_read + self.estimated_drops
        return self.estimated_drops / total if total else 0.0

    def interval_percentiles_ms(self) -> dict[str, float]:
        if not self._intervals_ns:
            return {}
        a = np.asarray(self._intervals_ns, dtype=np.float64) / 1e6
        return {
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
        }

    def to_dict(self) -> dict:
        return {
            "frames_read": self.frames_read,
            "read_failures": self.read_failures,
            "estimated_drops": self.estimated_drops,
            "elapsed_s": round(self.elapsed_s, 3),
            "effective_fps": round(self.effective_fps, 2),
            "drop_rate": round(self.drop_rate, 5),
            "interval_ms": {k: round(v, 2) for k, v in self.interval_percentiles_ms().items()},
        }


class FrameSource(ABC):
    """すべての映像入力の共通インターフェース。"""

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self._state = SourceState.STOPPED
        self._lock = threading.Lock()
        self.stats = SourceStats()

    @property
    def state(self) -> SourceState:
        with self._lock:
            return self._state

    def _set_state(self, s: SourceState) -> None:
        with self._lock:
            self._state = s

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities:
        ...

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """次のフレームを返す。取得できなければ None。ブロッキング。"""

    def __enter__(self) -> "FrameSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
