"""4台同時取得の帯域/CPU 成立性を実測する（トラックB-2、最優先リスク）。

ここが崩れると全体設計が変わるので、台数を段階的に増やしながら
「どこで破綻するか」を数値で押さえる。

  python poc/capture_bench.py --cameras 0 --duration 20
  python poc/capture_bench.py --cameras 0,1,2,3 --duration 120
  python poc/capture_bench.py --cameras 0,1,2,3 --duration 120 --record
  python poc/capture_bench.py --cameras 0,1,2,3 --ramp --duration 30
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from core.frame_source import SourceState  # noqa: E402
from core.uvc_source import UvcSource  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
# ドロップ率の合否ライン（計画の判定基準）
DROP_RATE_THRESHOLD = 0.01


class CaptureWorker(threading.Thread):
    """1カメラ = 1スレッド。read() がブロッキングなので分離は必須。"""

    def __init__(self, source: UvcSource, duration_s: float,
                 writer: "SegmentWriter | None" = None) -> None:
        super().__init__(name=f"cap-{source.source_id}", daemon=True)
        self.source = source
        self.duration_s = duration_s
        self.writer = writer
        self.error: Exception | None = None
        self.timestamps_ns: list[int] = []
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            deadline = time.perf_counter() + self.duration_s
            while not self._stop.is_set() and time.perf_counter() < deadline:
                frame = self.source.read()
                if frame is None:
                    continue
                self.timestamps_ns.append(frame.timestamp_ns)
                if self.writer is not None:
                    self.writer.write(frame.image)
        except Exception as e:  # スレッド内例外を握り潰さない
            self.error = e
        finally:
            if self.writer is not None:
                self.writer.close()


class SegmentWriter:
    """再エンコードせずに書き出す。

    5800U で H.264 を 4 本同時エンコードするとレビュー再生用の CPU が
    残らない。ここでは MJPEG のまま AVI コンテナに入れる（実質 remux）。
    """

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        import cv2

        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._w = cv2.VideoWriter(
            str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps or 30.0, (width, height)
        )
        if not self._w.isOpened():
            raise RuntimeError(f"書き出しを開始できません: {path}")
        self.frames = 0

    def write(self, img: np.ndarray) -> None:
        self._w.write(img)
        self.frames += 1

    def close(self) -> None:
        if self._w is not None:
            self._w.release()
            self._w = None


class ResourceSampler(threading.Thread):
    """取得中の CPU / メモリ / ディスク書き込みを 0.5 秒ごとに採る。"""

    def __init__(self, interval_s: float = 0.5) -> None:
        super().__init__(name="res-sampler", daemon=True)
        self.interval_s = interval_s
        self.cpu: list[float] = []
        self.proc_cpu: list[float] = []
        self.rss_mb: list[float] = []
        self.write_mb: list[float] = []
        self._stop = threading.Event()
        self._proc = psutil.Process()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        self._proc.cpu_percent(None)
        psutil.cpu_percent(None)
        last_io = psutil.disk_io_counters()
        last_t = time.perf_counter()
        while not self._stop.wait(self.interval_s):
            self.cpu.append(psutil.cpu_percent(None))
            # プロセス CPU は論理コア数で正規化し、システム全体と同じ土俵で見る
            self.proc_cpu.append(self._proc.cpu_percent(None) / (psutil.cpu_count() or 1))
            self.rss_mb.append(self._proc.memory_info().rss / 1e6)
            io = psutil.disk_io_counters()
            now = time.perf_counter()
            if io and last_io and now > last_t:
                self.write_mb.append((io.write_bytes - last_io.write_bytes) / 1e6 / (now - last_t))
            last_io, last_t = io, now

    def summary(self) -> dict:
        def stat(a: list[float]) -> dict:
            if not a:
                return {}
            return {"mean": round(statistics.fmean(a), 1), "max": round(max(a), 1)}

        return {
            "cpu_percent_total": stat(self.cpu),
            "cpu_percent_process": stat(self.proc_cpu),
            "process_rss_mb": stat(self.rss_mb),
            "disk_write_mb_per_s": stat(self.write_mb),
        }


def cross_camera_skew_ms(workers: list[CaptureWorker]) -> dict:
    """カメラ間のフレーム時刻ずれ。

    基準カメラの各フレーム時刻に対し、他カメラの最近傍フレーム時刻との差を取る。
    これが「Webカメラで現実的に達成できる同期精度」の実測値になり、
    要件定義書の非機能要件にそのまま入る。
    """
    active = [w for w in workers if len(w.timestamps_ns) > 1]
    if len(active) < 2:
        return {"note": "カメラが2台未満のため同期ずれは測定できません"}

    ref = active[0]
    ref_ts = np.asarray(ref.timestamps_ns, dtype=np.int64)
    out: dict = {"reference": ref.source.source_id, "pairs": {}}
    worst = 0.0
    for w in active[1:]:
        other = np.asarray(sorted(w.timestamps_ns), dtype=np.int64)
        idx = np.searchsorted(other, ref_ts)
        idx = np.clip(idx, 1, len(other) - 1)
        left, right = other[idx - 1], other[idx]
        nearest = np.where(np.abs(ref_ts - left) <= np.abs(right - ref_ts), left, right)
        d = np.abs(ref_ts - nearest) / 1e6  # ms
        out["pairs"][w.source.source_id] = {
            "p50_ms": round(float(np.percentile(d, 50)), 2),
            "p95_ms": round(float(np.percentile(d, 95)), 2),
            "max_ms": round(float(d.max()), 2),
        }
        worst = max(worst, float(np.percentile(d, 95)))
    out["worst_p95_ms"] = round(worst, 2)
    return out


def run_once(indices: list[int], duration_s: float, width: int, height: int,
             fps: float, record: bool, outdir: Path,
             exposure: int | None) -> dict:
    sources: list[UvcSource] = []
    workers: list[CaptureWorker] = []

    for i in indices:
        src = UvcSource(f"cam{i}", index=i, width=width, height=height, fps=fps,
                        exposure=exposure)
        try:
            src.start()
        except RuntimeError as e:
            for s in sources:
                s.stop()
            return {"error": str(e), "cameras": indices}
        sources.append(src)

    negotiated = {s.source_id: str(s.capabilities) for s in sources}
    mismatched = [
        s.source_id for s in sources
        if s.capabilities.width != width or s.capabilities.height != height
    ]

    try:
        for s in sources:
            writer = None
            if record:
                c = s.capabilities
                writer = SegmentWriter(
                    outdir / f"{s.source_id}.avi", c.width, c.height, c.fps)
            workers.append(CaptureWorker(s, duration_s, writer))

        sampler = ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=duration_s + 30)
        wall = time.perf_counter() - t0
        sampler.stop()
        sampler.join(timeout=2)
    finally:
        for s in sources:
            s.stop()

    per_cam = {}
    for w in workers:
        d = w.source.stats.to_dict()
        d["negotiated"] = str(w.source.capabilities)
        d["state"] = w.source.state.value
        if w.error:
            d["error"] = repr(w.error)
        if w.writer is not None:
            d["written_frames"] = w.writer.frames
            d["file"] = str(w.writer.path)
            d["file_mb"] = round(w.writer.path.stat().st_size / 1e6, 1) \
                if w.writer.path.exists() else 0.0
        per_cam[w.source.source_id] = d

    drops = [d["drop_rate"] for d in per_cam.values()]
    return {
        "cameras": indices,
        "camera_count": len(indices),
        "requested": f"{width}x{height}@{fps:g} MJPG",
        "negotiated": negotiated,
        "resolution_mismatch": mismatched,
        "record": record,
        "exposure": exposure,
        "duration_s": round(wall, 2),
        "per_camera": per_cam,
        "worst_drop_rate": round(max(drops), 5) if drops else None,
        "pass": bool(drops) and max(drops) < DROP_RATE_THRESHOLD and not mismatched,
        "resources": sampler.summary(),
        "sync_skew": cross_camera_skew_ms(workers),
    }


def print_result(r: dict) -> None:
    if "error" in r:
        print(f"  ERROR: {r['error']}")
        return
    n = r["camera_count"]
    verdict = "PASS" if r["pass"] else "FAIL"
    print(f"\n--- {n}台 / {r['requested']} / record={r['record']} -> {verdict} ---")
    if r["resolution_mismatch"]:
        print(f"  !! 要求解像度が通らなかったカメラ: {', '.join(r['resolution_mismatch'])}")
    for cid, d in r["per_camera"].items():
        line = (f"  {cid}: {d['negotiated']:24s} fps={d['effective_fps']:5.2f} "
                f"drop={d['drop_rate'] * 100:5.2f}% frames={d['frames_read']} "
                f"fail={d['read_failures']}")
        if "file_mb" in d:
            line += f" file={d['file_mb']}MB"
        print(line)
        iv = d.get("interval_ms") or {}
        if iv:
            print(f"        受信間隔 p50={iv.get('p50')}ms p95={iv.get('p95')}ms "
                  f"max={iv.get('max')}ms")
    res = r["resources"]
    cpu = res.get("cpu_percent_total", {})
    wr = res.get("disk_write_mb_per_s", {})
    pcpu = res.get("cpu_percent_process", {})
    print(f"  CPU  system mean={cpu.get('mean')}% max={cpu.get('max')}%  /  "
          f"process mean={pcpu.get('mean')}% max={pcpu.get('max')}%")
    if wr:
        print(f"  Disk write mean={wr.get('mean')}MB/s max={wr.get('max')}MB/s")
    sk = r["sync_skew"]
    if "worst_p95_ms" in sk:
        print(f"  カメラ間ずれ(基準 {sk['reference']}) worst p95={sk['worst_p95_ms']}ms")
        for cid, p in sk["pairs"].items():
            print(f"        {cid}: p50={p['p50_ms']}ms p95={p['p95_ms']}ms max={p['max_ms']}ms")
    elif "note" in sk:
        print(f"  {sk['note']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="4台同時取得の成立性ベンチマーク")
    ap.add_argument("--cameras", default="0", help="OpenCV インデックスをカンマ区切りで (例 0,1,2,3)")
    ap.add_argument("--duration", type=float, default=20.0, help="計測秒数 (試合想定なら120)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--record", action="store_true", help="remux 書き出しを併用する")
    ap.add_argument("--ramp", action="store_true",
                    help="1台→N台と増やして、どこで破綻するかを見る")
    ap.add_argument("--exposure", type=int, default=-7,
                    help="手動露出 2^n 秒。既定 -7 (約7.8ms)。オート露出は fps が落ちる")
    ap.add_argument("--auto-exposure", action="store_true",
                    help="露出をカメラ任せにする（比較用。fps が落ちることの確認）")
    ap.add_argument("--tag", default="", help="結果ファイル名に付ける識別子")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    indices = [int(x) for x in args.cameras.split(",") if x.strip() != ""]
    if not indices:
        print("カメラインデックスを指定してください")
        return 2

    exposure = None if args.auto_exposure else args.exposure

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = RESULTS / f"rec-{stamp}"
    sets = [indices[: i + 1] for i in range(len(indices))] if args.ramp else [indices]

    print(f"環境: CPU 論理 {psutil.cpu_count()} コア / "
          f"RAM {psutil.virtual_memory().total / 1e9:.1f} GB")
    print(f"計測: {args.duration:g}秒 x {len(sets)} 条件 / "
          f"露出={'auto' if exposure is None else f'manual 2^{exposure}s'}")

    runs = []
    for s in sets:
        print(f"\n>>> {len(s)}台 {s} を {args.duration:g} 秒取得中...")
        r = run_once(s, args.duration, args.width, args.height, args.fps,
                     args.record, outdir, exposure)
        print_result(r)
        runs.append(r)

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"-{args.tag}" if args.tag else ""
    out = RESULTS / f"bench-{stamp}{tag}.json"
    out.write_text(json.dumps({
        "timestamp": stamp,
        "host": {
            "cpu_logical": psutil.cpu_count(),
            "cpu_physical": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        },
        "args": vars(args),
        "runs": runs,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n保存: {out}")

    ok = [r for r in runs if r.get("pass")]
    print(f"\n結論: {len(ok)}/{len(runs)} 条件が判定基準"
          f"(ドロップ率 < {DROP_RATE_THRESHOLD * 100:g}%)を満たしました")
    return 0 if runs and runs[-1].get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
