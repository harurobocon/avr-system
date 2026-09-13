"""接続カメラの能力と USB トポロジを列挙する（トラックB-1）。

型番不明・複数メーカー混在が前提なので、「手元の4台で何ができるか」を
まず事実として確定させる。ここで MJPEG FHD30 に対応していない個体が
見つかれば、それ自体が要件定義書の制約になる。

  python poc/enumerate_cameras.py
  python poc/enumerate_cameras.py --probe   # OpenCV で実際に開いて確認
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


@dataclass
class Mode:
    pixel_format: str
    width: int
    height: int
    fps_min: float
    fps_max: float


@dataclass
class Camera:
    name: str
    modes: list[Mode] = field(default_factory=list)
    opencv_index: int | None = None
    opencv_probe: dict | None = None

    def best_mjpeg(self) -> Mode | None:
        mj = [m for m in self.modes if "mjpeg" in m.pixel_format.lower()]
        if not mj:
            return None
        return max(mj, key=lambda m: (m.width * m.height, m.fps_max))

    def supports_fhd30_mjpeg(self) -> bool:
        return any(
            "mjpeg" in m.pixel_format.lower()
            and m.width >= 1920 and m.height >= 1080 and m.fps_max >= 29.0
            for m in self.modes
        )


def _run(cmd: list[str]) -> str:
    """ffmpeg は情報を stderr に出すので統合して受ける。

    ffmpeg は UTF-8 で出力するが、Windows の既定ロケールは cp932 なので
    text=True に任せるとデバイス名が化ける（結果として仮想カメラの
    名前判定にも失敗する）。バイトで受けて明示的に UTF-8 で復号する。
    """
    p = subprocess.run(cmd, capture_output=True)
    return (p.stdout or b"").decode("utf-8", "replace") + \
           (p.stderr or b"").decode("utf-8", "replace")


# 仮想カメラは物理USB帯域を消費しないので、実機検証の対象から外す
_VIRTUAL_RE = re.compile(r"仮想カメラ|virtual|NVIDIA Broadcast|OBS", re.I)


def list_dshow_devices(include_virtual: bool = False) -> list[str]:
    """DirectShow のビデオ入力名を返す。

    ffmpeg 9.x は `"名前" (video)` 形式で、8.x 以前のような
    「DirectShow video devices」見出しを出さない。両方に対応する。
    """
    out = _run(["ffmpeg", "-hide_banner", "-f", "dshow", "-list_devices", "true", "-i", "dummy"])
    names: list[str] = []
    section: str | None = None
    for line in out.splitlines():
        if "DirectShow video devices" in line:
            section = "video"
            continue
        if "DirectShow audio devices" in line:
            section = "audio"
            continue
        if "Alternative name" in line:
            continue
        m = re.search(r'"([^"]+)"', line)
        if not m:
            continue
        name = m.group(1)
        tag = re.search(r"\((video|audio|none)\)\s*$", line.strip())
        if tag:                       # ffmpeg 9.x 形式
            if tag.group(1) != "video":
                continue
        elif section != "video":      # 旧形式
            continue
        if not include_virtual and _VIRTUAL_RE.search(name):
            continue
        names.append(name)
    return names


# 例: pixel_format=yuyv422  min s=640x480 fps=5 max s=640x480 fps=30
_OPT_RE = re.compile(
    r"(?:pixel_format=(?P<pix>\S+)|vcodec=(?P<vcodec>\S+)).*?"
    r"min s=(?P<w>\d+)x(?P<h>\d+) fps=(?P<fmin>[\d.]+)\s+"
    r"max s=(?P<W>\d+)x(?P<H>\d+) fps=(?P<fmax>[\d.]+)"
)


def list_modes(device_name: str) -> list[Mode]:
    out = _run([
        "ffmpeg", "-hide_banner", "-f", "dshow", "-list_options", "true",
        "-i", "video=" + device_name,
    ])
    modes: list[Mode] = []
    for line in out.splitlines():
        m = _OPT_RE.search(line)
        if not m:
            continue
        # MJPEG は vcodec=mjpeg として現れ、非圧縮は pixel_format=yuyv422 等で現れる
        fmt = m.group("vcodec") or m.group("pix") or "?"
        modes.append(Mode(
            pixel_format=fmt,
            width=int(m.group("W")), height=int(m.group("H")),
            fps_min=float(m.group("fmin")), fps_max=float(m.group("fmax")),
        ))
    seen: set = set()
    uniq: list[Mode] = []
    for md in modes:
        key = (md.pixel_format, md.width, md.height, md.fps_max)
        if key not in seen:
            seen.add(key)
            uniq.append(md)
    return uniq


_PS_TOPOLOGY = r"""
$ErrorActionPreference='SilentlyContinue'
Get-CimInstance Win32_USBController | ForEach-Object {
  $c = $_
  $kids = Get-CimAssociatedInstance -InputObject $c -ResultClassName Win32_PnPEntity |
          Where-Object { $_.Name } | Select-Object -ExpandProperty Name
  [PSCustomObject]@{ controller = $c.Name; devices = @($kids) }
} | ConvertTo-Json -Depth 4
"""


def usb_topology() -> list[dict]:
    """どのカメラがどの USB コントローラ配下かを見る。

    ノートPCは複数ポートが同一コントローラに束ねられていることが多く、
    そこが 4 台同時取得の帯域ボトルネックになる。
    """
    p = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_TOPOLOGY],
        capture_output=True, text=True, errors="replace",
    )
    try:
        data = json.loads(p.stdout)
    except Exception:
        return []
    return data if isinstance(data, list) else [data]


def probe_opencv(max_index: int = 10) -> dict[int, dict]:
    """OpenCV(DirectShow) で実際に開けるかを確認する。

    ffmpeg が列挙できても OpenCV から開けなければ意味がないので、
    実際に使う経路で裏取りする。
    """
    try:
        import cv2
    except ImportError:
        print("  (opencv-python 未導入のため --probe をスキップ)", file=sys.stderr)
        return {}

    found: dict[int, dict] = {}
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        # 順序が重要：FOURCC を先に指定しないと解像度が非圧縮で確定してしまう
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, frame = cap.read()
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        found[i] = {
            "opened": True,
            "read_ok": bool(ok),
            "actual_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "actual_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "actual_fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "fourcc": "".join(chr((fourcc >> (8 * k)) & 0xFF) for k in range(4)),
            "frame_shape": list(frame.shape) if ok and frame is not None else None,
        }
        cap.release()
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="カメラ能力と USB トポロジの列挙")
    ap.add_argument("--probe", action="store_true",
                    help="OpenCV で実際に開いて MJPEG FHD30 を要求してみる")
    ap.add_argument("--max-index", type=int, default=10)
    ap.add_argument("--include-virtual", action="store_true",
                    help="仮想カメラ（OBS/スマホ連携等）も対象に含める")
    args = ap.parse_args()

    # コンソールが cp932 でも日本語を落とさない
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    print("== DirectShow ビデオデバイス ==")
    names = list_dshow_devices(include_virtual=args.include_virtual)
    if not names:
        print("  カメラが見つかりません。接続と、他アプリが占有していないかを確認してください。")
        return 1

    cameras: list[Camera] = []
    for n in names:
        cam = Camera(name=n, modes=list_modes(n))
        cameras.append(cam)
        print("\n[" + n + "]")
        if not cam.modes:
            print("  モード列挙に失敗（デバイスが他アプリに占有されている可能性）")
            continue
        best = cam.best_mjpeg()
        print(f"  モード数: {len(cam.modes)}")
        if best:
            print(f"  MJPEG 最大: {best.width}x{best.height}@{best.fps_max:g}")
        else:
            print("  MJPEG モードなし  <-- 帯域的に致命的")
        print(f"  MJPEG FHD30: {'OK' if cam.supports_fhd30_mjpeg() else 'NG'}")

    if args.probe:
        print("\n== OpenCV(DSHOW) 実取得プローブ ==")
        probe = probe_opencv(args.max_index)
        for i, info in probe.items():
            print(f"  index {i}: {info['actual_width']}x{info['actual_height']}"
                  f"@{info['actual_fps']:g} {info['fourcc']} "
                  f"read={'OK' if info['read_ok'] else 'NG'}")
            if i < len(cameras):
                cameras[i].opencv_index = i
                cameras[i].opencv_probe = info
        if not probe:
            print("  開けたデバイスがありません")

    print("\n== USB コントローラ構成 ==")
    topo = usb_topology()
    shown = False
    for c in topo:
        devs = c.get("devices") or []
        cams = [d for d in devs if d and re.search(r"cam|webcam|video|uvc", d, re.I)]
        if cams:
            shown = True
            print(f"  {c.get('controller')}")
            for d in cams:
                print(f"    - {d}")
    if not topo:
        print("  取得できませんでした（PowerShell の実行権限を確認）")
    elif not shown:
        print("  コントローラ配下にカメラらしきデバイス名が見つかりませんでした")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "cameras.json"
    out.write_text(
        json.dumps({"cameras": [asdict(c) for c in cameras], "usb_controllers": topo},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n保存: " + str(out))

    # 要件定義書の制約セクションに直結する結論
    ng = [c.name for c in cameras if c.modes and not c.supports_fhd30_mjpeg()]
    if ng:
        print("\n!! MJPEG FHD30 非対応のカメラがあります（解像度/fps の後退が必要）:")
        for n in ng:
            print("   - " + n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
