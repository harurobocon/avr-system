"""VAR オペレータ画面（1台1画面で兼用）。

  python -m ui.main_window --source synthetic:4
  python -m ui.main_window --source synthetic:4 --width 1280 --height 720
  python -m ui.main_window --source uvc:0,1,2,3

キーボード操作を主にしている。VAR は秒単位で操作する道具で、
マウスだけの設計では実運用の速度に追いつかないため。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtGui import QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMainWindow, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from core.frame_source import FrameSource, SourceState  # noqa: E402
from core.synthetic_source import SyntheticSource  # noqa: E402
from core.timeline import Timeline  # noqa: E402
from ui.viewer import VideoView  # noqa: E402

# サッカーVARの運用に倣い、接触点確認用のスローと、反則の強さ/意図性を見る
# 実速度を、ワンタッチで行き来できるようにする
SPEED_PRESETS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
SLIDER_RESOLUTION = 10000


def _fmt(ns: int | None, origin_ns: int | None) -> str:
    if ns is None or origin_ns is None:
        return "--:--.---"
    s = max(0.0, (ns - origin_ns) / 1e9)
    return f"{int(s // 60):02d}:{s % 60:06.3f}"


class MainWindow(QMainWindow):
    def __init__(self, timeline: Timeline) -> None:
        super().__init__()
        self.timeline = timeline
        self.setWindowTitle("ロボコン VAR オペレータ画面")
        self.resize(1400, 900)

        self.views: dict[str, VideoView] = {}
        self.single: str | None = None      # None = 4分割
        self.selected: str = next(iter(timeline.buffers))
        self._scrubbing = False
        self._last_tick = time.perf_counter()
        self._fps_t0 = time.perf_counter()
        self._fps_counts = {sid: 0 for sid in timeline.buffers}

        self._build()
        self._bind_keys()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)                # 約60Hz で描画（取得とは独立）

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        outer.addLayout(self._build_header())

        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)
        for i, sid in enumerate(self.timeline.buffers):
            src = next(s for s in self.timeline.sources if s.source_id == sid)
            v = VideoView(sid, title=f"[{i + 1}] {src.capabilities.name or sid}")
            v.clicked.connect(self._on_view_clicked)
            v.doubleClicked.connect(self._toggle_single)
            self.views[sid] = v
        self._relayout()

        body = QHBoxLayout()
        body.addWidget(self.grid_host, 1)
        body.addLayout(self._build_tag_panel())
        outer.addLayout(body, 1)

        outer.addLayout(self._build_transport())
        self.setCentralWidget(root)
        self._apply_style()

    def _build_header(self) -> QHBoxLayout:
        h = QHBoxLayout()
        self.lbl_match = QLabel("試合時間 --:--")
        self.lbl_match.setObjectName("match")
        h.addWidget(self.lbl_match)
        h.addSpacing(16)

        self.lbl_rec = QLabel("● REC")
        self.lbl_rec.setObjectName("rec")
        h.addWidget(self.lbl_rec)
        h.addSpacing(16)

        self.lbl_mode = QLabel("LIVE")
        self.lbl_mode.setObjectName("mode")
        h.addWidget(self.lbl_mode)
        h.addStretch(1)

        self.lbl_health = QLabel("")
        h.addWidget(self.lbl_health)
        h.addSpacing(12)

        h.addWidget(QLabel("配信出力:"))
        self.cmb_program = QComboBox()
        self.cmb_program.addItem("選択中のカメラを送出")
        for sid in self.timeline.buffers:
            self.cmb_program.addItem(f"{sid} を固定送出")
        self.cmb_program.addItem("4分割を送出")
        h.addWidget(self.cmb_program)
        return h

    def _build_tag_panel(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.addWidget(QLabel("ハイライト"))
        self.lst_tags = QListWidget()
        self.lst_tags.setFixedWidth(240)
        self.lst_tags.itemActivated.connect(self._jump_to_tag)
        self.lst_tags.itemClicked.connect(self._jump_to_tag)
        col.addWidget(self.lst_tags, 1)
        btn = QPushButton("タグを打つ  (Space)")
        btn.clicked.connect(self._add_tag)
        col.addWidget(btn)
        return col

    def _build_transport(self) -> QVBoxLayout:
        box = QVBoxLayout()

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, SLIDER_RESOLUTION)
        self.slider.sliderPressed.connect(lambda: setattr(self, "_scrubbing", True))
        self.slider.sliderReleased.connect(self._on_scrub_end)
        self.slider.valueChanged.connect(self._on_scrub)
        box.addWidget(self.slider)

        row = QHBoxLayout()
        self.lbl_time = QLabel("--:--.---")
        self.lbl_time.setObjectName("time")
        row.addWidget(self.lbl_time)

        row.addWidget(QLabel("移動:"))
        self.edit_time = QLineEdit()
        self.edit_time.setPlaceholderText("mm:ss.mmm")
        self.edit_time.setFixedWidth(110)
        self.edit_time.returnPressed.connect(self._seek_typed)
        row.addWidget(self.edit_time)

        row.addSpacing(12)
        for label, delta in (("|◀ 1f", -1), ("1f ▶|", +1)):
            b = QPushButton(label)
            b.setFixedWidth(66)
            b.clicked.connect(lambda _=False, d=delta: self._step(d))
            row.addWidget(b)

        self.btn_play = QPushButton("一時停止")
        self.btn_play.setFixedWidth(90)
        self.btn_play.clicked.connect(self._toggle_pause)
        row.addWidget(self.btn_play)

        row.addSpacing(12)
        row.addWidget(QLabel("再生速度:"))
        self.speed_buttons: dict[float, QPushButton] = {}
        for sp in SPEED_PRESETS:
            b = QPushButton(f"{sp:g}x")
            b.setFixedWidth(52)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, s=sp: self._set_rate(s))
            row.addWidget(b)
            self.speed_buttons[sp] = b

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setFixedWidth(150)
        self.speed_slider.setRange(5, 400)      # 0.05x 〜 4.00x を連続で
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(lambda v: self._set_rate(v / 100.0, False))
        row.addWidget(self.speed_slider)

        row.addStretch(1)
        self.btn_live = QPushButton("LIVE へ戻る  (L)")
        self.btn_live.clicked.connect(self._go_live)
        row.addWidget(self.btn_live)
        box.addLayout(row)

        hint = QLabel("1-4 単一表示 / 0 4分割 / ←→ コマ送り / , . 速度 / K 一時停止 "
                      "/ Space タグ / L ライブ / R ズーム解除")
        hint.setObjectName("hint")
        box.addWidget(hint)
        return box

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #141417; color: #e6e6ea; }
            QLabel#match { font-size: 22px; font-weight: 600; }
            QLabel#time  { font-family: Consolas, monospace; font-size: 18px; }
            QLabel#rec   { color: #ff4d4d; font-weight: 600; }
            QLabel#mode  { font-weight: 600; padding: 2px 10px; border-radius: 3px;
                           background: #2a6df4; }
            QLabel#hint  { color: #8a8a95; font-size: 11px; }
            QPushButton  { background: #26262c; border: 1px solid #35353d;
                           padding: 5px 10px; border-radius: 3px; }
            QPushButton:hover   { background: #32323a; }
            QPushButton:checked { background: #2a6df4; border-color: #2a6df4; }
            QListWidget, QLineEdit, QComboBox {
                background: #1c1c21; border: 1px solid #35353d; border-radius: 3px;
                padding: 3px; }
        """)

    def _bind_keys(self) -> None:
        def sc(seq: str, fn) -> None:
            QShortcut(QKeySequence(seq), self, activated=fn)

        for i, sid in enumerate(list(self.timeline.buffers)[:9]):
            sc(str(i + 1), lambda s=sid: self._show_single(s))
        sc("0", lambda: self._show_single(None))
        sc("Left", lambda: self._step(-1))
        sc("Right", lambda: self._step(+1))
        sc("Shift+Left", lambda: self._step(-10))
        sc("Shift+Right", lambda: self._step(+10))
        sc(",", lambda: self._nudge_rate(-1))
        sc(".", lambda: self._nudge_rate(+1))
        sc("K", self._toggle_pause)
        sc("Space", self._add_tag)
        sc("L", self._go_live)
        sc("R", lambda: self.views[self.selected].reset_view())

    # -------------------------------------------------------------- 配置
    def _relayout(self) -> None:
        while self.grid.count():
            self.grid.takeAt(0).widget().setParent(None)
        if self.single is not None:
            self.grid.addWidget(self.views[self.single], 0, 0)
            self.views[self.single].show()
            for sid, v in self.views.items():
                if sid != self.single:
                    v.hide()
        else:
            for i, (sid, v) in enumerate(self.views.items()):
                v.show()
                self.grid.addWidget(v, i // 2, i % 2)
        for sid, v in self.views.items():
            v.set_selected(sid == self.selected)

    def _show_single(self, sid: str | None) -> None:
        if sid is not None and sid not in self.views:
            return
        self.single = sid
        if sid is not None:
            self.selected = sid
        self._relayout()

    def _toggle_single(self, sid: str) -> None:
        self._show_single(None if self.single else sid)

    def _on_view_clicked(self, sid: str) -> None:
        self.selected = sid
        for s, v in self.views.items():
            v.set_selected(s == sid)

    # ------------------------------------------------------------ 再生操作
    def _origin(self) -> int | None:
        sp = self.timeline.span()
        return sp[0] if sp else None

    def _step(self, delta: int) -> None:
        self.timeline.step_frames(delta, self.selected)
        self._sync_play_button()

    def _toggle_pause(self) -> None:
        tl = self.timeline
        if tl.is_live:
            tl.seek(tl.current_ns() or 0)   # ライブ中の一時停止は「その場で止まる」
            tl.paused = True
        else:
            tl.paused = not tl.paused
        self._sync_play_button()

    def _set_rate(self, rate: float, sync_slider: bool = True) -> None:
        self.timeline.rate = rate
        if self.timeline.is_live:
            self.timeline.seek(self.timeline.current_ns() or 0)
        self.timeline.paused = False
        if sync_slider:
            self.speed_slider.blockSignals(True)
            self.speed_slider.setValue(int(rate * 100))
            self.speed_slider.blockSignals(False)
        for sp, b in self.speed_buttons.items():
            b.setChecked(abs(sp - rate) < 1e-6)
        self._sync_play_button()

    def _nudge_rate(self, direction: int) -> None:
        cur = self.timeline.rate
        if direction < 0:
            cands = [s for s in SPEED_PRESETS if s < cur - 1e-9]
            self._set_rate(cands[-1] if cands else SPEED_PRESETS[0])
        else:
            cands = [s for s in SPEED_PRESETS if s > cur + 1e-9]
            self._set_rate(cands[0] if cands else SPEED_PRESETS[-1])

    def _go_live(self) -> None:
        self.timeline.go_live()
        self._set_rate(1.0)

    def _sync_play_button(self) -> None:
        self.btn_play.setText("再生" if self.timeline.paused else "一時停止")

    def _on_scrub(self, value: int) -> None:
        if not self._scrubbing:
            return
        sp = self.timeline.span()
        if not sp:
            return
        self.timeline.seek(sp[0] + int((sp[1] - sp[0]) * value / SLIDER_RESOLUTION))

    def _on_scrub_end(self) -> None:
        self._scrubbing = False
        self.timeline.paused = True
        self._sync_play_button()

    def _seek_typed(self) -> None:
        text = self.edit_time.text().strip()
        origin = self._origin()
        if not text or origin is None:
            return
        try:
            if ":" in text:
                m, s = text.split(":", 1)
                seconds = int(m) * 60 + float(s)
            else:
                seconds = float(text)
        except ValueError:
            self.edit_time.selectAll()
            return
        self.timeline.seek(origin + int(seconds * 1e9))
        self.timeline.paused = True
        self._sync_play_button()
        self.edit_time.clear()

    # ------------------------------------------------------------ タグ
    def _add_tag(self) -> None:
        tag = self.timeline.add_tag()
        self._refresh_tags()
        for i in range(self.lst_tags.count()):
            if self.lst_tags.item(i).data(Qt.ItemDataRole.UserRole) == tag["ts_ns"]:
                self.lst_tags.setCurrentRow(i)
                break

    def _refresh_tags(self) -> None:
        origin = self._origin()
        self.lst_tags.clear()
        for t in self.timeline.tags:
            item_text = f"{_fmt(t['ts_ns'], origin)}  {t['label']}"
            self.lst_tags.addItem(item_text)
            self.lst_tags.item(self.lst_tags.count() - 1).setData(
                Qt.ItemDataRole.UserRole, t["ts_ns"])

    def _jump_to_tag(self, item) -> None:
        ts = item.data(Qt.ItemDataRole.UserRole)
        if ts is not None:
            self.timeline.seek(int(ts))
            self.timeline.paused = True
            self._sync_play_button()

    # ------------------------------------------------------------ 毎フレーム
    def _tick(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_tick
        self._last_tick = now

        self.timeline.advance(elapsed)

        frames = self.timeline.frames_at_playhead()
        for sid, frame in frames.items():
            view = self.views[sid]
            if self.single is not None and sid != self.single:
                continue        # 非表示のビューは描画しない（無駄な負荷を避ける）
            view.set_frame(frame.image if frame else None)
            src = next(s for s in self.timeline.sources if s.source_id == sid)
            view.set_state(src.state)
            if frame:
                self._fps_counts[sid] += 1

        # 実効 fps を 1 秒ごとに更新して各ビューに出す
        if now - self._fps_t0 >= 1.0:
            for sid, view in self.views.items():
                src = next(s for s in self.timeline.sources if s.source_id == sid)
                st = src.stats
                view.set_badge(f"{st.effective_fps:.1f}fps  drop {st.drop_rate * 100:.1f}%")
                self._fps_counts[sid] = 0
            self._fps_t0 = now
            self._update_health()

        self._update_transport()

    def _update_health(self) -> None:
        bad = [s.source_id for s in self.timeline.sources
               if s.state in (SourceState.DISCONNECTED, SourceState.ERROR, SourceState.FROZEN)]
        if bad:
            self.lbl_health.setText("⚠ 異常: " + ", ".join(bad))
            self.lbl_health.setStyleSheet("color:#ff6b6b; font-weight:600;")
        else:
            self.lbl_health.setText("全カメラ正常")
            self.lbl_health.setStyleSheet("color:#63c96f;")

    def _update_transport(self) -> None:
        tl = self.timeline
        origin = self._origin()
        cur = tl.current_ns()
        self.lbl_time.setText(_fmt(cur, origin))

        if tl.is_live:
            self.lbl_mode.setText("LIVE")
            self.lbl_mode.setStyleSheet("background:#2a6df4;")
        else:
            self.lbl_mode.setText(f"リプレイ {tl.rate:g}x" + ("（停止）" if tl.paused else ""))
            self.lbl_mode.setStyleSheet("background:#c9822a;")

        sp = tl.span()
        if sp and sp[1] > sp[0] and cur is not None and not self._scrubbing:
            pos = int((cur - sp[0]) / (sp[1] - sp[0]) * SLIDER_RESOLUTION)
            self.slider.blockSignals(True)
            self.slider.setValue(max(0, min(SLIDER_RESOLUTION, pos)))
            self.slider.blockSignals(False)

        # 試合時間は RoLIMOA 連携で置き換える。未連携時はバッファ先頭からの経過。
        self.lbl_match.setText("試合時間 " + _fmt(cur, origin) + "  （RoLIMOA 未接続）")

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.timeline.stop()
        super().closeEvent(event)


def _apply_japanese_font(app: QApplication) -> None:
    """日本語グリフを持つフォントを明示的に選ぶ。

    既定フォント任せだと環境によって日本語が豆腐（□）になる。
    UI の文言が読めないのは致命的なので、候補から実在するものを選ぶ。
    """
    from PySide6.QtGui import QFont, QFontDatabase

    families = set(QFontDatabase.families())
    for name in ("Yu Gothic UI", "Meiryo UI", "Meiryo", "MS UI Gothic",
                 "Noto Sans CJK JP", "Noto Sans JP"):
        if name in families:
            f = QFont(name)
            f.setPointSize(10)
            app.setFont(f)
            return


def build_sources(spec: str, width: int, height: int, fps: float) -> list[FrameSource]:
    # "rtmp://host/live/a,rtmp://host/live/b" のように URL をそのまま渡せる形を優先。
    # "kind:arg" で partition すると URL の "://" を巻き込んで壊れるため。
    for scheme in ("rtmp", "rtmps", "rtsp"):
        if spec.startswith(f"{scheme}://"):
            kind, arg = scheme, spec
            break
    else:
        kind, _, arg = spec.partition(":")
    if kind == "synthetic":
        n = int(arg or 4)
        return [
            SyntheticSource(f"cam{i}", index=i, width=width, height=height, fps=fps)
            for i in range(n)
        ]
    if kind == "uvc":
        from core.uvc_source import UvcSource
        idx = [int(x) for x in arg.split(",") if x.strip()]
        return [
            UvcSource(f"cam{i}", index=i, width=width, height=height, fps=fps)
            for i in idx
        ]
    if kind in ("rtmp", "rtmps", "rtsp"):
        # 例: --source "rtmp://192.168.1.10/live/cam0,rtmp://192.168.1.10/live/cam1"
        from core.ffmpeg_source import FfmpegSource, PreviewSize
        urls = [u.strip() for u in arg.split(",") if u.strip()]
        preview = PreviewSize(width, height)
        factory = FfmpegSource.from_rtsp if kind == "rtsp" else FfmpegSource.from_rtmp
        return [factory(f"cam{i}", url, preview=preview, fps=fps)
                for i, url in enumerate(urls)]
    raise SystemExit(
        f"未知のソース指定: {spec}\n"
        "  synthetic:4 / uvc:0,1,2,3 / rtmp://host/live/a,rtmp://host/live/b")


def main() -> int:
    ap = argparse.ArgumentParser(description="VAR オペレータ画面")
    ap.add_argument("--source", default="synthetic:4",
                    help="synthetic:4 / uvc:0,1,2,3")
    ap.add_argument("--width", type=int, default=960,
                    help="合成ソースの生成解像度（UI検証は小さめで十分）")
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--buffer-seconds", type=float, default=120.0,
                    help="メモリ上に保持する秒数（試合長 2 分が既定）")
    ap.add_argument("--fault", action="store_true",
                    help="合成ソースに切断/フリーズを注入して異常系UIを確認する")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    _apply_japanese_font(app)
    sources = build_sources(args.source, args.width, args.height, args.fps)
    timeline = Timeline(sources, max_seconds=args.buffer_seconds)
    timeline.start()

    win = MainWindow(timeline)
    win.show()

    if args.fault and len(sources) >= 2:
        # 10秒後に cam1 をフリーズ、15秒後に cam2 を切断
        QTimer.singleShot(10_000, lambda: sources[1].freeze(5.0))
        if len(sources) >= 3:
            QTimer.singleShot(15_000, lambda: sources[2].disconnect())
            QTimer.singleShot(25_000, lambda: sources[2].reconnect())

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
