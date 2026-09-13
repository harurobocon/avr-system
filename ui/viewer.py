"""映像表示ウィジェット（ズーム/パン、状態オーバーレイつき）。

細部拡大は接触判定に必須なので、ズームは表示側で行う。
ホイールでカーソル位置を中心に拡大、ドラッグでパン。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from core.frame_source import SourceState

_STATE_COLOR = {
    SourceState.RUNNING: QColor(60, 200, 90),
    SourceState.FROZEN: QColor(240, 180, 40),
    SourceState.DISCONNECTED: QColor(230, 60, 60),
    SourceState.ERROR: QColor(230, 60, 60),
    SourceState.STOPPED: QColor(130, 130, 130),
}
_STATE_LABEL = {
    SourceState.RUNNING: "LIVE",
    SourceState.FROZEN: "FREEZE",
    SourceState.DISCONNECTED: "切断",
    SourceState.ERROR: "異常",
    SourceState.STOPPED: "停止",
}


class VideoView(QWidget):
    """1カメラ分の表示。クリックで単一表示への切替を要求する。"""

    clicked = Signal(str)          # source_id
    doubleClicked = Signal(str)

    MIN_ZOOM, MAX_ZOOM = 1.0, 16.0

    def __init__(self, source_id: str, title: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.source_id = source_id
        self.title = title or source_id
        self.setMinimumSize(160, 90)
        self.setAutoFillBackground(False)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self._pixmap: Optional[QPixmap] = None
        self._state = SourceState.STOPPED
        self._selected = False
        self._badge = ""            # 右上に出す補助情報（fps 等）

        self._zoom = 1.0
        self._center = QPointF(0.5, 0.5)   # 画像内の正規化座標
        self._drag_from: Optional[QPointF] = None
        self._drag_center: Optional[QPointF] = None

    # --- 外部から与えるもの ---
    def set_frame(self, image: Optional[np.ndarray]) -> None:
        if image is None:
            self._pixmap = None
        else:
            h, w = image.shape[:2]
            # BGR のまま QImage に渡して swap するのが最も安い（copy を避ける）
            qimg = QImage(image.data, w, h, image.strides[0], QImage.Format.Format_BGR888)
            self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_state(self, state: SourceState) -> None:
        if state is not self._state:
            self._state = state
            self.update()

    def set_selected(self, selected: bool) -> None:
        if selected != self._selected:
            self._selected = selected
            self.update()

    def set_badge(self, text: str) -> None:
        if text != self._badge:
            self._badge = text
            self.update()

    # --- ズーム/パン ---
    def reset_view(self) -> None:
        self._zoom = 1.0
        self._center = QPointF(0.5, 0.5)
        self.update()

    @property
    def zoom(self) -> float:
        return self._zoom

    def _clamp_center(self) -> None:
        half = 0.5 / self._zoom
        self._center.setX(min(max(self._center.x(), half), 1.0 - half))
        self._center.setY(min(max(self._center.y(), half), 1.0 - half))

    def wheelEvent(self, event) -> None:
        if self._pixmap is None:
            return
        old = self._zoom
        factor = 1.0015 ** event.angleDelta().y()
        self._zoom = min(max(self._zoom * factor, self.MIN_ZOOM), self.MAX_ZOOM)
        if self._zoom != old:
            # カーソル位置が動かないように中心を補正する
            pos = self._normalized_at(event.position())
            if pos is not None:
                k = old / self._zoom
                self._center = QPointF(
                    pos.x() + (self._center.x() - pos.x()) * k,
                    pos.y() + (self._center.y() - pos.y()) * k,
                )
            self._clamp_center()
            self.update()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_from = event.position()
            self._drag_center = QPointF(self._center)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.clicked.emit(self.source_id)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_from is None or self._pixmap is None:
            return
        rect = self._target_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        d = event.position() - self._drag_from
        self._center = QPointF(
            self._drag_center.x() - d.x() / rect.width() / self._zoom,
            self._drag_center.y() - d.y() / rect.height() / self._zoom,
        )
        self._clamp_center()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_from = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, event) -> None:
        self.doubleClicked.emit(self.source_id)

    # --- 描画 ---
    def _target_rect(self) -> QRectF:
        """アスペクト比を保って収めた表示領域。"""
        if self._pixmap is None:
            return QRectF(self.rect())
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw == 0 or ph == 0:
            return QRectF(self.rect())
        scale = min(self.width() / pw, self.height() / ph)
        w, h = pw * scale, ph * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _normalized_at(self, pos: QPointF) -> Optional[QPointF]:
        rect = self._target_rect()
        if not rect.contains(pos):
            return None
        half = 0.5 / self._zoom
        u = (pos.x() - rect.left()) / rect.width()
        v = (pos.y() - rect.top()) / rect.height()
        return QPointF(
            self._center.x() - half + u * 2 * half,
            self._center.y() - half + v * 2 * half,
        )

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 20))

        if self._pixmap is None:
            p.setPen(QColor(120, 120, 130))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "映像なし")
        else:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 4.0)
            target = self._target_rect()
            half = 0.5 / self._zoom
            src = QRectF(
                (self._center.x() - half) * self._pixmap.width(),
                (self._center.y() - half) * self._pixmap.height(),
                2 * half * self._pixmap.width(),
                2 * half * self._pixmap.height(),
            )
            p.drawPixmap(target, self._pixmap, src)

        self._draw_overlay(p)
        p.end()

    def _draw_overlay(self, p: QPainter) -> None:
        p.setPen(QColor(235, 235, 240))
        f = p.font()
        f.setPointSizeF(max(8.0, min(13.0, self.height() / 26)))
        p.setFont(f)
        p.drawText(10, int(f.pointSizeF()) + 10, self.title)

        # 状態バッジ。異常が一目で分かることが要件
        color = _STATE_COLOR.get(self._state, QColor(130, 130, 130))
        label = _STATE_LABEL.get(self._state, "?")
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(label) + 16
        th = fm.height() + 4
        x = self.width() - tw - 10
        p.fillRect(QRectF(x, 8, tw, th), color)
        p.setPen(QColor(20, 20, 20))
        p.drawText(QRectF(x, 8, tw, th), Qt.AlignmentFlag.AlignCenter, label)

        info = []
        if self._zoom > 1.001:
            info.append(f"x{self._zoom:.1f}")
        if self._badge:
            info.append(self._badge)
        if info:
            p.setPen(QColor(200, 200, 210))
            p.drawText(10, self.height() - 10, "  ".join(info))

        if self._selected:
            pen = QPen(QColor(80, 160, 255), 3)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(self.rect().adjusted(1, 1, -2, -2))
