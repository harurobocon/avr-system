# ロボコン VAR（映像判定・ハイライト配信）システム

ロボコン向けの VAR（Video Assistant Referee）システム。

1. 試合のハイライトシーンを配信映像に乗せる
2. 審判が試合中〜直後に映像で判定を確認できるようにする

映像入力は **RTMP を主**とし（別PCに既設の RTMP サーバーから購読する）、
USBカメラも併用できる。常時録画しておき、試合区間を後から切り出す方式。

> **開発中。** 正式な要件定義書は未作成。取得層と UI が動く段階。

## ドキュメント

| 文書 | 内容 |
|---|---|
| **[docs/HANDOVER.md](docs/HANDOVER.md)** | **まずこれを読む。** 経緯・設計判断・現状・残タスク |
| [poc/results/FINDINGS.md](poc/results/FINDINGS.md) | 実測結果と、そこから確定した設計判断 |
| [docs/requirements-draft.md](docs/requirements-draft.md) | 当初の要求ドラフト（原文保存） |
| [docs/ui/shortcuts.md](docs/ui/shortcuts.md) | 操作モデルとキーボードショートカット |

## セットアップ

**ffmpeg と ffprobe に PATH が通っていること**が前提（本システムの中核）。
Python は 3.12 以降のネイティブ Windows 版を使う
（MSYS2/mingw の python では PyPI の Windows wheel が入らない）。

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## 起動

```bash
# カメラ不要。合成映像4系統で UI を確認する（--fault で異常系も再現）
python -m ui.main_window --source synthetic:4 --fault

# RTMP サーバーから購読する（本命）
python -m ui.main_window --source "rtmp://host/live/cam0,rtmp://host/live/cam1"

# USB カメラ
python -m ui.main_window --source uvc:0,1,2,3
```

操作は [docs/ui/shortcuts.md](docs/ui/shortcuts.md) を参照。
`1`〜`4` 単一表示 / `0` 4分割 / `←→` コマ送り / `,` `.` 速度 / `Space` タグ / `L` ライブ。

## 環境の確認・計測

```bash
# 接続カメラの能力と USB コントローラ構成を列挙
python poc/enumerate_cameras.py --probe

# 同時取得の帯域/CPU/ドロップ/カメラ間ずれを実測
python poc/capture_bench.py --cameras 0,1,2,3 --ramp --duration 120
```

RTMP の疎通確認:

```python
from core.ffmpeg_source import probe_stream
print(probe_stream("rtmp://host/live/cam0"))
```

## 構成

```
core/   frame_source.py     … 全入力の共通インターフェース（UI はこれにしか依存しない）
        ffmpeg_source.py    … RTMP / RTSP / USB を同じ扱いにする本命の取得層。
                              1プロセスで無劣化セグメント録画 + 縮小プレビューを同時に出す
        uvc_source.py       … OpenCV 直叩きの USB 取得（露出固定を内蔵）
        synthetic_source.py … カメラ不要の合成映像。異常系の注入もできる
        timeline.py         … リングバッファと再生ヘッド。取得と再生を分離する要
ui/     main_window.py      … オペレータ画面
        viewer.py           … 映像表示（ズーム/パン、状態バッジ）
poc/    enumerate_cameras.py / capture_bench.py … 実測スクリプト
```
