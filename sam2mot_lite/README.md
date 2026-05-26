# SAM2MOT-lite

SAM2 (Segment Anything Model 2) を Multi-Object Tracking (MOT) に応用する、モジュール型の Tracking-by-Segmentation ベースライン実装です。  
[SAM2MOT](https://github.com/TripleJoy/SAM2MOT) の論文設計を参考にした再実装であり、公式コードのコピーではありません。

## 基本方針

- detector の bbox をそのまま最終結果にするのではなく、**SAM2 への box prompt** として使う
- SAM2 が出力した **mask から bbox を生成** し、MOT 形式で保存する
- track の主表現は bbox ではなく **mask**

## 入出力フォーマット

### 入力: detections.txt

事前生成済みの検出結果ファイル。以下の2形式に対応しています。

**標準 MOT 形式（推奨）:**
```
frame, id, x, y, w, h, score, class, visibility
```

**最小形式（id/class/visibility なし）:**
```
frame, x, y, w, h, score
```

> ⚠️ id 列は読み飛ばされます。track_id としては使用しません。

### 出力: trajectories.txt

TrackEval 互換の MOT 形式です。

```
frame, track_id, x, y, w, h, score, -1, -1, -1
```

- bbox は detector 由来ではなく、**SAM2 mask から算出した bbox** です
- frame_id は **1始まり**（MOT 標準）

### Frame ID の変換

| 形式 | インデックス | 用途 |
|------|-------------|------|
| MOT frame_id | 1始まり | detections.txt, trajectories.txt |
| SAM2 frame_idx | 0始まり | SAM2 API 内部 |

変換関数 `mot_to_sam2_frame()` / `sam2_to_mot_frame()` を `tracker/sam2_wrapper.py` に用意しています。

## ディレクトリ構成

```
sam2mot_lite/
├── configs/
│   └── default.yaml              # 全パラメータ設定
├── tracker/
│   ├── __init__.py
│   ├── detection.py              # Detection dataclass・読み込み・フィルタ
│   ├── result_writer.py          # MOT 形式の trajectory 書き出し
│   ├── mask_utils.py             # mask/bbox ユーティリティ
│   ├── matching.py               # Hungarian matching (IoU ベース)
│   ├── sam2_wrapper.py           # SAM2VideoPredictor ラッパー
│   ├── track.py                  # Track dataclass（M4〜）
│   ├── trajectory_manager.py     # トラック管理（M4〜）
│   ├── cross_object_interaction.py  # COI 近似（M8〜）
│   └── result_writer.py
├── scripts/
│   ├── run_sequence.py           # シーケンス単位の推論（M4〜）
│   ├── run_dancetrack.py         # DanceTrack 評価（M9〜）
│   └── visualize.py              # 可視化（M4〜）
├── tests/
│   ├── test_detection_reader.py  # Unit test: 検出読み込み
│   ├── test_result_writer.py     # Unit test: 結果書き出し
│   ├── test_mask_utils.py        # Unit test: mask/bbox 操作
│   ├── test_matching.py          # Unit test: マッチング
│   ├── smoke_sam2_single_object.py  # Smoke test: 単一オブジェクト
│   └── smoke_sam2_multi_object.py   # Smoke test: 複数オブジェクト
├── README.md
├── DESIGN.md                     # M4〜M9 の設計ドキュメント
└── SAM2_COMMIT.txt               # ベースとした SAM2 の commit hash
```

## 主要モジュール

### tracker/detection.py

| 関数・クラス | 説明 |
|-------------|------|
| `Detection` | dataclass: `frame_id`, `box_xyxy`, `score`, `cls`(任意), `raw`(任意) |
| `Detection.from_xywh(...)` | xywh → xyxy 変換して生成 |
| `Detection.to_xywh()` | xyxy → xywh に戻す |
| `read_detections(file_path, score_thr)` | detections.txt を読み込み、Detection リストを返す |
| `filter_detections_by_score(detections, score_thr)` | スコア閾値でフィルタリング |

### tracker/mask_utils.py

| 関数 | 説明 |
|------|------|
| `mask_to_box(mask)` | binary mask → xyxy bbox。空 mask なら `None` |
| `mask_iou(mask_a, mask_b)` | 2つの binary mask 間の IoU |
| `union_masks(masks)` | mask リストの和集合 |
| `bbox_iou(box_a, box_b)` | 2つの xyxy bbox 間の IoU |
| `compute_free_area_ratio(box, free_mask)` | bbox 内の空き領域比率 |

### tracker/matching.py

| 関数 | 説明 |
|------|------|
| `iou_matching(tracks, detections, iou_thr)` | IoU コストによる Hungarian matching |

戻り値: `(matches, unmatched_tracks, unmatched_dets)`
- `matches`: `List[Tuple[track_idx, det_idx]]`
- `unmatched_tracks`: `List[int]`
- `unmatched_dets`: `List[int]`

> `scipy.optimize.linear_sum_assignment` を使用。scipy 未インストール時は明示的なエラーメッセージを表示します。

### tracker/sam2_wrapper.py

| 関数・クラス | 説明 |
|-------------|------|
| `mot_to_sam2_frame(mot_frame_id)` | MOT 1始まり → SAM2 0始まり |
| `sam2_to_mot_frame(sam2_frame_idx)` | SAM2 0始まり → MOT 1始まり |
| `SAM2Wrapper(config_path, checkpoint_path, device)` | SAM2VideoPredictor のラッパー |
| `.init_video(frames_dir)` | 動画シーケンスの初期化 |
| `.add_box_prompt(frame_idx, obj_id, box_xyxy)` | box prompt 追加 → `(mask, bbox_xyxy, score)` |
| `.propagate_in_video(start_frame_idx)` | フレーム間のマスク伝播（generator） |
| `.extract_result(out_obj_ids, out_mask_logits, obj_id)` | 特定 obj_id の結果取得 |

> score は暫定値（正の mask logits の平均値）です。SAM2 のネイティブスコアが利用可能になり次第改善予定。

## テストの実行

### Unit Test（CPU のみ、GPU 不要）

検出読み込み・mask 操作・マッチング・結果書き出しの動作確認です。

```bash
cd /path/to/repo_root
PYTHONPATH=./sam2mot_lite .venv-sam2mot/bin/python -m unittest \
  sam2mot_lite.tests.test_detection_reader \
  sam2mot_lite.tests.test_result_writer \
  sam2mot_lite.tests.test_mask_utils \
  sam2mot_lite.tests.test_matching -v
```

### Smoke Test（GPU + SAM2 checkpoint 必須）

SAM2 モデルをロードし、ダミー動画でマスク生成・追跡を検証します。

**共通引数:**

| 引数 | デフォルト値 | 説明 |
|------|-------------|------|
| `--config` | `configs/sam2.1/sam2.1_hiera_t.yaml` | SAM2 config 名（Hydra config search path 内） |
| `--checkpoint` | `sam2/checkpoints/sam2.1_hiera_tiny.pt` | SAM2 checkpoint ファイルのパス |

**Single Object（1オブジェクト追跡テスト）:**
```bash
PYTHONPATH=./sam2mot_lite .venv-sam2mot/bin/python \
  sam2mot_lite/tests/smoke_sam2_single_object.py
```
- ダミー動画（5フレーム）を生成し、1つの box prompt から mask を生成・保存

**Multi Object（複数オブジェクト追跡テスト）:**
```bash
PYTHONPATH=./sam2mot_lite .venv-sam2mot/bin/python \
  sam2mot_lite/tests/smoke_sam2_multi_object.py
```
- 2つのオブジェクトに box prompt → 5フレーム伝播 → trajectories.txt に保存

## 設定ファイル

`configs/default.yaml` に全パラメータが定義されています。

```yaml
# 検出フィルタ・マッチング
det_conf_thr: 0.5           # 検出スコア閾値
iou_match_thr: 0.5          # IoU マッチング閾値
free_ratio_thr: 0.7         # 新規オブジェクト追加の空き領域閾値

# トラック状態管理
reliable_thr: 2.0           # reliable 状態の閾値
pending_thr: 0.0            # pending 状態の閾値
lost_thr: -2.0              # lost 状態の閾値
lost_tolerance: 25          # lost 許容フレーム数

# 機能の ON/OFF
enable_object_addition: true
enable_object_removal: true
enable_quality_reconstruction: true
enable_cross_object_interaction: false

# Cross-object Interaction
coi_miou_thr: 0.8
coi_score_gap_thr: 2.0
coi_var_window: 10

# 出力制御
save_masks: true
save_visualization: true
max_frames: null
```

## Milestone 一覧

| # | 内容 | 状態 |
|---|------|------|
| M0 | Detection Reader / MOT Writer | 実装済み |
| M1 | Mask Utility / Matching | 実装済み |
| M2 | SAM2 Single Object Prompt | 実装済み |
| M3 | SAM2 Multi Object Prompt | 実装済み |
| M4 | 最小 SAM2MOT-lite 推論 | 設計のみ（DESIGN.md） |
| M5 | Object Addition | 設計のみ |
| M6 | Object Removal | 設計のみ |
| M7 | Quality Reconstruction | 設計のみ |
| M8 | Cross-object Interaction 近似 | 設計のみ |
| M9 | TrackEval 評価接続 | 設計のみ |

M4 以降の設計詳細は [DESIGN.md](./DESIGN.md) を参照してください。
