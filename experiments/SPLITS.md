# VigiLens Scene Partitioning & Leakage Audit

**Audit Status**: PASSED (Zero Scene Leakage on Disjoint Partitions)

---

## 1. Local Workspace Smoke Split

The local test and development environment uses synthetic / smoke video pairs to verify pipeline integrity, causality, and thresholding without downloading external datasets.

| Split | Clips | Scenes | Scene IDs | Description |
| :--- | :--- | :--- | :--- | :--- |
| `train` | 4 | 1 | `train` | `Data/train` & `cache/train` (clips `train_000`..`train_003`) |
| `val` | 2 | 1 | `val` | `Data/val` & `cache/val` (clips `val_000`, `val_001`) |
| `test` | 4 | 2 | `normal`, `anomaly` | `Evaluation/manifest.json` & `cache/eval` (clips `normal_000`, `normal_001`, `anomaly_000`, `anomaly_001`) |

### Pairwise Overlap Audit (Workspace Split)
> [!NOTE]
> All scene sets are strictly disjoint:
> - `train` ∩ `val` = ∅
> - `train` ∩ `test` = ∅
> - `val` ∩ `test` = ∅
> No frame or scene leakage exists.

---

## 2. VidLLVIP Benchmark Split Audit & Problem P8 Verification

### The Baseline Leakage (Problem P8 — Confirmed)
In the prior VidLLVIP baseline configuration:
- Validation was designated as scenes `[12, 13, 14]`.
- The evaluation test set contained 114 clips across scenes `['09', '13']` (45 clips from scene 09, 69 clips from scene 13).
- **Leakage Finding**: Scene 13 (69 clips, representing 60.5% of the test set) was used during training validation / threshold calibration AND in the held-out test evaluation.

`scripts/audit_splits.py` was executed on the baseline run data:
```
[LEAKAGE AUDIT FAILED] Scene overlap detected across splits:
  Overlap between 'val' and 'test': scenes ['13']
    Scene 13 in val: ['13_0000_0005']
    Scene 13 in test: ['13_0000_0005', '13_0005_0010', '13_0010_0015', ...]
```

### Clean 4-Way Scene-Disjoint Partition
To eliminate all leakage across model selection, threshold calibration, and testing, VidLLVIP's 14 scenes are partitioned strictly by scene:

| Split Role | Scene IDs | Number of Scenes | Purpose |
| :--- | :--- | :--- | :--- |
| **Train** | `01`, `02`, `03`, `04`, `05`, `06`, `07`, `08`, `10`, `11` | 10 scenes (~71%) | Model weight optimization (fusion + temporal LSTM) |
| **Validation** | `12` | 1 scene (~7%) | Model selection via `val_ratio` and early stopping |
| **Calibration** | `14` | 1 scene (~7%) | Disjoint normal scene used strictly for threshold calibration |
| **Test** | `09`, `13` | 2 scenes (~14%) | Strictly held-out evaluation touched only once at evaluation time |

### Disjointness Proof
- `Train ∩ Val` = `{01..08, 10, 11} ∩ {12}` = ∅
- `Train ∩ Calib` = `{01..08, 10, 11} ∩ {14}` = ∅
- `Train ∩ Test` = `{01..08, 10, 11} ∩ {09, 13}` = ∅
- `Val ∩ Calib` = `{12} ∩ {14}` = ∅
- `Val ∩ Test` = `{12} ∩ {09, 13}` = ∅
- `Calib ∩ Test` = `{14} ∩ {09, 13}` = ∅

All intersections are provably empty. Zero data leakage at the scene level.

---

## 3. 5-Fold Grouped Leave-Scene-Out Cross Validation Schedule

To tune and evaluate models strictly without touching the frozen final holdout (`scenes 09, 13`), K=5 folds are defined across the remaining 12 scenes (`experiments/splits/folds.json`).

| Fold ID | Test Scenes (Held-out) | Calibration Scene | Validation Scene (Selection) | Training Scenes (Fit) |
| :--- | :--- | :--- | :--- | :--- |
| **Fold 0** | `01`, `02` | `03` | `04` | `05`, `06`, `07`, `08`, `10`, `11`, `12`, `14` |
| **Fold 1** | `03`, `04` | `05` | `06` | `01`, `02`, `07`, `08`, `10`, `11`, `12`, `14` |
| **Fold 2** | `05`, `06` | `07` | `08` | `01`, `02`, `03`, `04`, `10`, `11`, `12`, `14` |
| **Fold 3** | `07`, `08` | `10` | `11` | `01`, `02`, `03`, `04`, `05`, `06`, `12`, `14` |
| **Fold 4** | `10`, `11` | `12` | `14` | `01`, `02`, `03`, `04`, `05`, `06`, `07`, `08` |

- **Frozen Final Holdout**: Scenes `09` and `13` (114 clips total) remain 100% untouched across all 5 folds.
- **Audit Verification**: Every fold satisfies `Train ∩ Val ∩ Calib ∩ Test ∩ Holdout = ∅` as audited by `scripts/audit_splits.py`.
