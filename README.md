# GridVAD: Open-Set Video Anomaly Detection via Spatial Reasoning over Stratified Frame Grids

**Authors:** Mohamed Eltahir, Ahmed O. Ibrahim, Obada Siralkhatim, Tabarak Abdallah, Sondos Mohamed

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2603.25467-b31b1b)](https://arxiv.org/abs/2603.25467)

</div>

<div align="center">
  <img src="figures/GridVAD pipeline.png" width="1000">
  <p><em>The GridVAD pipeline. A VLM reasons over stratified grid representations of video clips to produce natural-language anomaly proposals. Self-Consistency Consolidation (SCC) filters hallucinations by retaining only proposals that recur across independent samplings. Grounding DINO anchors each surviving proposal to a bounding box, and SAM2 propagates it as a dense mask through the anomaly interval.</em></p>
</div>

---

## Highlights

- **Propose-Ground-Propagate Decomposition**: VLMs are repositioned from direct anomaly detectors to open-set *proposers*. Grounding DINO and SAM2 handle precise spatial and temporal localization — no task-specific training required.
- **Self-Consistency Consolidation (SCC)**: Treats the VLM as a stochastic sensor. Only proposals that recur across M independent stratified grid samplings are retained, eliminating one-off hallucinations without any learned parameters or score thresholds.
- **Stratified Grid Representation**: A video clip is tiled into a K×K spatial grid of temporally stratified frames, converting temporal anomaly detection into a single-pass image understanding task. The per-clip VLM budget is fixed at M+1 calls regardless of video length.
- **State-of-the-Art Zero-Shot Pixel Localization**: GridVAD achieves the highest Pixel-AUROC (77.59) on UCSD Ped2 among all compared methods, surpassing even the partially fine-tuned TAO (75.11).
- **Controllable Precision-Recall Tradeoff**: SCC's support threshold τ is an explicit knob — lower τ recovers object-level recall, higher τ tightens pixel-level precision.
- **2.7× Call Efficiency**: Structured spatial batching extracts more detection signal per VLM invocation than dense per-frame querying (Fr-AUC/call: 0.0094 vs. 0.0035).

---

## News

- **[March 2026]** GridVAD paper released on arXiv! Code released.

---

## Methodology

### Stage 1 — Stratified Grid Sampling

A clip of L frames is divided into K = g² equal temporal bins (default g=3, K=9). For each of M independent samplings, one frame is drawn at random from each bin and tiled into a single g×g grid image. This guarantees every temporal bin is represented while ensuring the M grids show different frames, providing complementary views at negligible cost.

### Stage 2 — Open-Set VLM Proposal

Each grid is passed to **Qwen3-VL-30B-A3B** with a structured prompt requesting free-form anomaly detection, description, and temporal localization. The VLM is given no predefined category list — it decides what constitutes an anomaly from first principles.

### Stage 3 — Self-Consistency Consolidation (SCC)

All M proposal sets are pooled and passed to a single text-only LLM call that:
1. Groups proposals referring to the same object instance across samplings
2. Merges their temporal intervals (union)
3. Counts cross-sampling support σⱼ per proposal

Only proposals with σⱼ ≥ τ survive. Genuine anomalies recur, while hallucinations do not.

### Stage 4 — Grounding and Propagation

Each surviving proposal's natural-language description is passed to **Grounding DINO** to localize a bounding box in the most confident anchor frame. **SAM2** then propagates a dense binary mask forward and backward through the SCC-refined temporal window. No component is trained on anomaly data.

---

## Results

### UCSD Ped2 — Pixel-Level and Object-Level Metrics

| Method | Training | Pixel-AUROC | Pixel-AP | Pixel-AUPRO | Pixel-F1 | RBDC | TBDC |
|--------|----------|:-----------:|:--------:|:-----------:|:--------:|:----:|:----:|
| AdaCLIP | Fine-tuned | 53.06 | 4.97 | 50.66 | 11.19 | 12.3 | 15.5 |
| AnomalyCLIP | Fine-tuned | 54.25 | 23.73 | 38.59 | 7.48 | 13.1 | 21.0 |
| DDAD | Trained | 55.87 | 5.61 | 15.12 | 2.67 | 18.01 | 13.29 |
| SimpleNet | Trained | 52.49 | 20.51 | 44.05 | 10.71 | 51.18 | 27.75 |
| DRAEM | Trained | 69.58 | 30.63 | 35.78 | 10.89 | 44.26 | 70.64 |
| TAO | Partial fine-tune | 75.11 | **50.78** | **72.97** | **64.12** | **83.6** | **93.2** |
| AdaCLIP | Zero-shot | 51.02 | 1.32 | 33.98 | 2.61 | 5.8 | 10.6 |
| AnomalyCLIP | Zero-shot | 51.63 | 21.20 | 36.34 | 5.92 | 7.5 | 11.2 |
| **GridVAD (Ours)** | **Zero-shot** | **77.59** | 38.53 | 66.82 | 42.09 | 38.96 | 37.70 |

GridVAD achieves the highest Pixel-AUROC across all methods including partially fine-tuned ones, and outperforms other zero-shot approaches on RBDC by over **5×**.


### SCC Ablation — Precision-Recall Tradeoff (ShanghaiTech, 15 videos)

| Config | Px-AUROC | Px-AP | Px-F1 | RBDC | TBDC |
|--------|:--------:|:-----:|:-----:|:----:|:----:|
| M=1, no SCC | 62.71 | 25.36 | 29.64 | **32.31** | **33.18** |
| M=5, τ=3 | **70.04** | **37.90** | **43.43** | 27.97 | 25.91 |

SCC trades object-level recall for substantially better pixel-level mask quality.

### Efficiency Comparison (UCSD Ped2)

| Paradigm | VLM Calls | Frame-AUROC | Time (s) | Fr-AUC / Call |
|----------|:---------:|:-----------:|:--------:|:-------------:|
| Uniform sampling | 201 | 0.6948 | 640.6 | 0.0035 |
| **GridVAD (Ours)** | **43** | 0.4024 | **276.7** | **0.0094** |

GridVAD uses **4.7× fewer VLM calls** and runs **2.3× faster**, while additionally producing dense pixel-level segmentation masks that uniform sampling cannot.

---

## Usage

### Running Evaluation

```bash
python eval/eval_runner.py \
    --dataset      shanghaitech \
    --dataset_root /path/to/shanghaitech/testing \
    --vlm_model    Qwen/Qwen3-VL-30B-A3B-Instruct \
    --output_dir   outputs/shanghaitech
```

```bash
python eval/eval_runner.py \
    --dataset      ucsd_ped2 \
    --dataset_root /path/to/ucsd_ped2/testing \
    --vlm_model    Qwen/Qwen3-VL-30B-A3B-Instruct \
    --output_dir   outputs/ucsd_ped2
```



## Output Structure

Each evaluated video produces a `results.json` under `outputs/<dataset>/<video_id>/`:

```
outputs/
  shanghaitech/
    01_0014/
      results.json      # anomaly proposals, grounding info, tracks, scores
    01_0015/
      results.json
    ...
```

### results.json Schema

```json
{
  "video_id": "01_0014",
  "anomalies": [
    {
      "label": "person running",
      "confidence": 0.85,
      "start_t": 4.2,
      "end_t": 7.8,
      "grounding": {
        "anchor_frame_idx": 130,
        "anchor_box": [120, 45, 310, 280]
      },
      "tracks": [
        {
          "frames": [
            {"frame_idx": 105, "bbox": [118, 42, 308, 275]},
            ...
          ]
        }
      ]
    }
  ],
  "frame_scores": [0.0, 0.0, 0.91, 0.91, ...]
}
```

---

## Citation

If you use GridVAD in your research, please cite:

```bibtex
@misc{eltahir2026gridvadopensetvideoanomaly,
      title={GridVAD: Open-Set Video Anomaly Detection via Spatial Reasoning over Stratified Frame Grids}, 
      author={Mohamed Eltahir and Ahmed O. Ibrahim and Obada Siralkhatim and Tabarak Abdallah and Sondos Mohamed},
      year={2026},
      eprint={2603.25467},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.25467}, 
}
```
