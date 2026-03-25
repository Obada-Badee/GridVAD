import argparse
import os
import sys
import numpy as np
import json
from tqdm import tqdm
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from grid_vad.data.benchmarks import BenchmarkLoader
from grid_vad.core.grounding import GroundingModule
from grid_vad.core.propagate import PropagationModule
from grid_vad.pipeline.runner import GridVADPipeline
from grid_vad.pipeline.exporter import Exporter
from grid_vad.eval.tao_object import compute_dataset_tbdc_rbdc, read_tracks, read_detected_anomalies
from grid_vad.eval.pixel import calculate_pixel_metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from grid_vad.data.video_io import VideoReader


def main():
    parser = argparse.ArgumentParser(description="GridVAD Evaluation Runner")
    parser.add_argument("--dataset", required=True, choices=['shanghaitech', 'ucsd_ped2'])
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--video", help="Run specific video ID")
    parser.add_argument("--gt_tracks_dir", help="Path to GT tracks directory (for TBDC/RBDC)")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--vlm_model", default="")
    parser.add_argument("--grid_size", type=int, default=3, help="Grid size (NxN). Also determines sampling K = grid_size^2.")
    parser.add_argument("--M", type=int, default=5, help="Number of VLM passes per clip")
    parser.add_argument("--pad_sec", type=float, default=0.0, help="Padding in seconds for propagation window")
    parser.add_argument("--min_support", type=int, default=1, help="Minimum support for SCC")
    parser.add_argument("--box_threshold", type=float, default=0.05, help="Grounding Box Threshold (set 0 for top-1)")
    parser.add_argument("--text_threshold", type=float, default=0.05, help="Grounding Text Threshold")
    parser.add_argument("--sam_area_thresh", type=int, default=50, help="SAM2 Tracking Area Threshold")
    parser.add_argument("--R", type=int, default=5, help="Number of anchor frames to try")
    parser.add_argument("--clip_len", type=float, default=60.0, help="Clip length in seconds")
    parser.add_argument("--overlap", type=float, default=2.0, help="Overlap in seconds")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Verbose output. When False (default), shows only a tqdm bar.")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip videos whose results.json already exists (default: True).")
    
    args = parser.parse_args()

    # Suppress third-party loggers when not in debug mode
    if not args.debug:
        import logging
        logging.disable(logging.CRITICAL)   # silences transformers, SAM2, etc.

    # 1. Setup
    loader = BenchmarkLoader(args.dataset, args.dataset_root)
    exporter = Exporter(args.output_dir)
    
    # Cache dir
    cache_dir = os.environ.get("HF_HOME", "/ibex/user/hamidme/cache_dir/")
    device = "cuda" if torch.cuda.is_available() else "cpu"


    from grid_vad.core.vlm import LocalVLMClient
    vlm_client = LocalVLMClient(model_id=args.vlm_model, cache_dir=cache_dir, device=device)
    
    # Init Grounding and Propagation (SAM3)
    grounding = GroundingModule(device=device, cache_dir=cache_dir)
    propagator = PropagationModule(device=device, cache_dir=cache_dir)

    config = {
        "grid_size": args.grid_size,
        "clip_len": args.clip_len,
        "overlap": args.overlap,
        "M": args.M,
        "pad_sec": args.pad_sec,
        "min_support": args.min_support,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "sam_area_thresh": args.sam_area_thresh,
        "R": args.R,
        "prompts": ["anomaly"]
    }
    
    pipeline = GridVADPipeline(config, vlm_client, grounding, propagator)

    # 2. Get videos
    if args.video:
        videos = [args.video]
    else:
        videos = loader.get_video_paths()

    log = print if args.debug else lambda *a, **k: None
    log(f"Running evaluation on {len(videos)} videos from {args.dataset}...")

    # Metrics accumulators
    all_frame_scores  = []
    all_frame_labels  = []

    # Pixel-level (collected per annotated frame, then averaged)
    all_p_aurocs = []
    all_p_aps    = []
    all_p_aupros = []
    all_p_f1s    = []

    # Object-level TAO — accumulated globally, computed once after loop
    global_gt_tracks  = []   # ContinuousTrack objects from all videos
    global_detections = []   # Region objects from all videos
    global_total_frames = 0  # sum of frame counts across all videos

    # 3. Loop
    pbar = tqdm(videos, desc="Evaluating", unit="vid", disable=args.debug)
    for vid in pbar:
        if not args.debug:
            pbar.set_postfix_str(vid)
        log(f"\nProcessing {vid}...")
        
        # Paths
        vid_output_dir = os.path.join(args.output_dir, vid)
        os.makedirs(vid_output_dir, exist_ok=True)

        results_json_path = os.path.join(vid_output_dir, "results.json")
        vpath = loader.get_video_file_path(vid)

        # ── Auto-resume: load cached results if available ─────────────────
        if args.resume and os.path.exists(results_json_path):
            log(f"  [RESUME] Loading cached results from {results_json_path}")
            with open(results_json_path) as f:
                results = json.load(f)
            tmp_reader = VideoReader(vpath)
            num_frames = tmp_reader.frame_count
            fps        = tmp_reader.fps
            tmp_reader.close()
        else:
            # A. Run Pipeline
            results = pipeline.run_on_video(vpath)
            tmp_reader = VideoReader(vpath)
            num_frames = tmp_reader.frame_count
            fps        = tmp_reader.fps
            tmp_reader.close()
            # B. Export
            exporter.save_json(results, filename=f"{vid}/results.json")
            exporter.save_tao_txt(results['anomalies'], vid, fps=fps)

        # C. Evaluation
        # C1. Frame Level AUROC
        gt_frames = loader.load_gt_frames(vid)
        if len(gt_frames) > 0:
            scores = np.zeros(num_frames)
            for inst in results['anomalies']:
                conf = inst.get('confidence', 0.5)
                # Guard: replace any NaN/None/inf confidence with 0.0
                if conf is None or (isinstance(conf, float) and (conf != conf or conf == float('inf'))):
                    conf = 0.0
                conf = float(np.clip(conf, 0.0, 1.0))

                # --- Track-coverage scoring (preferred) ---
                # Use the exact frames where SAM2 tracked the object.
                # This is far more temporally precise than the coarse VLM interval.
                track_frames_hit = False
                for track in inst.get('tracks', []):
                    for fd in track.get('frames', []):
                        f_idx = fd.get('frame_idx')
                        if f_idx is not None and 0 <= f_idx < num_frames:
                            scores[f_idx] = max(scores[f_idx], conf)
                            track_frames_hit = True

                # --- Fallback: flat VLM interval (only if no tracks) ---
                if not track_frames_hit:
                    s_f = max(0, int(inst['start_t'] * fps))
                    e_f = min(num_frames - 1, int(inst['end_t'] * fps))
                    if s_f <= e_f:
                        scores[s_f:e_f+1] = np.maximum(scores[s_f:e_f+1], conf)
            
            # Replace any remaining NaN values in scores with 0
            scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Align lengths
            L = min(len(gt_frames), len(scores))
            gt_slice = np.array(gt_frames[:L])
            sc_slice = scores[:L]
            # roc_auc_score is undefined (raises or gives nan) when only one class is present
            if L > 0 and len(np.unique(gt_slice)) > 1:
                auc_score = roc_auc_score(gt_slice, sc_slice)
                all_frame_scores.extend(sc_slice.tolist())
                all_frame_labels.extend(gt_slice.tolist())
                log(f"  [Frame] AUROC: {auc_score:.4f}")
            elif L > 0:
                all_frame_scores.extend(sc_slice.tolist())
                all_frame_labels.extend(gt_slice.tolist())
                log(f"  [Frame] AUROC: N/A (single class)")

        # C2. Object Level (TAO) — accumulate globally; compute once after loop
        # Priority 1: explicit --gt_tracks_dir (pre-made TAO .txt files)
        # Priority 2: auto-generate pseudo-tracks from pixel GT masks
        det_path = os.path.join(args.output_dir, f"{vid}.txt")
        gt_tracks_for_video = []

        if args.gt_tracks_dir:
            gt_path = os.path.join(args.gt_tracks_dir, f"{vid}.txt")
            if os.path.exists(gt_path):
                gt_tracks_for_video = read_tracks(gt_path)
        else:
            gt_tracks_for_video = loader.load_gt_tracks(vid)

        if gt_tracks_for_video:
            global_gt_tracks.extend(gt_tracks_for_video)
            global_total_frames += num_frames
            if os.path.exists(det_path):
                global_detections.extend(read_detected_anomalies(det_path))


        
        # C3. Pixel Level
        pixel_gt = loader.load_pixel_gt(vid)
        if pixel_gt:
            p_aurocs, p_aps, p_aupros, p_f1s = [], [], [], []
            for f_idx, gt_mask in pixel_gt.items():
                if f_idx >= num_frames: continue
                h, w = gt_mask.shape
                pred_mask = np.zeros((h, w), dtype=np.float32)

                for inst in results['anomalies']:
                    conf = float(np.clip(inst.get('confidence', 0.5), 0.0, 1.0))
                    for track in inst.get('tracks', []):
                        for fd in track.get('frames', []):
                            if fd['frame_idx'] != f_idx:
                                continue
                            sam_mask = fd.get('mask')          # uint8 H×W binary, or None
                            if sam_mask is not None:
                                # Resize to GT resolution if needed (SAM2 output is at video res)
                                if sam_mask.shape != (h, w):
                                    import cv2 as _cv2
                                    sam_mask = _cv2.resize(
                                        sam_mask, (w, h),
                                        interpolation=_cv2.INTER_NEAREST)
                                pred_mask = np.maximum(pred_mask, sam_mask.astype(np.float32) * conf)
                            else:
                                # Legacy fallback: fill bbox rectangle
                                box = fd.get('bbox')
                                if box:
                                    x1, y1, x2, y2 = map(int, box)
                                    pred_mask[y1:y2+1, x1:x2+1] = np.maximum(
                                        pred_mask[y1:y2+1, x1:x2+1], conf)

                pa, pap, paupro, pf1 = calculate_pixel_metrics(gt_mask, pred_mask)
                if pa is not None:
                    p_aurocs.append(pa)
                    p_aps.append(pap)
                    p_aupros.append(paupro)
                    p_f1s.append(pf1)

            if p_aurocs:
                vid_pa     = float(np.mean(p_aurocs))
                vid_pap    = float(np.mean(p_aps))
                vid_paupro = float(np.mean(p_aupros))
                vid_pf1    = float(np.mean(p_f1s))
                all_p_aurocs.append(vid_pa)
                all_p_aps.append(vid_pap)
                all_p_aupros.append(vid_paupro)
                all_p_f1s.append(vid_pf1)
                log(f"  [Pixel] AUROC:{vid_pa:.4f}  AP:{vid_pap:.4f}  AUPRO:{vid_paupro:.4f}  F1:{vid_pf1:.4f}")

        # D. Visualization (debug only — slow for 199 videos)
        if args.debug:
            from grid_vad.data.visualization import Annotator
            annotator = Annotator(vid_output_dir)
            annotator.create_annotated_video(
                vpath, results['anomalies'],
                output_name=f"{vid}_annotated.mp4")

    # ── Final Dataset-Level Summary Table ─────────────────────────────────
    #   Pixel-level (5 cols)          | Object-level (2 cols)
    #   Frame-AUROC  AUROC  AP  AUPRO  F1 | RBDC  TBDC
    PXW = 13   # pixel col width
    OBW = 10   # object col width
    PX_SECTION = PXW * 5 + 4   # 5 cols + 4 spaces between them = 69
    OB_SECTION = OBW * 2 + 1   # 2 cols + 1 space = 21
    SEP  = "+" + "-" * (PX_SECTION + 2) + "+" + "-" * (OB_SECTION + 2) + "+"
    HEAD = "| {:^{pw}s} | {:^{ow}s} |".format(
        "Pixel-level", "Object-level", pw=PX_SECTION, ow=OB_SECTION)

    def _col(names, w): return " ".join(f"{n:<{w}s}" for n in names)
    PX_NAMES = ["Frame-AUROC↑", "Px-AUROC↑", "Px-AP↑", "Px-AUPRO↑", "Px-F1↑"]
    OB_NAMES = ["RBDC↑", "TBDC↑"]
    COL = f"| {_col(PX_NAMES, PXW)} | {_col(OB_NAMES, OBW)} |"

    def _fmt(vals): return f"{np.mean(vals):.4f}" if vals else "N/A"

    # Frame-level dataset AUC
    frame_auc_str = "N/A"
    if all_frame_labels:
        arr_l = np.array(all_frame_labels)
        arr_s = np.nan_to_num(np.array(all_frame_scores), nan=0.0, posinf=1.0, neginf=0.0)
        if len(np.unique(arr_l)) > 1:
            frame_auc_str = f"{roc_auc_score(arr_l, arr_s):.4f}"

    # Object-level: single global computation (matches reference exactly)
    tbdc_str = rbdc_str = "N/A"
    if global_gt_tracks and global_detections:
        g_tbdc, g_rbdc = compute_dataset_tbdc_rbdc(
            all_gt_tracks=global_gt_tracks,
            all_detections=global_detections,
            total_frames=global_total_frames,
        )
        tbdc_str = f"{g_tbdc:.4f}"
        rbdc_str = f"{g_rbdc:.4f}"

    px_vals = [frame_auc_str, _fmt(all_p_aurocs), _fmt(all_p_aps), _fmt(all_p_aupros), _fmt(all_p_f1s)]
    ob_vals = [rbdc_str, tbdc_str]
    VAL = f"| {_col(px_vals, PXW)} | {_col(ob_vals, OBW)} |"

    print(f"\n{'=' * (PX_SECTION + OB_SECTION + 7)}")
    print(f"  DATASET RESULTS  —  {args.dataset.upper()}")
    print(SEP)
    print(HEAD)
    print(SEP)
    print(COL)
    print(SEP)
    print(VAL)
    print(SEP)
    print("\nEvaluation Complete.")

if __name__ == "__main__":
    main()

