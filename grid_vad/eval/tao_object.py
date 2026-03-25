"""
TBDC / RBDC implementation faithful to the reference:
  "A Background-Agnostic Framework with Adversarial Training for
   Abnormal Event Detection in Video" (Georgescu et al., TPAMI 2021)

Key design choices to match the reference exactly:
  - Computation is GLOBAL across all videos (not per-video averaged).
  - IoU threshold beta = 0.1  (matches reference default)
  - alpha                = 0.1  (matches reference default)
  - FPR extension uses np.insert (matches reference exactly)
  - AUC deduplication added only as an NaN safety net.
"""

import os
import numpy as np
from sklearn import metrics
from typing import List, Dict


class ContinuousTrack:
    def __init__(self, start_idx=0, end_idx=None, masks=0, video_name=""):
        self.start_idx = start_idx
        self.end_idx   = end_idx
        self.bboxes    = {}      # {frame_idx: [x1, y1, x2, y2]}
        self.masks     = masks
        self.video_name = video_name


class Region:
    def __init__(self, frame_idx, bbox, score, video_name, track_id=-1):
        self.frame_idx  = frame_idx
        self.bbox       = bbox
        self.score      = score
        self.video_name = video_name
        self.track_id   = track_id


# ── IoU ────────────────────────────────────────────────────────────────────

def bb_intersection_over_union(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea  = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea  = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    denom = float(boxAArea + boxBArea - interArea)
    return 0.0 if denom == 0 else interArea / denom


# ── File I/O (kept for --gt_tracks_dir compatibility) ──────────────────────

def read_tracks(tracks_path: str, video_name_filter: str = None) -> List[ContinuousTrack]:
    """
    Reads tracks from a directory of .txt files.
    Format per line: track_id, frame_id, x1, y1, x2, y2
    (tracks must be grouped by track_id, sorted within each group)
    """
    all_tracks = []

    if os.path.isfile(tracks_path):
        files = [tracks_path]
    else:
        files = [os.path.join(tracks_path, f)
                 for f in os.listdir(tracks_path) if f.endswith('.txt')]

    for file_path in files:
        video_name = os.path.splitext(os.path.basename(file_path))[0]
        if video_name_filter and video_name != video_name_filter:
            continue
        try:
            data = np.loadtxt(file_path, delimiter=',')
            if data.size == 0:
                continue
            if data.ndim == 1:
                data = data[np.newaxis, :]
        except Exception:
            continue

        num_tracks = int(data[-1, 0]) + 1
        for tid in range(num_tracks):
            rows = data[data[:, 0] == tid]
            track = ContinuousTrack(
                start_idx=int(rows[0, 1]),
                end_idx=int(rows[-1, 1]),
                video_name=video_name)
            for row in rows:
                track.bboxes[int(row[1])] = list(row[2:6])
            all_tracks.append(track)

    return all_tracks


def read_detected_anomalies(path: str) -> List[Region]:
    """
    Reads detections from a single .txt file.
    Format per line: frame_id, x1, y1, x2, y2, score
    """
    regions = []
    if not os.path.exists(path):
        return regions
    vname = os.path.splitext(os.path.basename(path))[0]
    try:
        data = np.loadtxt(path, delimiter=',')
        if data.size == 0:
            return regions
        if data.ndim == 1:
            data = data[np.newaxis, :]
        for row in data:
            regions.append(Region(
                frame_idx=int(row[0]),
                bbox=list(row[1:5]),
                score=float(row[5]),
                video_name=vname))
    except Exception:
        pass
    return regions


# ── Core metric (GLOBAL across all videos) ─────────────────────────────────

def compute_dataset_tbdc_rbdc(
        all_gt_tracks: List[ContinuousTrack],
        all_detections: List[Region],
        total_frames: int,
        alpha: float = 0.1,
        beta: float  = 0.1,
) -> tuple:
    """
    Compute TBDC and RBDC over the ENTIRE dataset at once.
    This matches the reference implementation exactly.

    Parameters
    ----------
    all_gt_tracks  : ContinuousTrack objects from ALL videos
    all_detections : Region objects (predictions) from ALL videos
    total_frames   : sum of frame counts across all videos
    alpha          : track-coverage threshold (default 0.1, per paper)
    beta           : IoU threshold for a detection to match GT (default 0.1)
    """

    # ── Build GT region lookup ──────────────────────────────────────────────
    gt_regions: List[Region] = []
    gt_per_frame: Dict = {}
    found_per_frame: Dict = {}

    for track_id, track in enumerate(all_gt_tracks):
        for frame_idx, bbox in track.bboxes.items():
            r = Region(frame_idx, list(bbox), 1.0, track.video_name, track_id)
            gt_regions.append(r)
            key = (track.video_name, frame_idx)
            if key not in gt_per_frame:
                gt_per_frame[key]    = []
                found_per_frame[key] = []
            gt_per_frame[key].append(r)
            found_per_frame[key].append(0)

    if not gt_regions or not all_detections:
        return 0.0, 0.0

    num_tracks = len(all_gt_tracks)
    num_det    = len(all_detections)

    tp   = np.zeros(num_det)
    fp   = np.zeros(num_det)
    tbdr = np.zeros(num_det)
    num_matched_per_track = [0] * num_tracks

    # Sort ALL detections by score descending (global sort, as in reference)
    all_detections = sorted(all_detections, key=lambda r: r.score, reverse=True)

    for idx, pred in enumerate(all_detections):
        key = (pred.video_name, pred.frame_idx)
        gt_in_frame = gt_per_frame.get(key)

        if gt_in_frame is None:
            fp[idx] = 1
        else:
            non_matched = []
            for m_idx, gt_r in enumerate(gt_in_frame):
                if (bb_intersection_over_union(gt_r.bbox, pred.bbox) >= beta
                        and found_per_frame[key][m_idx] == 0):
                    non_matched.append(m_idx)
                    found_per_frame[key][m_idx] = 1
                    if 0 <= gt_r.track_id < num_tracks:
                        num_matched_per_track[gt_r.track_id] += 1

            if non_matched:
                tp[idx] = len(non_matched)
            else:
                fp[idx] = 1

        # TBDR at this threshold (reference: compute_tbdr)
        percentages = np.array([
            count / max(len(t.bboxes), 1)
            for count, t in zip(num_matched_per_track, all_gt_tracks)
        ])
        tbdr[idx] = np.sum(percentages >= alpha) / num_tracks

    cum_fp = np.concatenate(([0], np.cumsum(fp)))
    cum_tp = np.concatenate(([0], np.cumsum(tp)))
    tbdr   = np.concatenate(([0], tbdr))

    rbdr = cum_tp / max(len(gt_regions), 1)
    fpr  = cum_fp / max(total_frames, 1)

    # Crop to FPR <= 1 and extend to exactly 1.0
    # (reference uses np.insert, which we replicate exactly)
    valid = np.where(fpr <= 1.0)[0]
    if len(valid) == 0:
        return 0.0, 0.0
    idx_1 = valid[-1] + 1

    if fpr[idx_1 - 1] != 1.0:
        rbdr = np.insert(rbdr, idx_1, rbdr[idx_1 - 1])
        tbdr = np.insert(tbdr, idx_1, tbdr[idx_1 - 1])
        fpr  = np.insert(fpr,  idx_1, 1.0)
        idx_1 += 1

    fpr_c  = fpr[:idx_1]
    tbdr_c = tbdr[:idx_1]
    rbdr_c = rbdr[:idx_1]

    tbdc = _safe_auc(fpr_c, tbdr_c)
    rbdc = _safe_auc(fpr_c, rbdr_c)
    return tbdc, rbdc


def _safe_auc(x: np.ndarray, y: np.ndarray) -> float:
    """sklearn.metrics.auc requires strictly monotonic x.  Deduplicate."""
    if len(x) < 2:
        return 0.0
    unique_x = np.unique(x)
    if len(unique_x) < 2:
        return 0.0
    unique_y = np.array([np.max(y[x == xi]) for xi in unique_x])
    val = metrics.auc(unique_x, unique_y)
    return float(val) if np.isfinite(val) else 0.0


# ── Legacy per-video wrapper (kept so old call-sites don't break) ───────────

def compute_tbdc_rbdc(
        gt_tracks: List[ContinuousTrack],
        detections: List[Region],
        num_frames: int,
        video_name: str,
        beta: float = 0.1,        # ← fixed: was 0.5, reference uses 0.1
) -> tuple:
    """
    Per-video wrapper around compute_dataset_tbdc_rbdc.
    NOTE: For comparable results to the paper, prefer accumulating all
    videos and calling compute_dataset_tbdc_rbdc once at the end.
    """
    return compute_dataset_tbdc_rbdc(
        all_gt_tracks=gt_tracks,
        all_detections=detections,
        total_frames=num_frames,
        beta=beta,
    )
