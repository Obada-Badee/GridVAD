import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import glob
from scipy.io import loadmat


class BenchmarkLoader:
    """
    Supports three dataset layouts:

    ShanghaiTech — Official layout (preferred):
        <root>/frames/<vid>/001.jpg ...
        <root>/test_frame_mask/<vid>.npy   (1-D binary, one entry per frame)
        <root>/test_pixel_mask/<vid>.npy   (H×W×T binary pixel masks)

    ShanghaiTech — Kaggle layout:
        <root>/frames/<vid>/0001.jpg ...
        <root>/label/<vid>.npy             (1-D binary, one entry per frame)

    UCSD Ped2:
        <root>/Test/TestXXX/001.tif ...
        <root>/Test/TestXXX_gt/*.bmp       (per-frame pixel masks)
    """

    def __init__(self, dataset_name: str, root_dir: str):
        self.dataset_name = dataset_name.lower()
        self.root_dir = root_dir
        self._layout = self._detect_layout()

    def _detect_layout(self) -> str:
        """Detect which dataset layout is present at root_dir."""
        if self.dataset_name == 'shanghaitech':
            # Official: has test_frame_mask/ subdirectory
            if os.path.isdir(os.path.join(self.root_dir, 'test_frame_mask')):
                return 'shanghaitech_official'
        elif self.dataset_name == 'ucsd_ped2':
            return 'ucsd_ped2'
        raise ValueError(f"Unknown dataset: {self.dataset_name}")

    # ── Video enumeration ──────────────────────────────────────────────────

    def get_video_paths(self) -> List[str]:
        """Returns sorted list of video IDs."""
        if self.dataset_name == 'shanghaitech':
            frames_dir = os.path.join(self.root_dir, 'frames')
            if os.path.isdir(frames_dir):
                dirs = glob.glob(os.path.join(frames_dir, '*'))
                return sorted([os.path.basename(d) for d in dirs if os.path.isdir(d)])
            # mp4 fallback
            video_dir = os.path.join(self.root_dir, 'videos')
            files = glob.glob(os.path.join(video_dir, '*.mp4'))
            return sorted([os.path.splitext(os.path.basename(f))[0] for f in files])

        elif self.dataset_name == 'ucsd_ped2':
            test_dir = os.path.join(self.root_dir, 'Test')
            if not os.path.isdir(test_dir):
                test_dir = os.path.join(self.root_dir, 'test')
            if not os.path.isdir(test_dir):
                test_dir = self.root_dir
            dirs = glob.glob(os.path.join(test_dir, 'Test*'))
            video_dirs = [d for d in dirs if '_gt' not in d and os.path.isdir(d)]
            return sorted([os.path.basename(d) for d in video_dirs])

        raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def get_video_file_path(self, video_id: str) -> str:
        """
        Returns the path (file or sequence pattern) to pass to VideoReader.
        For image-sequence folders, returns '<dir>/%03d.jpg' which triggers
        VideoReader's glob-all-and-sort mode.
        """
        if self.dataset_name == 'shanghaitech':
            frames_dir = os.path.join(self.root_dir, 'frames', video_id)
            if os.path.isdir(frames_dir):
                # Find extension from the first file
                sample = sorted(glob.glob(os.path.join(frames_dir, '*.*')))
                ext = os.path.splitext(sample[0])[1] if sample else '.jpg'
                # The %03d triggers VideoReader's image-sequence mode;
                # actual filenames are globbed and sorted numerically.
                return os.path.join(frames_dir, f'%03d{ext}')
            # mp4 fallback
            return os.path.join(self.root_dir, 'videos', f'{video_id}.mp4')

        elif self.dataset_name == 'ucsd_ped2':
            test_root = os.path.join(self.root_dir, 'Test')
            img_path = os.path.join(test_root, video_id, '001.tif')
            if os.path.exists(img_path):
                return os.path.join(test_root, video_id, '%03d.tif')
            return os.path.join(test_root, video_id)  # avi fallback

        raise ValueError(f"Unknown dataset: {self.dataset_name}")

    # ── Frame-level GT ─────────────────────────────────────────────────────

    def load_gt_frames(self, video_id: str) -> np.ndarray:
        """Returns 1-D binary array (1=anomaly) indexed by frame number."""
        if self.dataset_name == 'shanghaitech':
            # Official: test_frame_mask/<vid>.npy
            npy = os.path.join(self.root_dir, 'test_frame_mask', f'{video_id}.npy')
            if os.path.exists(npy):
                return np.load(npy, allow_pickle=True).astype(np.int32).flatten()
            # Kaggle: label/<vid>.npy
            npy = os.path.join(self.root_dir, 'label', f'{video_id}.npy')
            if os.path.exists(npy):
                return np.load(npy).astype(np.int32).flatten()
            return np.array([])

        elif self.dataset_name == 'ucsd_ped2':
            # Derive from pixel masks: a frame is positive if its mask is non-zero
            gt_masks = self.load_pixel_gt(video_id)
            if gt_masks:
                max_idx = max(gt_masks.keys())
                gt_frames = np.zeros(max_idx + 1, dtype=np.int32)
                for idx, mask in gt_masks.items():
                    if np.sum(mask) > 0:
                        gt_frames[idx] = 1
                return gt_frames
            # .mat fallback
            for mat_path in [
                os.path.join(self.root_dir, 'Test', f'{video_id}_gt.mat'),
                os.path.join(self.root_dir, f'{video_id}_gt.mat'),
            ]:
                if os.path.exists(mat_path):
                    try:
                        mat = loadmat(mat_path)
                        return mat['gt'].flatten().astype(np.int32)
                    except Exception:
                        pass
            return np.array([])

        raise ValueError(f"Unknown dataset: {self.dataset_name}")

    # ── Pixel-level GT ─────────────────────────────────────────────────────

    def load_pixel_gt(self, video_id: str) -> Dict[int, np.ndarray]:
        """
        Returns {frame_idx: binary_mask_H_W}.
        Only returns frames where the mask is non-zero (or all frames if unclear).
        """
        if self.dataset_name == 'shanghaitech':
            npy = os.path.join(self.root_dir, 'test_pixel_mask', f'{video_id}.npy')
            if not os.path.exists(npy):
                return {}
            data = np.load(npy, allow_pickle=True)
            # Possible shapes: (H, W, T), (T, H, W)
            # Normalise to (T, H, W)
            if data.ndim == 3:
                if data.shape[2] < data.shape[0]:
                    # Likely (H, W, T) — channels-last
                    data = np.transpose(data, (2, 0, 1))
                # else already (T, H, W) or square — use as-is
            elif data.ndim == 2:
                # Single mask — treat as frame 0
                data = data[np.newaxis]
            else:
                return {}

            gt_masks = {}
            for frame_idx in range(data.shape[0]):
                mask = (data[frame_idx] > 0).astype(np.uint8)
                if np.sum(mask) > 0:       # only store non-empty masks
                    gt_masks[frame_idx] = mask
            return gt_masks

        elif self.dataset_name == 'ucsd_ped2':
            gt_dir = os.path.join(self.root_dir, 'Test', f'{video_id}_gt')
            if not os.path.isdir(gt_dir):
                gt_dir = os.path.join(self.root_dir, f'{video_id}_gt')
            if not os.path.isdir(gt_dir):
                return {}
            gt_masks = {}
            for fpath in sorted(glob.glob(os.path.join(gt_dir, '*.bmp'))):
                fname = os.path.splitext(os.path.basename(fpath))[0]
                try:
                    fidx = int(fname) - 1   # 1-based filenames → 0-based index
                except ValueError:
                    continue
                mask = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    gt_masks[fidx] = (mask > 0).astype(np.uint8)
            return gt_masks

        raise ValueError(f"Unknown dataset: {self.dataset_name}")

    # ── Object-level GT tracks (for RBDC/TBDC) ────────────────────────────

    def load_gt_tracks(self, video_id: str):
        """
        Generates ContinuousTrack objects from pixel masks by running
        connected-components per frame and linking blobs across frames by
        maximum-IoU. Returns [] when no pixel GT is available.
        """
        from grid_vad.eval.tao_object import ContinuousTrack

        pixel_gt = self.load_pixel_gt(video_id)
        if not pixel_gt:
            return []

        # Step 1: per-frame connected components → list of bbox regions
        frame_blobs = {}   # frame_idx -> list of [x1,y1,x2,y2]
        for frame_idx in sorted(pixel_gt.keys()):
            mask = pixel_gt[frame_idx]
            n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
            blobs = []
            for lbl in range(1, n_labels):
                ys, xs = np.where(labels == lbl)
                blobs.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
            if blobs:
                frame_blobs[frame_idx] = blobs

        if not frame_blobs:
            return []

        # Step 2: greedy IoU-based track linking across consecutive frames
        def _iou(a, b):
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            iw, ih = max(0, ix2-ix1+1), max(0, iy2-iy1+1)
            inter = iw * ih
            aa = (a[2]-a[0]+1) * (a[3]-a[1]+1)
            ab = (b[2]-b[0]+1) * (b[3]-b[1]+1)
            denom = aa + ab - inter
            return inter / denom if denom > 0 else 0.0

        IOU_LINK = 0.3   # minimum IoU to link blobs into the same track
        tracks: List[ContinuousTrack] = []
        active_tracks: List[Tuple[ContinuousTrack, List[float]]] = []  # (track, last_bbox)

        for frame_idx in sorted(frame_blobs.keys()):
            blobs = frame_blobs[frame_idx]
            matched_blob = set()
            matched_track = set()
            pairs = []
            for ti, (trk, last_box) in enumerate(active_tracks):
                for bi, blob in enumerate(blobs):
                    iou = _iou(last_box, blob)
                    if iou >= IOU_LINK:
                        pairs.append((iou, ti, bi))
            pairs.sort(reverse=True)
            for iou, ti, bi in pairs:
                if ti in matched_track or bi in matched_blob:
                    continue
                trk, _ = active_tracks[ti]
                trk.bboxes[frame_idx] = blobs[bi]
                trk.end_idx = frame_idx
                active_tracks[ti] = (trk, blobs[bi])
                matched_track.add(ti)
                matched_blob.add(bi)
            # New tracks for unmatched blobs
            for bi, blob in enumerate(blobs):
                if bi not in matched_blob:
                    trk = ContinuousTrack(start_idx=frame_idx, end_idx=frame_idx,
                                          video_name=video_id)
                    trk.bboxes[frame_idx] = blob
                    active_tracks.append((trk, blob))
                    tracks.append(trk)
            # Retire tracks not seen for 2+ frames
            new_active = []
            for ti, (trk, last_box) in enumerate(active_tracks):
                if trk.end_idx >= frame_idx - 1:
                    new_active.append((trk, last_box))
            active_tracks = new_active

        return tracks
