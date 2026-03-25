import cv2
import os
import numpy as np
from typing import Generator, List, Tuple, Dict, Any

import glob

class VideoReader:
    def __init__(self, video_path: str):
        self.mode = 'video'
        self.images = []
        
        # Check if it's a pattern or explicit sequence request
        if '%' in video_path or '*' in video_path:
            self.mode = 'sequence'
            
            parent_dir = os.path.dirname(video_path)
            if not os.path.exists(parent_dir):
                raise FileNotFoundError(f"Sequence directory not found: {parent_dir}")
                
            # Determine extension from path or just grab all likely images
            # video_path is like .../Test001/%03d.tif
            # Split ext
            _, ext = os.path.splitext(video_path)
            
            # Glob
            search_pattern = os.path.join(parent_dir, f"*{ext}")
            files = glob.glob(search_pattern)
            
            # Sort numerically if possible, else alphabetically
            # Usually files are 001.tif, 002.tif... so string sort works or we can be fancy
            files = sorted(files)
            
            if not files:
                 raise FileNotFoundError(f"No files found matching {search_pattern}")
                 
            self.images = files
            self.video_path = video_path
            
            # Read metadata from first frame
            frame0 = cv2.imread(self.images[0])
            if frame0 is None:
                raise IOError(f"Could not read first image: {self.images[0]}")
                
            self.height, self.width = frame0.shape[:2]
            self.frame_count = len(self.images)
            self.fps = 25.0 # Default for image sequences (UCSD/ShanghaiTech behavior)
            
        else:
            # Standard video file
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found at: {video_path}")
            
            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                raise IOError(f"Could not open video: {video_path}")
                
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration_sec = self.frame_count / self.fps if self.fps > 0 else 0

    @property
    def duration_sec(self):
        if self.fps > 0:
            return self.frame_count / self.fps
        return 0

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Random access to a specific frame."""
        if self.mode == 'sequence':
            if 0 <= frame_idx < len(self.images):
                frame = cv2.imread(self.images[frame_idx])
                if frame is None:
                    raise ValueError(f"Failed to load image: {self.images[frame_idx]}")
                return frame
            else:
                raise IndexError(f"Frame index {frame_idx} out of bounds (0-{len(self.images)-1})")
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError(f"Could not read frame {frame_idx}")
            return frame

    def iter_frames(self, start_idx: int = 0, end_idx: int = None) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """Iterate frames from start to end (exclusive). Yields (idx, frame_bgr, timestamp)."""
        if end_idx is None:
            end_idx = self.frame_count
            
        if self.mode == 'sequence':
            for idx in range(start_idx, min(end_idx, len(self.images))):
                frame = cv2.imread(self.images[idx])
                if frame is None:
                    continue # or break?
                timestamp = idx / self.fps
                yield idx, frame, timestamp
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            for idx in range(start_idx, end_idx):
                ret, frame = self.cap.read()
                if not ret:
                    break
                timestamp = idx / self.fps
                yield idx, frame, timestamp

    def close(self):
        if self.mode == 'video' and hasattr(self, 'cap'):
            self.cap.release()

class ClipGenerator:
    def __init__(self, clip_len_sec: float, overlap_sec: float):
        self.clip_len_sec = clip_len_sec
        self.overlap_sec = overlap_sec

    def generate_clips(self, video_reader: VideoReader) -> List[Dict[str, Any]]:
        """
        Splits video into overlapping clips.
        Returns list of clip metadata.
        """
        clips = []
        stride_sec = self.clip_len_sec - self.overlap_sec
        if stride_sec <= 0:
            raise ValueError("Overlap must be smaller than clip length")

        current_start_sec = 0.0
        clip_id = 0

        while current_start_sec < video_reader.duration_sec:
            current_end_sec = min(current_start_sec + self.clip_len_sec, video_reader.duration_sec)
            
            # Convert to frame indices
            start_frame = int(current_start_sec * video_reader.fps)
            end_frame = int(current_end_sec * video_reader.fps)
            
            # Ensure we don't go out of bounds
            end_frame = min(end_frame, video_reader.frame_count)
            
            clips.append({
                "clip_id": clip_id,
                "start_time": current_start_sec,
                "end_time": current_end_sec,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "video_path": video_reader.video_path
            })

            if end_frame == video_reader.frame_count:
                break
                
            current_start_sec += stride_sec
            clip_id += 1
            
        return clips
