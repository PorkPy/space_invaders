"""
Video recording utilities for smooth gameplay display
"""
import numpy as np
import tempfile
import os
from typing import List
import logging

logger = logging.getLogger(__name__)

class GameplayRecorder:
    """Records gameplay frames and creates smooth video"""
    
    def __init__(self, fps: int = 10, frame_size: tuple = (160, 210)):
        self.fps = fps
        self.frame_size = frame_size
        self.frames: List[np.ndarray] = []
        self.temp_file = None
        
    def add_frame(self, frame: np.ndarray):
        """Add a frame to the recording"""
        if frame is not None:
            try:
                import cv2
                # Resize frame to consistent size
                if len(frame.shape) == 2:  # Grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
                # Resize to target size
                resized_frame = cv2.resize(frame, self.frame_size)
                self.frames.append(resized_frame)
            except ImportError:
                logger.error("OpenCV not available for frame processing")
    
    def create_video_segment(self, num_frames: int = 50) -> str:
        """Create a video from recent frames"""
        if len(self.frames) < 2:
            return None
            
        try:
            import cv2
            
            # Create temporary video file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                temp_path = f.name
            
            # Use recent frames (last num_frames)
            recent_frames = self.frames[-num_frames:] if len(self.frames) > num_frames else self.frames
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, self.fps, self.frame_size)
            
            for frame in recent_frames:
                # OpenCV uses BGR, convert from RGB
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
            
            out.release()
            
            # Clean up old temp file
            if self.temp_file and os.path.exists(self.temp_file):
                os.unlink(self.temp_file)
            
            self.temp_file = temp_path
            return temp_path
            
        except ImportError:
            logger.error("OpenCV not available for video creation")
            return None
        except Exception as e:
            logger.error(f"Failed to create video: {e}")
            return None
    
    def clear_frames(self, keep_last: int = 10):
        """Clear old frames to prevent memory issues"""
        if len(self.frames) > keep_last:
            self.frames = self.frames[-keep_last:]
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_file and os.path.exists(self.temp_file):
            os.unlink(self.temp_file)
            self.temp_file = None