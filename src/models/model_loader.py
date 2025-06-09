"""
RL Model loading and management
"""
import logging
import requests
from pathlib import Path
from typing import Optional
import ale_py  # Import first to register ALE environments
from stable_baselines3 import DQN
from config.settings import MODEL_CONFIG
import gymnasium as gym
import gym as old_gym  # Some models need the old gym interface

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles loading and caching of RL models"""
    
    def __init__(self):
        self.model: Optional[DQN] = None
        self.model_path = MODEL_CONFIG["local_model_path"]
        self.model_url = MODEL_CONFIG["model_url"]
        
    def download_model(self) -> bool:
        """Download pre-trained model if not cached locally"""
        if self.model_path.exists():
            logger.info(f"Model already exists at {self.model_path}")
            return True
            
        try:
            logger.info(f"Downloading model from {self.model_url}")
            response = requests.get(self.model_url, stream=True)
            response.raise_for_status()
            
            with open(self.model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"Model downloaded successfully to {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False
    
    def load_model(self) -> Optional[DQN]:
        """Load the RL model, downloading if necessary"""
        if self.model is not None:
            return self.model
            
        # Download model if needed
        if not self.download_model():
            return None
            
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = DQN.load(
                self.model_path,
                device=MODEL_CONFIG["device"]
            )
            logger.info("Model loaded successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def is_model_ready(self) -> bool:
        """Check if model is loaded and ready for inference"""
        return self.model is not None
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if not self.is_model_ready():
            return {"status": "not_loaded"}
            
        return {
            "status": "loaded",
            "algorithm": MODEL_CONFIG["algorithm"],
            "device": MODEL_CONFIG["device"],
            "model_path": str(self.model_path),
        }
    