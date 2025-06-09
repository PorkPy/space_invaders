"""
Model management system for switching between different trained models
"""
import logging
import numpy as np
import requests
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any
from stable_baselines3 import DQN
import tempfile
import os

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages different types of models (pre-trained, custom, random)"""
    
    def __init__(self, game_name: str):
        self.game_name = game_name
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Available model types
        self.model_configs = {
            "random": {
                "name": "Random Agent",
                "description": "Random actions with game-specific heuristics",
                "type": "random"
            },
            "pretrained": {
                "name": "Pre-trained DQN",
                "description": "Pre-trained model from RL Baselines Zoo",
                "type": "dqn",
                "url": self._get_pretrained_url(game_name)
            },
            "custom": {
                "name": "Your Model",
                "description": "Your custom trained model (when available)",
                "type": "dqn",
                "path": self.models_dir / f"{game_name}_custom.zip"
            }
        }
        
        self.current_model = None
        self.current_model_type = "random"
    
    def _get_pretrained_url(self, game_name: str) -> str:
        """Get pre-trained model URL for the game"""
        # RL Baselines Zoo models
        model_urls = {
            "SpaceInvaders": "https://huggingface.co/sb3/dqn-SpaceInvadersNoFrameskip-v4/resolve/main/dqn-SpaceInvadersNoFrameskip-v4.zip",
            "Breakout": "https://huggingface.co/sb3/dqn-BreakoutNoFrameskip-v4/resolve/main/dqn-BreakoutNoFrameskip-v4.zip",
            "Pong": "https://huggingface.co/sb3/dqn-PongNoFrameskip-v4/resolve/main/dqn-PongNoFrameskip-v4.zip",
        }
        
        # Extract base game name
        base_name = game_name.split("-")[0]  # "SpaceInvaders-v4" -> "SpaceInvaders"
        return model_urls.get(base_name, "")
    
    def download_pretrained_model(self) -> bool:
        """Download pre-trained model if available"""
        config = self.model_configs["pretrained"]
        if not config.get("url"):
            logger.warning(f"No pre-trained model available for {self.game_name}")
            return False
        
        model_path = self.models_dir / f"{self.game_name}_pretrained.zip"
        
        # Check if already downloaded
        if model_path.exists():
            logger.info(f"Pre-trained model already exists: {model_path}")
            return True
        
        try:
            logger.info(f"Downloading pre-trained model for {self.game_name}...")
            response = requests.get(config["url"], stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded pre-trained model: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download pre-trained model: {e}")
            return False
    
    def load_model(self, model_type: str) -> bool:
        """Load specified model type"""
        try:
            if model_type == "random":
                self.current_model = None
                self.current_model_type = "random"
                logger.info("Loaded random agent")
                return True
            
            elif model_type == "pretrained":
                # Download if needed
                if not self.download_pretrained_model():
                    return False
                
                model_path = self.models_dir / f"{self.game_name}_pretrained.zip"
                self.current_model = DQN.load(str(model_path))
                self.current_model_type = "pretrained"
                logger.info(f"Loaded pre-trained model: {model_path}")
                return True
            
            elif model_type == "custom":
                model_path = self.model_configs["custom"]["path"]
                if not model_path.exists():
                    logger.warning(f"Custom model not found: {model_path}")
                    return False
                
                self.current_model = DQN.load(str(model_path))
                self.current_model_type = "custom"
                logger.info(f"Loaded custom model: {model_path}")
                return True
            
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model {model_type}: {e}")
            return False
    
    def predict_action(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Predict action using current model"""
        if self.current_model_type == "random":
            return self._random_action(observation)
        
        elif self.current_model is not None:
            try:
                action, _ = self.current_model.predict(observation, deterministic=deterministic)
                return int(action)
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                return self._random_action(observation)
        
        else:
            return self._random_action(observation)
    
    def _random_action(self, observation: np.ndarray) -> int:
        """Generate random action with game-specific heuristics"""
        # Assume 6 actions for Atari games
        action_space_size = 6
        
        # Game-specific heuristics
        if "SpaceInvaders" in self.game_name:
            # Bias towards firing
            if np.random.random() < 0.4:
                return 1  # FIRE
            else:
                return np.random.randint(0, action_space_size)
        else:
            return np.random.randint(0, action_space_size)
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models with their status"""
        available = {}
        
        for model_id, config in self.model_configs.items():
            status = {
                **config,
                "available": False,
                "current": model_id == self.current_model_type
            }
            
            if model_id == "random":
                status["available"] = True
            elif model_id == "pretrained":
                model_path = self.models_dir / f"{self.game_name}_pretrained.zip"
                status["available"] = model_path.exists() or bool(config.get("url"))
            elif model_id == "custom":
                status["available"] = config["path"].exists()
            
            available[model_id] = status
        
        return available
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about currently loaded model"""
        config = self.model_configs.get(self.current_model_type, {})
        return {
            "type": self.current_model_type,
            "name": config.get("name", "Unknown"),
            "description": config.get("description", ""),
            "loaded": self.current_model is not None or self.current_model_type == "random"
        }
    