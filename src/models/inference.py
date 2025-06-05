"""
Model inference and prediction logic
"""
import logging
import numpy as np
from typing import Optional, Tuple
from stable_baselines3 import DQN
from src.models.model_loader import ModelLoader
from src.game.environment import SpaceInvadersEnvironment

logger = logging.getLogger(__name__)

class GameInference:
    """Handles RL model inference for game playing"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.environment = SpaceInvadersEnvironment()
        self.model: Optional[DQN] = None
        self.is_ready = False
        
    def initialize(self) -> bool:
        """Initialize model and environment"""
        try:
            # Create environment first
            if not self.environment.create_environment():
                logger.error("Failed to create environment")
                return False
            
            # Try to load the model (but don't fail if it doesn't work)
            self.model = self.model_loader.load_model()
            if self.model is None:
                logger.warning("Failed to load model - will use random agent")
            else:
                logger.info("Model loaded successfully")
            
            self.is_ready = True
            logger.info("Inference system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize inference system: {e}")
            return False
    
    def start_new_game(self) -> Optional[np.ndarray]:
        """Start a new game episode"""
        if not self.is_ready:
            logger.error("Inference system not initialized")
            return None
            
        return self.environment.reset()
    
    def predict_action(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, Optional[np.ndarray]]:
        """Predict the next action given current observation"""
        if not self.is_ready:
            logger.error("Model not ready for prediction")
            return 0, None
            
        # If model failed to load, use random agent as fallback
        if self.model is None:
            action_space_size = self.environment.get_action_space().n
            random_action = np.random.randint(0, action_space_size)
            logger.info(f"Using random action: {random_action}")
            return random_action, None
            
        try:
            action, _states = self.model.predict(
                observation, 
                deterministic=deterministic
            )
            return int(action), _states
            
        except Exception as e:
            logger.error(f"Failed to predict action: {e}")
            # Fallback to random action
            action_space_size = self.environment.get_action_space().n
            return np.random.randint(0, action_space_size), None
    
    def step_game(self, action: int) -> Tuple[Optional[np.ndarray], float, bool, dict]:
        """Take a step in the game with the given action"""
        if not self.is_ready:
            logger.error("Inference system not initialized")
            return None, 0.0, True, {}
            
        return self.environment.step(action)
    
    def play_step(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[Optional[np.ndarray], float, bool, dict, int]:
        """Complete game step: predict action and execute it"""
        # Predict action
        action, _ = self.predict_action(observation, deterministic)
        
        # Execute action
        next_obs, reward, done, info = self.step_game(action)
        
        return next_obs, reward, done, info, action
    
    def get_render_frame(self) -> Optional[np.ndarray]:
        """Get current game frame for visualization"""
        if not self.is_ready:
            return None
        return self.environment.render()
    
    def get_game_stats(self) -> dict:
        """Get current game statistics"""
        if not self.is_ready:
            return {}
        
        env_stats = self.environment.get_game_stats()
        model_info = self.model_loader.get_model_info()
        
        return {
            **env_stats,
            "model_status": model_info.get("status", "unknown"),
            "algorithm": model_info.get("algorithm", "unknown")
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.environment:
            self.environment.close()
        logger.info("Inference system cleaned up")