"""
Enhanced game inference with model switching and synchronized dual outputs
"""
import logging
import numpy as np
from typing import Optional, Tuple
from src.game.dual_environment import DualOutputEnvironment
from src.models.model_manager import ModelManager

logger = logging.getLogger(__name__)

class EnhancedGameInference:
    """Enhanced inference system with model switching and synchronized dual outputs"""
    
    def __init__(self, environment_name: str = "SpaceInvaders-v4"):
        self.environment_name = environment_name
        self.dual_env = DualOutputEnvironment(environment_name)
        self.model_manager = ModelManager(environment_name)
        self.is_ready = False
        
    def initialize(self, model_type: str = "random") -> bool:
        """Initialize synchronized environment and load specified model"""
        try:
            # Create synchronized dual-output environment
            if not self.dual_env.create_environment():
                logger.error("Failed to create dual-output environment")
                return False
            
            # Load specified model
            if not self.model_manager.load_model(model_type):
                logger.warning(f"Failed to load {model_type} model, falling back to random")
                self.model_manager.load_model("random")
            
            self.is_ready = True
            logger.info(f"Enhanced inference initialized: {self.environment_name} with {model_type} model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced inference: {e}")
            return False
    
    def switch_model(self, model_type: str) -> bool:
        """Switch to a different model type"""
        return self.model_manager.load_model(model_type)
    
    def start_new_game(self) -> Optional[np.ndarray]:
        """Start a new game episode"""
        if not self.is_ready:
            logger.error("Enhanced inference not initialized")
            return None
            
        return self.dual_env.reset()
    
    def play_step(self, deterministic: bool = True) -> Tuple[Optional[np.ndarray], float, bool, dict, int]:
        """Complete game step: predict action and execute it on synchronized environment"""
        if not self.is_ready:
            logger.error("Enhanced inference not ready")
            return None, 0.0, True, {}, 0
        
        # Get current agent observation (preprocessed)
        agent_obs = self.dual_env.get_agent_observation()
        if agent_obs is None:
            return None, 0.0, True, {}, 0
        
        # Predict action using current model
        action = self.model_manager.predict_action(agent_obs, deterministic)
        
        # Execute action in synchronized environment (both agent and display updated)
        next_obs, reward, done, info = self.dual_env.step(action)
        
        return next_obs, reward, done, info, action
    
    def get_display_frame(self) -> Optional[np.ndarray]:
        """Get current high-resolution display frame"""
        if not self.is_ready:
            return None
        return self.dual_env.get_display_frame()
    
    def get_agent_observation(self) -> Optional[np.ndarray]:
        """Get current agent observation (preprocessed)"""
        if not self.is_ready:
            return None
        return self.dual_env.get_agent_observation()
    
    def get_game_stats(self) -> dict:
        """Get current game statistics including model info"""
        if not self.is_ready:
            return {}
        
        env_stats = self.dual_env.get_game_stats()
        model_info = self.model_manager.get_current_model_info()
        
        return {
            **env_stats,
            "model_type": model_info["type"],
            "model_name": model_info["name"],
            "model_description": model_info["description"]
        }
    
    def get_available_models(self) -> dict:
        """Get available models for this game"""
        return self.model_manager.get_available_models()
    
    def cleanup(self):
        """Clean up resources"""
        if self.dual_env:
            self.dual_env.close()
        logger.info(f"Enhanced inference cleaned up: {self.environment_name}")
        