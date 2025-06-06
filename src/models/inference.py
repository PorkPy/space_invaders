"""
Model inference and prediction logic - Generic for all Atari games
"""
import logging
import numpy as np
from typing import Optional, Tuple
from stable_baselines3 import DQN
from src.models.model_loader import ModelLoader
from src.game.environment import AtariEnvironment

logger = logging.getLogger(__name__)

class GameInference:
    """Handles RL model inference for game playing"""
    
    def __init__(self, environment_name: str = "SpaceInvaders-v0"):
        self.environment_name = environment_name
        self.model_loader = ModelLoader()
        self.environment = AtariEnvironment(environment_name)
        self.model: Optional[DQN] = None
        self.is_ready = False
        
    def initialize(self) -> bool:
        """Initialize model and environment"""
        try:
            # Create environment first
            if not self.environment.create_environment():
                logger.error(f"Failed to create environment: {self.environment_name}")
                return False
            
            # Try to load the model (but don't fail if it doesn't work)
            # For multi-game demo, we'll use random agents for simplicity
            self.model = None  # self.model_loader.load_model()
            if self.model is None:
                logger.info(f"Using random agent for {self.environment_name}")
            else:
                logger.info(f"Model loaded successfully for {self.environment_name}")
            
            self.is_ready = True
            logger.info(f"Inference system initialized successfully: {self.environment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize inference system for {self.environment_name}: {e}")
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
            
        # For multi-game demo, use intelligent random agent
        if self.model is None:
            action_space_size = self.environment.get_action_space().n
            
            # Simple heuristic: bias towards action (FIRE) for shooting games
            if "SpaceInvaders" in self.environment_name or "Asteroids" in self.environment_name:
                # 40% chance of firing, 60% other actions
                if np.random.random() < 0.4:
                    action = 1  # FIRE action in most Atari games
                else:
                    action = np.random.randint(0, action_space_size)
            else:
                # Pure random for other games
                action = np.random.randint(0, action_space_size)
                
            return action, None
            
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
            "model_status": "random_agent",
            "algorithm": "Random"
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.environment:
            self.environment.close()
        logger.info(f"Inference system cleaned up: {self.environment_name}")