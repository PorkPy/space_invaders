"""
Atari game environment setup and management
"""
import logging
import ale_py  # Import this first to register ALE environments
import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Any
from config.settings import GAME_CONFIG

logger = logging.getLogger(__name__)

class SpaceInvadersEnvironment:
    """Manages the Space Invaders game environment"""
    
    def __init__(self):
        self.env: Optional[gym.Env] = None
        self.current_obs: Optional[np.ndarray] = None
        self.total_reward = 0
        self.episode_steps = 0
        self.game_over = False
        
    def create_environment(self) -> bool:
        """Create and configure the Atari environment"""
        try:
            # Create base environment
            self.env = gym.make(
                GAME_CONFIG["environment_name"],
                render_mode=GAME_CONFIG["render_mode"],
                max_episode_steps=None  # Remove step limit
            )
            
            # Try WITHOUT AtariPreprocessing to see raw game
            # Comment out preprocessing to see if shots become visible
            """
            # Apply Atari wrappers for preprocessing
            self.env = gym.wrappers.AtariPreprocessing(
                self.env,
                noop_max=30,
                frame_skip=GAME_CONFIG["frame_skip"],
                screen_size=GAME_CONFIG["screen_size"],
                terminal_on_life_loss=False,  # Changed: Don't end on life loss
                grayscale_obs=True,
                grayscale_newaxis=False,
                scale_obs=True
            )
            
            # Stack frames for temporal information
            self.env = gym.wrappers.FrameStackObservation(
                self.env, 
                GAME_CONFIG["frame_stack"]
            )
            """
            
            logger.info("Environment created successfully (RAW - no preprocessing)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            return False
    
    def reset(self) -> Optional[np.ndarray]:
        """Reset the environment for a new episode"""
        if self.env is None:
            logger.error("Environment not created")
            return None
            
        try:
            self.current_obs, _ = self.env.reset()
            self.total_reward = 0
            self.episode_steps = 0
            self.game_over = False
            logger.info("Environment reset successfully")
            return self.current_obs
            
        except Exception as e:
            logger.error(f"Failed to reset environment: {e}")
            return None
    
    def step(self, action: int) -> Tuple[Optional[np.ndarray], float, bool, dict]:
        """Take a step in the environment"""
        if self.env is None or self.current_obs is None:
            logger.error("Environment not ready")
            return None, 0.0, True, {}
            
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            self.current_obs = obs
            self.total_reward += reward
            self.episode_steps += 1
            self.game_over = terminated or truncated
            
            return obs, reward, self.game_over, info
            
        except Exception as e:
            logger.error(f"Failed to step environment: {e}")
            return None, 0.0, True, {}
    
    def get_action_space(self) -> Optional[gym.Space]:
        """Get the action space of the environment"""
        if self.env is None:
            return None
        return self.env.action_space
    
    def get_observation_space(self) -> Optional[gym.Space]:
        """Get the observation space of the environment"""
        if self.env is None:
            return None
        return self.env.observation_space
    
    def render(self) -> Optional[np.ndarray]:
        """Get the current rendered frame"""
        if self.env is None:
            return None
            
        try:
            return self.env.render()
        except Exception as e:
            logger.error(f"Failed to render environment: {e}")
            return None
    
    def get_game_stats(self) -> dict:
        """Get current game statistics"""
        return {
            "total_reward": self.total_reward,
            "episode_steps": self.episode_steps,
            "game_over": self.game_over,
            "action_space_size": self.env.action_space.n if self.env else 0
        }
    
    def close(self):
        """Clean up the environment"""
        if self.env is not None:
            self.env.close()
            logger.info("Environment closed")