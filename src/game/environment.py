"""
Atari game environment setup and management - Generic for all games
"""
import logging
import traceback
import ale_py  # Import this first to register ALE environments
import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

class AtariEnvironment:
    """Manages any Atari game environment"""
    
    def __init__(self, environment_name: str = "SpaceInvaders-v0", render_mode: str = "rgb_array"):
        self.environment_name = environment_name
        self.render_mode = render_mode
        self.env: Optional[gym.Env] = None
        self.current_obs: Optional[np.ndarray] = None
        self.total_reward = 0
        self.episode_steps = 0
        self.game_over = False
        
    def create_environment(self) -> bool:
        """Create and configure the Atari environment"""
        try:
            logger.info(f"Creating environment: {self.environment_name}")
            
            # Create base environment WITHOUT frame skipping to avoid conflicts
            self.env = gym.make(
                self.environment_name,
                render_mode=self.render_mode,
                frameskip=1,  # Disable frame skipping in base env
                max_episode_steps=None  # Remove step limit
            )
            logger.info(f"Base environment created: {self.environment_name}")
            
            # Try WITHOUT AtariPreprocessing to see raw game like Space Invaders fix
            # Comment out preprocessing to see if shots/objects become visible
            """
            # Use basic preprocessing for better compatibility across games
            logger.info("Applying AtariPreprocessing...")
            self.env = gym.wrappers.AtariPreprocessing(
                self.env,
                noop_max=30,
                frame_skip=4,  # Now we can safely apply frame skipping
                screen_size=84,
                terminal_on_life_loss=False,
                grayscale_obs=True,
                grayscale_newaxis=False,
                scale_obs=True
            )
            logger.info("AtariPreprocessing applied")
            
            # Stack frames for temporal information
            logger.info("Applying FrameStackObservation...")
            self.env = gym.wrappers.FrameStackObservation(
                self.env, 
                4
            )
            logger.info("FrameStackObservation applied")
            """
            
            logger.info(f"Environment created successfully (RAW - no preprocessing): {self.environment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create environment {self.environment_name}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
            logger.debug(f"Environment reset successfully: {self.environment_name}")
            return self.current_obs
            
        except Exception as e:
            logger.error(f"Failed to reset environment {self.environment_name}: {e}")
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
            logger.error(f"Failed to step environment {self.environment_name}: {e}")
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
            logger.error(f"Failed to render environment {self.environment_name}: {e}")
            return None
    
    def get_game_stats(self) -> dict:
        """Get current game statistics"""
        return {
            "total_reward": self.total_reward,
            "episode_steps": self.episode_steps,
            "game_over": self.game_over,
            "environment_name": self.environment_name,
            "action_space_size": self.env.action_space.n if self.env else 0
        }
    
    def close(self):
        """Clean up the environment"""
        if self.env is not None:
            self.env.close()
            logger.debug(f"Environment closed: {self.environment_name}")


# Legacy compatibility class
class SpaceInvadersEnvironment(AtariEnvironment):
    """Legacy compatibility wrapper for Space Invaders"""
    
    def __init__(self):
        super().__init__("SpaceInvaders-v0", "rgb_array")