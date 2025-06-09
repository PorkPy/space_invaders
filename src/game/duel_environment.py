"""
Single environment with dual outputs - agent observation + display frame
"""
import logging
import traceback
import ale_py
import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

class DualOutputEnvironment:
    """Single environment that provides both agent observations and display frames"""
    
    def __init__(self, environment_name: str = "SpaceInvaders-v4"):
        self.environment_name = environment_name
        self.env: Optional[gym.Env] = None
        self.raw_env: Optional[gym.Env] = None  # For display frames
        
        # Current state
        self.agent_obs: Optional[np.ndarray] = None
        self.display_frame: Optional[np.ndarray] = None
        self.total_reward = 0
        self.episode_steps = 0
        self.game_over = False
        
    def create_environment(self) -> bool:
        """Create single environment with preprocessing chain"""
        try:
            logger.info(f"Creating dual-output environment for: {self.environment_name}")
            
            # Create base environment with rendering enabled
            self.raw_env = gym.make(
                self.environment_name,
                render_mode="rgb_array",  # Enable rendering for display frames
                frameskip=1,
                max_episode_steps=None
            )
            
            # Create preprocessed environment for agent
            self.env = gym.make(
                self.environment_name,
                render_mode=None,  # No rendering needed for agent path
                frameskip=1,
                max_episode_steps=None
            )
            
            # Apply standard Atari preprocessing for agent observations
            self.env = gym.wrappers.AtariPreprocessing(
                self.env,
                noop_max=30,
                frame_skip=4,
                screen_size=84,
                terminal_on_life_loss=False,
                grayscale_obs=True,
                grayscale_newaxis=False,
                scale_obs=True
            )
            
            # Frame stacking for temporal information
            self.env = gym.wrappers.FrameStackObservation(self.env, 4)
            
            logger.info(f"Dual-output environment created successfully: {self.environment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dual-output environment: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def reset(self) -> Optional[np.ndarray]:
        """Reset environment and return agent observation"""
        if self.env is None or self.raw_env is None:
            logger.error("Environment not created")
            return None
            
        try:
            # Reset both environments with the same seed for synchronization
            seed = np.random.randint(0, 2**31)
            
            self.agent_obs, _ = self.env.reset(seed=seed)
            self.raw_env.reset(seed=seed)
            
            # Get initial display frame
            self.display_frame = self.raw_env.render()
            
            self.total_reward = 0
            self.episode_steps = 0
            self.game_over = False
            
            logger.debug(f"Environment reset - Agent obs: {self.agent_obs.shape}, Display frame: {self.display_frame.shape}")
            return self.agent_obs
            
        except Exception as e:
            logger.error(f"Failed to reset dual-output environment: {e}")
            return None
    
    def step(self, action: int) -> Tuple[Optional[np.ndarray], float, bool, dict]:
        """Step environment and capture both agent obs and display frame"""
        if self.env is None or self.raw_env is None:
            logger.error("Environment not ready")
            return None, 0.0, True, {}
            
        try:
            # Step the agent environment (preprocessed)
            agent_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Step the raw environment with the SAME action for display
            self.raw_env.step(action)
            
            # Get display frame from raw environment
            self.display_frame = self.raw_env.render()
            
            # Update states
            self.agent_obs = agent_obs
            self.total_reward += reward
            self.episode_steps += 1
            self.game_over = terminated or truncated
            
            return agent_obs, reward, self.game_over, info
            
        except Exception as e:
            logger.error(f"Failed to step dual-output environment: {e}")
            return None, 0.0, True, {}
    
    def get_agent_observation(self) -> Optional[np.ndarray]:
        """Get preprocessed observation for agent decisions"""
        return self.agent_obs
    
    def get_display_frame(self) -> Optional[np.ndarray]:
        """Get raw, full-resolution frame for display"""
        return self.display_frame
    
    def get_action_space(self) -> Optional[gym.Space]:
        """Get action space"""
        if self.env is None:
            return None
        return self.env.action_space
    
    def get_observation_space(self) -> Optional[gym.Space]:
        """Get agent observation space (preprocessed)"""
        if self.env is None:
            return None
        return self.env.observation_space
    
    def get_game_stats(self) -> dict:
        """Get current game statistics"""
        return {
            "total_reward": self.total_reward,
            "episode_steps": self.episode_steps,
            "game_over": self.game_over,
            "environment_name": self.environment_name,
            "action_space_size": self.env.action_space.n if self.env else 0,
            "agent_obs_shape": self.agent_obs.shape if self.agent_obs is not None else None,
            "display_frame_shape": self.display_frame.shape if self.display_frame is not None else None
        }
    
    def close(self):
        """Clean up environment"""
        if self.env is not None:
            self.env.close()
        if self.raw_env is not None:
            self.raw_env.close()
        logger.debug(f"Dual-output environment closed: {self.environment_name}")