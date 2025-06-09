"""
Multi-game management for grid display - Fixed version
"""
import logging
import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List
from src.models.inference import GameInference
from src.data.games_config import GAMES_CONFIG, GRID_CONFIG

logger = logging.getLogger(__name__)

class MultiGameManager:
    """Manages multiple Atari games simultaneously"""
    
    def __init__(self, selected_games: List[str] = None):
        self.selected_games = selected_games or list(GAMES_CONFIG.keys())[:6]  # First 6 games
        self.game_instances: Dict[str, GameInference] = {}
        self.game_frames: Dict[str, np.ndarray] = {}
        self.game_stats: Dict[str, dict] = {}
        self.game_observations: Dict[str, np.ndarray] = {}
        self.game_running: Dict[str, bool] = {}
        self.step_counts: Dict[str, int] = {}
        
    def initialize_games(self) -> bool:
        """Initialize all selected games"""
        success_count = 0
        
        for game_id in self.selected_games:
            if game_id not in GAMES_CONFIG:
                logger.warning(f"Unknown game: {game_id}")
                continue
                
            try:
                # Create game instance with specific environment
                game_config = GAMES_CONFIG[game_id]
                game_inference = GameInference(environment_name=game_config["env_name"])
                
                # Initialize the game
                if game_inference.initialize():
                    self.game_instances[game_id] = game_inference
                    self.game_running[game_id] = False
                    self.step_counts[game_id] = 0
                    success_count += 1
                    logger.info(f"Initialized {game_config['display_name']}")
                else:
                    logger.error(f"Failed to initialize {game_config['display_name']}")
                    
            except Exception as e:
                logger.error(f"Error initializing {game_id}: {e}")
        
        logger.info(f"Successfully initialized {success_count}/{len(self.selected_games)} games")
        return success_count > 0
    
    def start_all_games(self) -> bool:
        """Start all games and return success status"""
        success_count = 0
        
        for game_id, game_instance in self.game_instances.items():
            try:
                observation = game_instance.start_new_game()
                if observation is not None:
                    self.game_observations[game_id] = observation
                    self.game_running[game_id] = True
                    self.step_counts[game_id] = 0
                    
                    # Get initial frame
                    frame = game_instance.get_render_frame()
                    if frame is not None:
                        self.game_frames[game_id] = self._process_frame(frame, game_id)
                    
                    # Get initial stats
                    self.game_stats[game_id] = game_instance.get_game_stats()
                    success_count += 1
                    logger.info(f"Started {GAMES_CONFIG[game_id]['display_name']}")
                else:
                    logger.error(f"Failed to start {game_id}")
                    
            except Exception as e:
                logger.error(f"Failed to start {game_id}: {e}")
        
        logger.info(f"Successfully started {success_count}/{len(self.game_instances)} games")
        return success_count > 0
    
    def step_all_games(self, deterministic: bool = True) -> int:
        """Take one step in all running games, return number of active games"""
        active_games = 0
        
        for game_id, game_instance in self.game_instances.items():
            if not self.game_running.get(game_id, False):
                continue
                
            try:
                current_obs = self.game_observations.get(game_id)
                if current_obs is None:
                    continue
                
                # Take step
                observation, reward, done, info, action = game_instance.play_step(
                    current_obs, deterministic=deterministic
                )
                
                self.step_counts[game_id] += 1
                
                # Update frame
                frame = game_instance.get_render_frame()
                if frame is not None:
                    self.game_frames[game_id] = self._process_frame(frame, game_id)
                
                # Update stats
                self.game_stats[game_id] = game_instance.get_game_stats()
                
                if done or self.step_counts[game_id] >= GRID_CONFIG["max_steps_per_game"]:
                    if GRID_CONFIG["auto_restart"]:
                        # Restart the game
                        new_obs = game_instance.start_new_game()
                        if new_obs is not None:
                            self.game_observations[game_id] = new_obs
                            self.step_counts[game_id] = 0
                            active_games += 1
                        else:
                            self.game_running[game_id] = False
                    else:
                        self.game_running[game_id] = False
                else:
                    self.game_observations[game_id] = observation
                    active_games += 1
                    
            except Exception as e:
                logger.error(f"Error stepping {game_id}: {e}")
                self.game_running[game_id] = False
        
        return active_games
    
    def _process_frame(self, frame: np.ndarray, game_id: str) -> np.ndarray:
        """Process frame for grid display"""
        try:
            if frame is None:
                return None
                
            # Handle different frame formats
            if len(frame.shape) == 2:
                # Grayscale to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 3 and frame.shape[2] == 1:
                # Single channel to RGB
                frame = np.repeat(frame, 3, axis=2)
            
            # Ensure uint8 format
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Resize to grid size
            target_size = GAMES_CONFIG[game_id]["grid_size"]
            processed_frame = cv2.resize(frame, target_size)
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error processing frame for {game_id}: {e}")
            # Return a black frame as fallback
            target_size = GAMES_CONFIG[game_id]["grid_size"]
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    def get_game_frame(self, game_id: str) -> Optional[np.ndarray]:
        """Get current frame for a specific game"""
        return self.game_frames.get(game_id)
    
    def get_game_stats(self, game_id: str) -> dict:
        """Get current stats for a specific game"""
        return self.game_stats.get(game_id, {})
    
    def get_all_frames(self) -> Dict[str, np.ndarray]:
        """Get all current frames"""
        return self.game_frames.copy()
    
    def cleanup(self):
        """Clean up all game instances"""
        for game_instance in self.game_instances.values():
            try:
                game_instance.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        
        self.game_instances.clear()
        self.game_frames.clear()
        self.game_stats.clear()
        self.game_observations.clear()
        self.game_running.clear()
        self.step_counts.clear()
        
        logger.info("Multi-game manager cleaned up")