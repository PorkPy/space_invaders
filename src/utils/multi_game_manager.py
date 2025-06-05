"""
Multi-game management for grid display
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
                game_inference = GameInference()
                
                # Override the environment name for this game
                from config.settings import GAME_CONFIG
                temp_config = GAME_CONFIG.copy()
                temp_config["environment_name"] = game_config["env_name"]
                
                # Initialize with custom config
                if self._initialize_single_game(game_inference, temp_config):
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
    
    def _initialize_single_game(self, game_inference: GameInference, config: dict) -> bool:
        """Initialize a single game with custom config"""
        try:
            # Temporarily override the config
            from config import settings
            original_config = settings.GAME_CONFIG.copy()
            settings.GAME_CONFIG.update(config)
            
            # Initialize
            success = game_inference.initialize()
            
            # Restore original config
            settings.GAME_CONFIG = original_config
            
            return success
        except Exception as e:
            logger.error(f"Failed to initialize single game: {e}")
            return False
    
    def start_all_games(self) -> Dict[str, bool]:
        """Start all games and return success status"""
        results = {}
        
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
                    results[game_id] = True
                else:
                    results[game_id] = False
                    
            except Exception as e:
                logger.error(f"Failed to start {game_id}: {e}")
                results[game_id] = False
        
        return results
    
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
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            # Resize to grid size
            target_size = GAMES_CONFIG[game_id]["grid_size"]
            processed_frame = cv2.resize(frame, target_size)
            
            return processed_frame
        except Exception as e:
            logger.error(f"Error processing frame for {game_id}: {e}")
            return frame
    
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
            except:
                pass
        
        self.game_instances.clear()
        self.game_frames.clear()
        self.game_stats.clear()
        self.game_observations.clear()
        self.game_running.clear()
        self.step_counts.clear()