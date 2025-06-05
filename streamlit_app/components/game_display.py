"""
Game display and visualization components
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

class GameDisplay:
    """Handles game frame rendering and visualization"""
    
    def __init__(self):
        self.frame_placeholder = None
        self.stats_placeholder = None
        
    def setup_display(self):
        """Setup the display layout"""
        # Create two columns: game display and stats
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("🎮 Live Gameplay")
            self.frame_placeholder = st.empty()
            
        with col2:
            st.subheader("📊 Game Stats")
            self.stats_placeholder = st.empty()
    
    def display_frame(self, frame: Optional[np.ndarray]):
        """Display a single game frame"""
        if self.frame_placeholder is None:
            return
            
        if frame is None:
            self.frame_placeholder.warning("No frame available")
            return
            
        try:
            # Create figure with proper sizing
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(frame, cmap='gray' if len(frame.shape) == 2 else None)
            ax.axis('off')
            ax.set_title("Space Invaders - AI Playing", fontsize=14, pad=20)
            
            # Display in Streamlit
            self.frame_placeholder.pyplot(fig, clear_figure=True)
            plt.close(fig)  # Important: close figure to prevent memory leaks
            
        except Exception as e:
            self.frame_placeholder.error(f"Error displaying frame: {e}")
    
    def display_stats(self, stats: dict):
        """Display game statistics"""
        if self.stats_placeholder is None:
            return
            
        with self.stats_placeholder.container():
            # Current game stats
            st.metric("Score", stats.get("total_reward", 0))
            st.metric("Steps", stats.get("episode_steps", 0))
            
            # Game status
            if stats.get("game_over", False):
                st.error("🔴 Game Over")
            else:
                st.success("🟢 Playing")
            
            # Model info
            st.divider()
            st.text("Model Info:")
            st.text(f"Algorithm: {stats.get('algorithm', 'Unknown')}")
            st.text(f"Actions: {stats.get('action_space_size', 0)}")
            
            # Action space info for Space Invaders
            if stats.get('action_space_size') == 6:
                st.text("Actions: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE")
    
    def display_action_info(self, action: int, action_names: Optional[list] = None):
        """Display current action being taken"""
        if action_names is None:
            action_names = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
            
        if 0 <= action < len(action_names):
            action_name = action_names[action]
            st.sidebar.info(f"Current Action: **{action_name}** ({action})")
        else:
            st.sidebar.warning(f"Unknown action: {action}")
    
    def show_game_controls(self):
        """Show game control interface"""
        st.sidebar.subheader("🎮 Game Controls")
        
        # Game control buttons
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            start_game = st.button("🚀 Start New Game", use_container_width=True)
            
        with col2:
            pause_game = st.button("⏸️ Pause", use_container_width=True)
        
        # Game settings
        st.sidebar.subheader("⚙️ Settings")
        
        deterministic = st.sidebar.checkbox("Deterministic Play", value=True, 
                                          help="If checked, AI always chooses best action. If unchecked, adds some randomness.")
        
        auto_play = st.sidebar.checkbox("Auto Play", value=False,
                                      help="Automatically play continuously")
        
        return {
            "start_game": start_game,
            "pause_game": pause_game,
            "deterministic": deterministic,
            "auto_play": auto_play
        }