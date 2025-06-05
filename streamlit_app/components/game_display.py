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
        # Game display and stats side by side
        col1, col2 = st.columns([3, 1])
        
        with col1:
            self.frame_placeholder = st.empty()
            
        with col2:
            st.markdown("**📊 Stats**")
            self.stats_placeholder = st.empty()
    
    def display_frame(self, frame: Optional[np.ndarray]):
        """Display a single game frame"""
        if self.frame_placeholder is None:
            return
            
        if frame is None:
            self.frame_placeholder.warning("No frame available")
            return
            
        try:
            # Use st.image for display
            if len(frame.shape) == 2:
                from PIL import Image
                pil_image = Image.fromarray(frame, mode='L')
            else:
                from PIL import Image
                pil_image = Image.fromarray(frame, mode='RGB')
            
            self.frame_placeholder.image(
                pil_image,
                caption="AI Playing Space Invaders",
                width=300,
                use_container_width=False
            )
            
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
            st.text(f"Algorithm: {stats.get('algorithm', 'Random Agent')}")
            st.text(f"Actions: {stats.get('action_space_size', 0)}")
    
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
        """Show game control interface - SINGLE HEADER ONLY"""
        st.sidebar.subheader("🎮 Space Invaders Demo")
        
        start_demo = st.sidebar.button("🚀 Start AI Demo", use_container_width=True)
        
        # Demo length controls
        st.sidebar.markdown("**🎬 Demo Length**")
        demo_type = st.sidebar.selectbox(
            "Demo Type",
            options=["Quick Demo (50 steps)", "Full Game (until death)", "Extended Demo (200 steps)", "Custom Length"],
            index=1,
            help="Choose how long the AI should play"
        )
        
        if demo_type == "Custom Length":
            custom_length = st.sidebar.slider("Steps", min_value=10, max_value=500, value=100)
        else:
            custom_length = None
        
        # Settings
        st.sidebar.markdown("**⚙️ Settings**")
        deterministic = st.sidebar.checkbox("Deterministic Play", value=True, 
                                          help="If checked, AI always chooses best action")
        
        return {
            "start_demo": start_demo,
            "demo_type": demo_type,
            "custom_length": custom_length,
            "deterministic": deterministic
        }