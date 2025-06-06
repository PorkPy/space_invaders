"""
Main Streamlit application for Space Invaders RL Demo - Now with Multi-Game Grid
"""
import streamlit as st
import time
import sys
import os
import numpy as np  # Added missing import
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import STREAMLIT_CONFIG
from src.models.inference import GameInference
from streamlit_app.components.game_display import GameDisplay

# Page configuration
st.set_page_config(
    page_title="Atari RL Demo Suite",
    page_icon="🎮",
    layout=STREAMLIT_CONFIG["layout"],
    initial_sidebar_state=STREAMLIT_CONFIG["initial_sidebar_state"]
)

def init_session_state():
    """Initialize Streamlit session state"""
    if 'game_inference' not in st.session_state:
        st.session_state.game_inference = None
    if 'game_display' not in st.session_state:
        st.session_state.game_display = GameDisplay()
    if 'current_observation' not in st.session_state:
        st.session_state.current_observation = None
    if 'game_running' not in st.session_state:
        st.session_state.game_running = False
    if 'initialization_done' not in st.session_state:
        st.session_state.initialization_done = False
    if 'auto_play_active' not in st.session_state:
        st.session_state.auto_play_active = False
    if 'step_counter' not in st.session_state:
        st.session_state.step_counter = 0

def initialize_system():
    """Initialize the RL system"""
    if st.session_state.initialization_done:
        return True
        
    with st.spinner("🤖 Loading AI model and initializing game environment..."):
        # Default to Space Invaders for single game demo
        game_inference = GameInference("SpaceInvaders-v0")
        
        if game_inference.initialize():
            st.session_state.game_inference = game_inference
            st.session_state.initialization_done = True
            st.success("✅ System initialized successfully!")
            return True
        else:
            st.error("❌ Failed to initialize system. Please check the logs.")
            return False

def show_single_game_demo():
    """Show the single Space Invaders demo"""
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("👾 Space Invaders AI Demo")
    st.markdown("**Watch an AI agent play Space Invaders using Deep Q-Learning!**")
    
    # Initialize system
    if not initialize_system():
        st.stop()
    
    # Setup display layout
    st.session_state.game_display.setup_display()
    
    # Game controls
    controls = st.session_state.game_display.show_game_controls()
    
    # Handle demo controls
    game_inference = st.session_state.game_inference
    
    # Auto-start demo when page loads
    if st.session_state.current_observation is None and not controls["start_demo"]:
        st.info("👆 Click 'Start AI Demo' to watch the AI play Space Invaders!")
    
    # Auto-start demo when page loads OR when button is clicked
    if controls["start_demo"] or st.session_state.current_observation is None:
        
        # Determine demo length
        if controls["demo_type"] == "Quick Demo (50 steps)":
            max_steps = 50
        elif controls["demo_type"] == "Full Game (until death)":
            max_steps = 1000  # Max safety limit
        elif controls["demo_type"] == "Extended Demo (200 steps)":
            max_steps = 200
        elif controls["demo_type"] == "Custom Length":
            max_steps = controls["custom_length"]
        else:
            max_steps = 100
        
        with st.spinner(f"🎬 Recording AI gameplay ({controls['demo_type']})..."):
            # Start new game
            observation = game_inference.start_new_game()
            if observation is None:
                st.error("Failed to start game")
                st.stop()
                
            st.session_state.current_observation = observation
            st.session_state.game_running = True
            st.session_state.step_counter = 0
            
            # Record frames until game ends or max steps reached
            try:
                import cv2
                import tempfile
                from PIL import Image
                
                frames = []
                current_obs = observation
                
                for step in range(max_steps):
                    if st.session_state.game_running:
                        # Take game step
                        observation, reward, done, info, action = game_inference.play_step(
                            current_obs,
                            deterministic=controls["deterministic"]
                        )
                        
                        st.session_state.step_counter += 1
                        
                        # Get frame after action
                        frame_after = game_inference.get_render_frame()
                        frame = frame_after
                        
                        if frame is not None:
                            if len(frame.shape) == 2:
                                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                            # Scale to good TV-like proportions
                            resized_frame = cv2.resize(frame, (320, 200))
                            frames.append(resized_frame)
                        
                        if done:
                            st.session_state.game_running = False
                            break
                        else:
                            current_obs = observation
                            st.session_state.current_observation = observation
                
                # Create animated GIF
                if len(frames) >= 2:
                    # Convert frames to PIL Images
                    pil_frames = []
                    for frame in frames:
                        if frame.dtype != np.uint8:
                            frame = (frame * 255).astype(np.uint8)
                        pil_img = Image.fromarray(frame)
                        pil_frames.append(pil_img)
                    
                    # Create temporary GIF file
                    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
                        gif_path = f.name
                    
                    # Save as animated GIF with better settings for fast objects
                    pil_frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=pil_frames[1:],
                        duration=50,  # Faster: 50ms = 20 FPS instead of 100ms
                        loop=0
                    )
                    
                    # Display the GIF
                    if os.path.exists(gif_path) and os.path.getsize(gif_path) > 0:
                        st.success(f"✅ AI Demo Complete! Played {len(frames)} steps.")
                        
                        with open(gif_path, 'rb') as gif_file:
                            gif_data = gif_file.read()
                            st.image(gif_data, caption="AI Playing Space Invaders", width=600)  # Much bigger!
                        
                        # Show final stats
                        final_stats = game_inference.get_game_stats()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Final Score", final_stats.get('total_reward', 0))
                        with col2:
                            st.metric("Steps Played", st.session_state.step_counter)
                        with col3:
                            game_status = "Game Over" if not st.session_state.game_running else "Stopped"
                            st.metric("Status", game_status)
                        
                        # Cleanup
                        if os.path.exists(gif_path):
                            os.unlink(gif_path)
                    else:
                        st.error("Failed to create demo animation")
                        
                else:
                    st.warning("Not enough frames captured for demo")
                    
            except Exception as e:
                st.error(f"Demo failed: {e}")

def main():
    """Main application function with navigation"""
    
    # Sidebar navigation
    st.sidebar.title("🎮 Atari RL Demo Suite")
    
    page = st.sidebar.selectbox(
        "Choose Demo Type",
        [
            "👾 Single Game (Space Invaders)",
            "🎯 Multi-Game Grid"
        ]
    )
    
    if page == "👾 Single Game (Space Invaders)":
        show_single_game_demo()
        
        # Information section
        with st.expander("ℹ️ About This AI Demo"):
            st.markdown("""
            This demo showcases a **Deep Q-Network (DQN)** agent playing Space Invaders.
            
            **How it works:**
            - The AI observes the game screen (preprocessed to 84x84 grayscale)
            - It uses a neural network to estimate the value of each possible action
            - It chooses the action with the highest expected reward
            - The model was trained using reinforcement learning on millions of game frames
            
            **Current Model:**
            - **Algorithm**: Deep Q-Network (DQN) 
            - **Framework**: Stable-Baselines3
            - **Environment**: Atari Space Invaders
            - **Actions**: 6 possible actions (NOOP, FIRE, LEFT, RIGHT, etc.)
            
            **Note**: This demo uses a random agent as the pre-trained model had compatibility issues.
            The random agent demonstrates the game mechanics while a proper trained model is being prepared.
            """)
    
    elif page == "🎯 Multi-Game Grid":
        # Import and show multi-game grid
        try:
            from streamlit_app.pages.multi_game_grid import show
            show()
        except ImportError as e:
            st.error(f"Failed to load multi-game grid: {e}")
            st.info("Make sure all dependencies are installed and the multi_game_grid.py file exists.")

if __name__ == "__main__":
    main()