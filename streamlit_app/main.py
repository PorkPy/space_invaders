"""
Main Streamlit application for Space Invaders RL Demo
"""
import streamlit as st
import time
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import STREAMLIT_CONFIG
from src.models.inference import GameInference
from streamlit_app.components.game_display import GameDisplay

# Page configuration
st.set_page_config(
    page_title=STREAMLIT_CONFIG["page_title"],
    page_icon=STREAMLIT_CONFIG["page_icon"],
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

def initialize_system():
    """Initialize the RL system"""
    if st.session_state.initialization_done:
        return True
        
    with st.spinner("🤖 Loading AI model and initializing game environment..."):
        game_inference = GameInference()
        
        if game_inference.initialize():
            st.session_state.game_inference = game_inference
            st.session_state.initialization_done = True
            st.success("✅ System initialized successfully!")
            return True
        else:
            st.error("❌ Failed to initialize system. Please check the logs.")
            return False

def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🎮 Space Invaders RL Demo")
    st.markdown("**Watch an AI agent play Space Invaders using Deep Q-Learning!**")
    
    # Initialize system
    if not initialize_system():
        st.stop()
    
    # Setup display layout
    st.session_state.game_display.setup_display()
    
    # Game controls
    controls = st.session_state.game_display.show_game_controls()
    
    # Handle game controls
    game_inference = st.session_state.game_inference
    
    # Start new game
    if controls["start_game"] or st.session_state.current_observation is None:
        with st.spinner("🚀 Starting new game..."):
            observation = game_inference.start_new_game()
            if observation is not None:
                st.session_state.current_observation = observation
                st.session_state.game_running = True
                st.sidebar.success("New game started!")
            else:
                st.sidebar.error("Failed to start game")
    
    # Game loop
    if st.session_state.current_observation is not None and st.session_state.game_running:
        
        # Get current frame for display
        frame = game_inference.get_render_frame()
        stats = game_inference.get_game_stats()
        
        # Display game
        st.session_state.game_display.display_frame(frame)
        st.session_state.game_display.display_stats(stats)
        
        # Auto-play or manual step
        if controls["auto_play"] and not stats.get("game_over", False):
            # Automatic gameplay
            observation, reward, done, info, action = game_inference.play_step(
                st.session_state.current_observation,
                deterministic=controls["deterministic"]
            )
            
            # Display current action
            st.session_state.game_display.display_action_info(action)
            
            if done:
                st.session_state.game_running = False
                st.sidebar.warning("Game Over! Click 'Start New Game' to play again.")
            else:
                st.session_state.current_observation = observation
                
            # Small delay and rerun for auto-play
            time.sleep(0.3)
            st.rerun()
            
        elif not controls["auto_play"]:
            # Manual step control
            col1, col2 = st.columns(2)
            with col1:
                if st.button("➡️ Take One Step", use_container_width=True):
                    observation, reward, done, info, action = game_inference.play_step(
                        st.session_state.current_observation,
                        deterministic=controls["deterministic"]
                    )
                    
                    # Display current action
                    st.session_state.game_display.display_action_info(action)
                    
                    if done:
                        st.session_state.game_running = False
                        st.sidebar.warning("Game Over! Click 'Start New Game' to play again.")
                    else:
                        st.session_state.current_observation = observation
                        st.rerun()
            
            with col2:
                if st.button("🎮 Take 5 Steps", use_container_width=True):
                    for i in range(5):
                        if not stats.get("game_over", False):
                            observation, reward, done, info, action = game_inference.play_step(
                                st.session_state.current_observation,
                                deterministic=controls["deterministic"]
                            )
                            
                            if done:
                                st.session_state.game_running = False
                                st.sidebar.warning("Game Over! Click 'Start New Game' to play again.")
                                break
                            else:
                                st.session_state.current_observation = observation
                    st.rerun()
    
    # Information section
    with st.expander("ℹ️ About This Demo"):
        st.markdown("""
        This demo showcases a **Deep Q-Network (DQN)** agent trained to play Space Invaders.
        
        **How it works:**
        - The AI observes the game screen (preprocessed to 84x84 grayscale)
        - It uses a neural network to estimate the value of each possible action
        - It chooses the action with the highest expected reward
        - The model was trained using reinforcement learning on millions of game frames
        
        **Controls:**
        - **Deterministic**: AI always picks the best action
        - **Non-deterministic**: AI adds some exploration/randomness
        - **Auto Play**: Watch the AI play continuously
        - **Manual Step**: Control the pace yourself
        
        **Model Details:**
        - Algorithm: Deep Q-Network (DQN)
        - Framework: Stable-Baselines3
        - Environment: Atari Space Invaders
        - Input: 4 stacked frames (84x84 grayscale)
        - Actions: 6 possible actions (NOOP, FIRE, LEFT, RIGHT, etc.)
        """)

if __name__ == "__main__":
    main()