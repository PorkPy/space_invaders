"""
Simple test to verify multi-game concept with just Space Invaders
"""
import streamlit as st
import time
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.inference import GameInference

st.set_page_config(
    page_title="Multi Space Invaders Test",
    page_icon="👾",
    layout="wide"
)

def init_session_state():
    """Initialize session state"""
    if 'games' not in st.session_state:
        st.session_state.games = {}
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

def create_multiple_space_invaders(num_games=4):
    """Create multiple Space Invaders instances"""
    games = {}
    
    for i in range(num_games):
        game_id = f"space_invaders_{i+1}"
        try:
            # Create GameInference with Space Invaders
            game = GameInference("SpaceInvaders-v0")
            if game.initialize():
                # Start the game
                obs = game.start_new_game()
                if obs is not None:
                    games[game_id] = {
                        'instance': game,
                        'observation': obs,
                        'running': True,
                        'steps': 0
                    }
                    st.success(f"✅ Created {game_id}")
                else:
                    st.error(f"❌ Failed to start {game_id}")
            else:
                st.error(f"❌ Failed to initialize {game_id}")
        except Exception as e:
            st.error(f"❌ Error creating {game_id}: {e}")
    
    return games

def step_all_games(games):
    """Step all games forward"""
    for game_id, game_data in games.items():
        if game_data['running']:
            try:
                game = game_data['instance']
                obs = game_data['observation']
                
                # Take a step
                new_obs, reward, done, info, action = game.play_step(obs, deterministic=True)
                
                if done:
                    # Restart the game
                    new_obs = game.start_new_game()
                    game_data['steps'] = 0
                    if new_obs is None:
                        game_data['running'] = False
                        continue
                
                game_data['observation'] = new_obs
                game_data['steps'] += 1
                
            except Exception as e:
                st.error(f"Error stepping {game_id}: {e}")
                game_data['running'] = False

def get_game_frame(game_data):
    """Get current frame from a game"""
    try:
        frame = game_data['instance'].get_render_frame()
        if frame is not None:
            # Process frame for display
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            # Resize for grid
            frame = cv2.resize(frame, (320, 200))
            return frame
    except Exception as e:
        st.error(f"Frame error: {e}")
    return None

def main():
    init_session_state()
    
    st.title("👾 Multi Space Invaders Test")
    st.markdown("Testing multiple Space Invaders instances running simultaneously")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start 4 Games"):
            with st.spinner("Creating games..."):
                games = create_multiple_space_invaders(4)
                if games:
                    st.session_state.games = games
                    st.session_state.initialized = True
    
    with col2:
        auto_play = st.checkbox("🔄 Auto Play")
    
    with col3:
        if st.button("➡️ Step All") and st.session_state.initialized:
            step_all_games(st.session_state.games)
            st.rerun()
    
    # Display games
    if st.session_state.initialized and st.session_state.games:
        
        # Auto-play logic
        if auto_play:
            step_all_games(st.session_state.games)
            time.sleep(0.2)
            st.rerun()
        
        # Display in 2x2 grid
        col1, col2 = st.columns(2)
        
        games_list = list(st.session_state.games.items())
        
        # Top row
        if len(games_list) >= 1:
            with col1:
                game_id, game_data = games_list[0]
                st.subheader(f"👾 {game_id}")
                frame = get_game_frame(game_data)
                if frame is not None:
                    st.image(frame, use_container_width=True)
                st.caption(f"Steps: {game_data['steps']}")
        
        if len(games_list) >= 2:
            with col2:
                game_id, game_data = games_list[1]
                st.subheader(f"👾 {game_id}")
                frame = get_game_frame(game_data)
                if frame is not None:
                    st.image(frame, use_container_width=True)
                st.caption(f"Steps: {game_data['steps']}")
        
        # Bottom row
        if len(games_list) >= 3:
            col3, col4 = st.columns(2)
            with col3:
                game_id, game_data = games_list[2]
                st.subheader(f"👾 {game_id}")
                frame = get_game_frame(game_data)
                if frame is not None:
                    st.image(frame, use_container_width=True)
                st.caption(f"Steps: {game_data['steps']}")
        
        if len(games_list) >= 4:
            with col4:
                game_id, game_data = games_list[3]
                st.subheader(f"👾 {game_id}")
                frame = get_game_frame(game_data)
                if frame is not None:
                    st.image(frame, use_container_width=True)
                st.caption(f"Steps: {game_data['steps']}")
    
    else:
        st.info("👆 Click 'Start 4 Games' to begin the test")

if __name__ == "__main__":
    main()