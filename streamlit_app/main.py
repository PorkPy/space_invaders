"""
Main Streamlit application for Atari Multi-Game RL Demo
"""
import streamlit as st
import time
import sys
import os
import numpy as np
import cv2
import tempfile
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.multi_game_manager import MultiGameManager
from src.data.games_config import GAMES_CONFIG, GRID_CONFIG

# Page configuration
st.set_page_config(
    page_title="Atari RL Multi-Game Demo",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize Streamlit session state"""
    if 'multi_game_manager' not in st.session_state:
        st.session_state.multi_game_manager = None
    if 'grid_initialized' not in st.session_state:
        st.session_state.grid_initialized = False
    if 'selected_game_detail' not in st.session_state:
        st.session_state.selected_game_detail = None
    if 'auto_play_active' not in st.session_state:
        st.session_state.auto_play_active = False

def show_grid_view():
    """Display the multi-game grid"""
    manager = st.session_state.multi_game_manager
    
    # Main title and description
    st.title("🎮 Atari Multi-Game AI Demo")
    st.markdown("**Watch AI agents play multiple Atari games simultaneously!**")
    
    # Control panel
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        if st.button("▶️ Start Auto-Play", use_container_width=True):
            st.session_state.auto_play_active = True
            st.rerun()
    
    with col2:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.auto_play_active = False
            st.rerun()
    
    with col3:
        if not st.session_state.auto_play_active:
            if st.button("➡️ Step All", use_container_width=True):
                active_games = manager.step_all_games(deterministic=True)
                st.rerun()
    
    with col4:
        # Status indicator
        if st.session_state.auto_play_active:
            st.success("🔄 Auto-play active - games are running...")
        else:
            st.info("⏸️ Paused - click 'Step All' or 'Start Auto-Play'")
    
    # Auto-play logic
    if st.session_state.auto_play_active:
        active_games = manager.step_all_games(deterministic=True)
        
        if active_games > 0:
            time.sleep(GRID_CONFIG["update_interval"])
            st.rerun()
        else:
            st.session_state.auto_play_active = False
            st.warning("All games finished - restarting...")
            st.rerun()
    
    st.divider()
    
    # Display games in grid
    frames = manager.get_all_frames()
    
    if not frames:
        st.warning("No game frames available")
        return
    
    # Create responsive grid layout
    num_games = len(frames)
    if num_games <= 2:
        cols = st.columns(num_games)
    elif num_games <= 4:
        cols = st.columns(2)
    elif num_games <= 6:
        cols = st.columns(3)
    else:
        cols = st.columns(4)
    
    # Display each game
    for i, (game_id, frame) in enumerate(frames.items()):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            game_config = GAMES_CONFIG[game_id]
            
            # Game header with color coding
            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, {game_config['color']}22, transparent);
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 10px;
                border-left: 4px solid {game_config['color']};
            ">
                <h4 style="margin: 0; color: {game_config['color']};">
                    {game_config['emoji']} {game_config['display_name']}
                </h4>
                <small style="color: #666;">{game_config['description']}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Display frame
            if frame is not None:
                try:
                    pil_image = Image.fromarray(frame)
                    st.image(pil_image, use_container_width=True)
                    
                    # Click to expand button
                    if st.button(f"🔍 View Full Screen", key=f"expand_{game_id}", use_container_width=True):
                        st.session_state.selected_game_detail = game_id
                        st.session_state.auto_play_active = False
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Display error: {str(e)[:50]}...")
            else:
                st.warning("⚠️ No frame available")
            
            # Game stats
            stats = manager.get_game_stats(game_id)
            if stats:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Score", f"{stats.get('total_reward', 0):.0f}")
                with col_b:
                    st.metric("Steps", stats.get('episode_steps', 0))

def show_detailed_game_view():
    """Show detailed view of a single game"""
    game_id = st.session_state.selected_game_detail
    manager = st.session_state.multi_game_manager
    game_config = GAMES_CONFIG[game_id]
    
    # Header with back button
    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button("← Back to Grid", use_container_width=True):
            st.session_state.selected_game_detail = None
            st.rerun()
    
    with col2:
        st.title(f"{game_config['emoji']} {game_config['display_name']}")
        st.markdown(f"**{game_config['description']}** - Full Screen View")
    
    # Large game display
    frame = manager.get_game_frame(game_id)
    if frame is not None:
        try:
            # Create much larger version for detailed view
            large_frame = cv2.resize(frame, (800, 600))  # Much bigger!
            pil_image = Image.fromarray(large_frame)
            
            # Center the image
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.image(pil_image, caption=f"AI Playing {game_config['display_name']}")
                
        except Exception as e:
            st.error(f"Frame display error: {e}")
    else:
        st.warning("No frame available for this game")
    
    # Game controls
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("➡️ Single Step", use_container_width=True):
            manager.step_all_games(deterministic=True)
            st.rerun()
    
    with col2:
        auto_detailed = st.checkbox("🔄 Auto-play This Game", value=False)
    
    with col3:
        if st.button("🎬 Record 30-sec GIF", use_container_width=True):
            record_single_game_gif(game_id, manager)
    
    with col4:
        if st.button("🔄 Restart This Game", use_container_width=True):
            # Restart just this game
            game_instance = manager.game_instances.get(game_id)
            if game_instance:
                obs = game_instance.start_new_game()
                if obs is not None:
                    manager.game_observations[game_id] = obs
                    manager.step_counts[game_id] = 0
                    manager.game_running[game_id] = True
                    st.success("Game restarted!")
                    st.rerun()
    
    # Auto-play for detailed view
    if auto_detailed:
        manager.step_all_games(deterministic=True)
        time.sleep(0.15)  # Slightly faster for single game
        st.rerun()
    
    # Detailed stats
    st.divider()
    stats = manager.get_game_stats(game_id)
    if stats:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Current Score", f"{stats.get('total_reward', 0):.0f}")
        with col2:
            st.metric("Steps Played", stats.get('episode_steps', 0))
        with col3:
            st.metric("Environment", stats.get('environment_name', 'Unknown').split('-')[0])
        with col4:
            game_status = "🟢 Playing" if manager.game_running.get(game_id, False) else "🔴 Stopped"
            st.metric("Status", game_status)
        with col5:
            st.metric("AI Agent", stats.get('algorithm', 'Random'))

def record_single_game_gif(game_id: str, manager: MultiGameManager):
    """Record a GIF of a single game"""
    game_config = GAMES_CONFIG[game_id]
    
    with st.spinner(f"🎬 Recording {game_config['display_name']} gameplay..."):
        frames = []
        
        try:
            # Record 60 steps for a longer, more interesting GIF
            for step in range(60):
                manager.step_all_games(deterministic=True)
                
                frame = manager.get_game_frame(game_id)
                if frame is not None:
                    # Resize for GIF (good balance of size and quality)
                    gif_frame = cv2.resize(frame, (480, 360))
                    frames.append(gif_frame)
            
            # Create GIF
            if len(frames) >= 20:
                pil_frames = [Image.fromarray(frame) for frame in frames]
                
                with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
                    gif_path = f.name
                
                pil_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=100,  # 100ms = 10 FPS for smooth playback
                    loop=0
                )
                
                # Display GIF
                st.success(f"✅ Recorded {len(frames)} frames of {game_config['display_name']}!")
                
                with open(gif_path, 'rb') as gif_file:
                    gif_data = gif_file.read()
                    st.image(gif_data, caption=f"{game_config['display_name']} - 30 Second Gameplay", width=480)
                
                # Download button
                st.download_button(
                    label="💾 Download GIF",
                    data=gif_data,
                    file_name=f"{game_id}_gameplay.gif",
                    mime="image/gif"
                )
                
                # Cleanup
                if os.path.exists(gif_path):
                    os.unlink(gif_path)
                    
            else:
                st.warning("Not enough frames recorded for GIF")
                
        except Exception as e:
            st.error(f"Failed to create GIF: {e}")

def main():
    """Main application function"""
    
    # Initialize session state
    init_session_state()
    
    # Sidebar controls
    with st.sidebar:
        st.title("🎮 Game Controls")
        
        # Game selection
        st.subheader("🎯 Select Games")
        available_games = list(GAMES_CONFIG.keys())
        selected_games = st.multiselect(
            "Choose games to play:",
            available_games,
            default=available_games[:4],  # First 4 games
            help="Select which Atari games to display in the grid"
        )
        
        st.divider()
        
        # Grid controls
        st.subheader("🚀 Grid Controls")
        start_grid = st.button("🟢 Start Multi-Game Grid", use_container_width=True)
        stop_grid = st.button("🔴 Stop All Games", use_container_width=True)
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        deterministic = st.checkbox("🎯 Deterministic Play", value=True, help="Use consistent AI decisions")
        
        # Game info
        if selected_games:
            st.subheader("🎮 Selected Games")
            for game_id in selected_games:
                config = GAMES_CONFIG[game_id]
                st.markdown(f"**{config['emoji']} {config['display_name']}**")
                st.caption(config['description'])
    
    # Initialize grid
    if start_grid and selected_games:
        with st.spinner("🎮 Initializing multi-game grid..."):
            try:
                manager = MultiGameManager(selected_games)
                if manager.initialize_games():
                    if manager.start_all_games():
                        st.session_state.multi_game_manager = manager
                        st.session_state.grid_initialized = True
                        st.session_state.auto_play_active = False
                        st.success(f"✅ Started {len(selected_games)} games!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to start games")
                else:
                    st.error("❌ Failed to initialize games")
            except Exception as e:
                st.error(f"❌ Error initializing grid: {e}")
                st.info("💡 Try running the debug script first: `python debug_test.py`")
    
    # Stop grid
    if stop_grid:
        if st.session_state.multi_game_manager:
            st.session_state.multi_game_manager.cleanup()
        st.session_state.multi_game_manager = None
        st.session_state.grid_initialized = False
        st.session_state.selected_game_detail = None
        st.session_state.auto_play_active = False
        st.success("🛑 All games stopped")
        st.rerun()
    
    # Main content
    if st.session_state.grid_initialized and st.session_state.multi_game_manager:
        if st.session_state.selected_game_detail:
            # Show detailed single game view
            show_detailed_game_view()
        else:
            # Show multi-game grid
            show_grid_view()
    
    elif not selected_games:
        # Welcome screen
        st.title("🎮 Atari Multi-Game AI Demo")
        st.markdown("### Welcome to the Ultimate Atari AI Experience!")
        
        st.info("👈 **Get Started:** Select games from the sidebar and click 'Start Multi-Game Grid'")
        
        # Show available games in a nice layout
        st.subheader("🎯 Available Games")
        
        cols = st.columns(3)
        for i, (game_id, config) in enumerate(GAMES_CONFIG.items()):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {config['color']}22, transparent);
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 10px;
                    border: 2px solid {config['color']}33;
                    text-align: center;
                ">
                    <h3 style="margin: 0; color: {config['color']};">
                        {config['emoji']}
                    </h3>
                    <h4 style="margin: 5px 0; color: {config['color']};">
                        {config['display_name']}
                    </h4>
                    <p style="margin: 0; color: #666; font-size: 14px;">
                        {config['description']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # Waiting to start
        st.title("🎮 Ready to Start!")
        st.info("👈 Click 'Start Multi-Game Grid' in the sidebar to begin the demo")
        
        # Show selected games
        st.subheader(f"Selected Games ({len(selected_games)})")
        cols = st.columns(len(selected_games))
        for i, game_id in enumerate(selected_games):
            config = GAMES_CONFIG[game_id]
            with cols[i]:
                st.markdown(f"**{config['emoji']} {config['display_name']}**")
                st.caption(config['description'])

if __name__ == "__main__":
    main()