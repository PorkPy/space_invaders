"""
Multi-game grid display page - Fixed version
"""
import streamlit as st
import time
import numpy as np
from PIL import Image
import tempfile
import cv2
import os
from src.utils.multi_game_manager import MultiGameManager
from src.data.games_config import GAMES_CONFIG, GRID_CONFIG

def show():
    """Display the multi-game grid page"""
    st.title("🎮 Atari Multi-Game Grid")
    st.markdown("**Watch AI agents play multiple Atari games simultaneously!**")
    
    # Initialize session state
    if 'multi_game_manager' not in st.session_state:
        st.session_state.multi_game_manager = None
    if 'grid_initialized' not in st.session_state:
        st.session_state.grid_initialized = False
    if 'selected_game_detail' not in st.session_state:
        st.session_state.selected_game_detail = None
    if 'auto_play_active' not in st.session_state:
        st.session_state.auto_play_active = False
    
    # Sidebar controls
    st.sidebar.subheader("🎯 Grid Controls")
    
    # Game selection
    available_games = list(GAMES_CONFIG.keys())
    selected_games = st.sidebar.multiselect(
        "Select Games",
        available_games,
        default=available_games[:4],  # First 4 games for better performance
        help="Choose which games to display in the grid"
    )
    
    # Grid controls
    start_grid = st.sidebar.button("🚀 Start Multi-Game Grid", use_container_width=True)
    stop_grid = st.sidebar.button("⏹️ Stop Grid", use_container_width=True)
    
    # Settings
    st.sidebar.markdown("**⚙️ Grid Settings**")
    deterministic = st.sidebar.checkbox("Deterministic Play", value=True)
    
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
                    else:
                        st.error("❌ Failed to start games")
                else:
                    st.error("❌ Failed to initialize games")
            except Exception as e:
                st.error(f"❌ Error initializing grid: {e}")
    
    # Stop grid
    if stop_grid:
        if st.session_state.multi_game_manager:
            st.session_state.multi_game_manager.cleanup()
        st.session_state.multi_game_manager = None
        st.session_state.grid_initialized = False
        st.session_state.selected_game_detail = None
        st.session_state.auto_play_active = False
        st.info("Grid stopped")
        st.rerun()
    
    # Display grid or detailed view
    if st.session_state.grid_initialized and st.session_state.multi_game_manager:
        
        if st.session_state.selected_game_detail:
            # Detailed single game view
            show_detailed_game_view()
        else:
            # Multi-game grid view
            show_grid_view()
    
    elif not selected_games:
        st.info("👆 Select games from the sidebar to start the multi-game grid")
    else:
        st.info("👆 Click 'Start Multi-Game Grid' to begin the demo")

def show_grid_view():
    """Display the multi-game grid"""
    manager = st.session_state.multi_game_manager
    
    # Control panel
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("▶️ Start Auto-Play"):
            st.session_state.auto_play_active = True
            st.rerun()
    
    with col2:
        if st.button("⏸️ Stop Auto-Play"):
            st.session_state.auto_play_active = False
            st.rerun()
    
    with col3:
        if not st.session_state.auto_play_active:
            if st.button("➡️ Step All Games"):
                active_games = manager.step_all_games(deterministic=True)
                st.rerun()
    
    # Auto-play logic
    if st.session_state.auto_play_active:
        active_games = manager.step_all_games(deterministic=True)
        
        if active_games > 0:
            time.sleep(GRID_CONFIG["update_interval"])
            st.rerun()
        else:
            st.session_state.auto_play_active = False
            st.warning("All games finished")
    
    # Status indicator
    if st.session_state.auto_play_active:
        st.success("🔄 Auto-play active - games are running...")
    else:
        st.info("⏸️ Auto-play stopped - click 'Step All Games' or 'Start Auto-Play'")
    
    # Display games in grid
    frames = manager.get_all_frames()
    
    # Create responsive grid layout
    num_games = len(frames)
    if num_games <= 2:
        cols = st.columns(num_games)
    elif num_games <= 4:
        cols = st.columns(2)
    else:
        cols = st.columns(3)
    
    for i, (game_id, frame) in enumerate(frames.items()):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            game_config = GAMES_CONFIG[game_id]
            
            # Game title
            st.markdown(f"**{game_config['emoji']} {game_config['display_name']}**")
            
            # Display frame
            if frame is not None:
                # Convert to PIL for display
                try:
                    pil_image = Image.fromarray(frame)
                    st.image(pil_image, use_container_width=True)
                    
                    # Click to expand button
                    if st.button(f"🔍 View {game_config['display_name']}", key=f"expand_{game_id}"):
                        st.session_state.selected_game_detail = game_id
                        st.session_state.auto_play_active = False
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Display error: {e}")
            else:
                st.warning("No frame available")
            
            # Game stats
            stats = manager.get_game_stats(game_id)
            if stats:
                st.caption(f"Score: {stats.get('total_reward', 0)} | Steps: {stats.get('episode_steps', 0)}")

def show_detailed_game_view():
    """Show detailed view of a single game"""
    game_id = st.session_state.selected_game_detail
    manager = st.session_state.multi_game_manager
    game_config = GAMES_CONFIG[game_id]
    
    # Header
    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button("← Back to Grid"):
            st.session_state.selected_game_detail = None
            st.rerun()
    
    with col2:
        st.subheader(f"{game_config['emoji']} {game_config['display_name']} - Detailed View")
    
    # Display large game frame
    frame = manager.get_game_frame(game_id)
    if frame is not None:
        try:
            # Create larger version for detailed view
            large_frame = cv2.resize(frame, (640, 400))
            pil_image = Image.fromarray(large_frame)
            st.image(pil_image, caption=f"AI Playing {game_config['display_name']}", use_container_width=False)
        except Exception as e:
            st.error(f"Frame display error: {e}")
    
    # Game controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➡️ Single Step"):
            manager.step_all_games(deterministic=True)
            st.rerun()
    
    with col2:
        auto_detailed = st.checkbox("🔄 Auto-play", value=False)
    
    with col3:
        if st.button("🎬 Record GIF"):
            # Record a short GIF of this specific game
            record_single_game_gif(game_id, manager)
    
    # Auto-play for detailed view
    if auto_detailed:
        manager.step_all_games(deterministic=True)
        time.sleep(0.2)
        st.rerun()
    
    # Detailed stats
    stats = manager.get_game_stats(game_id)
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Score", stats.get('total_reward', 0))
        with col2:
            st.metric("Steps", stats.get('episode_steps', 0))
        with col3:
            st.metric("Environment", stats.get('environment_name', 'Unknown'))
        with col4:
            game_status = "Playing" if manager.game_running.get(game_id, False) else "Stopped"
            st.metric("Status", game_status)

def record_single_game_gif(game_id: str, manager: MultiGameManager):
    """Record a GIF of a single game"""
    game_config = GAMES_CONFIG[game_id]
    
    with st.spinner(f"🎬 Recording {game_config['display_name']} gameplay..."):
        frames = []
        
        try:
            # Record 30 steps for shorter, faster GIF
            for step in range(30):
                # Take step
                manager.step_all_games(deterministic=True)
                
                # Get frame
                frame = manager.get_game_frame(game_id)
                if frame is not None:
                    # Resize for GIF
                    gif_frame = cv2.resize(frame, (400, 300))
                    frames.append(gif_frame)
            
            # Create GIF
            if len(frames) >= 10:
                pil_frames = [Image.fromarray(frame) for frame in frames]
                
                with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
                    gif_path = f.name
                
                pil_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=150,  # Slower for better visibility
                    loop=0
                )
                
                # Display GIF
                st.success(f"✅ Recorded {game_config['display_name']} gameplay!")
                
                with open(gif_path, 'rb') as gif_file:
                    gif_data = gif_file.read()
                    st.image(gif_data, caption=f"{game_config['display_name']} Gameplay")
                
                # Cleanup
                if os.path.exists(gif_path):
                    os.unlink(gif_path)
                    
            else:
                st.warning("Not enough frames recorded for GIF")
                
        except Exception as e:
            st.error(f"Failed to create GIF: {e}")