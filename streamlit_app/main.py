"""
Main Streamlit application for Atari Multi-Game RL Demo - Clean Version
"""
import streamlit as st
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.multi_game_manager import MultiGameManager
from src.data.games_config import GAMES_CONFIG

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
    if 'game_gif_files' not in st.session_state:
        st.session_state.game_gif_files = {}
    if 'recording_in_progress' not in st.session_state:
        st.session_state.recording_in_progress = False
    if 'auto_recorded' not in st.session_state:
        st.session_state.auto_recorded = False

def record_all_games_to_files(manager: MultiGameManager, max_steps: int = 1000):
    """Record GIFs for all games and save as files"""
    st.session_state.recording_in_progress = True
    
    try:
        # Create temp directory
        temp_dir = Path("temp_gifs")
        temp_dir.mkdir(exist_ok=True)
        
        # Clear old files
        for file in temp_dir.glob("*.gif"):
            file.unlink()
        
        # Dictionary to store frames for each game
        all_game_frames = {game_id: [] for game_id in manager.game_instances.keys()}
        
        # Record gameplay
        step_count = 0
        while step_count < max_steps:
            active_games = manager.step_all_games(deterministic=True)
            step_count += 1
            
            # Collect frames from all games
            for game_id in manager.game_instances.keys():
                frame = manager.get_game_frame(game_id)
                if frame is not None:
                    # Resize for display
                    gif_frame = cv2.resize(frame, (320, 240))
                    all_game_frames[game_id].append(gif_frame)
            
            if active_games == 0:
                break
        
        # Save GIFs to files
        game_gif_files = {}
        
        for game_id, frames in all_game_frames.items():
            if len(frames) >= 20:
                try:
                    # Convert to PIL
                    pil_frames = [Image.fromarray(frame) for frame in frames]
                    
                    # Save to file
                    gif_file = temp_dir / f"{game_id}.gif"
                    pil_frames[0].save(
                        str(gif_file),
                        save_all=True,
                        append_images=pil_frames[1:],
                        duration=80,
                        loop=0
                    )
                    
                    game_gif_files[game_id] = str(gif_file)
                        
                except Exception as e:
                    st.error(f"Failed to create GIF for {game_id}: {e}")
        
        st.session_state.game_gif_files = game_gif_files
        st.session_state.recording_in_progress = False
        st.session_state.auto_recorded = True
        
        return len(game_gif_files)
        
    except Exception as e:
        st.error(f"Failed to record games: {e}")
        st.session_state.recording_in_progress = False
        return 0

def show_grid_view():
    """Display the main grid"""
    manager = st.session_state.multi_game_manager
    
    # Main title
    st.title("🎮 Atari Multi-Game AI Demo")
    st.markdown("**AI agents playing multiple Atari games simultaneously**")
    
    # Auto-record when first loaded
    if not st.session_state.auto_recorded and not st.session_state.recording_in_progress:
        with st.spinner("🤖 AI agents are now playing games in real-time... Recording their gameplay to create demos..."):
            gifs_created = record_all_games_to_files(manager, max_steps=1000)
            if gifs_created > 0:
                st.success(f"✅ Recorded {gifs_created} AI gameplay demos!")
                st.rerun()
            else:
                st.error("❌ Failed to record games")
    
    # Status
    if st.session_state.recording_in_progress:
        st.info("🤖 RL agents are actively playing and learning... Please wait while their gameplay is being recorded...")
        st.stop()
    
    # Display GIFs
    if st.session_state.game_gif_files:
        show_clean_grid()
    else:
        st.info("Loading game demos...")

def show_clean_grid():
    """Show the clean grid with just GIFs"""
    game_gif_files = st.session_state.game_gif_files
    num_games = len(game_gif_files)
    
    # Responsive grid
    if num_games <= 2:
        cols = st.columns(2)
    elif num_games <= 4:
        cols = st.columns(2)  # 2x2 grid
    elif num_games <= 6:
        cols = st.columns(3)  # 3x2 grid
    else:
        cols = st.columns(4)  # 4x2 grid
    
    # Display each game GIF - NO BUTTONS AT ALL
    for i, (game_id, gif_file) in enumerate(game_gif_files.items()):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            # Just display the GIF file - clean and simple
            try:
                st.image(gif_file, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying {game_id}: {e}")

def main():
    """Main application function"""
    
    # Initialize session state
    init_session_state()
    
    # Sidebar controls
    with st.sidebar:
        st.title("🎮 Controls")
        
        # Game selection
        st.subheader("🎯 Select Games")
        available_games = list(GAMES_CONFIG.keys())
        selected_games = st.multiselect(
            "Choose games:",
            available_games,
            default=available_games[:4],
            help="Select which Atari games to include"
        )
        
        # Actions - record button higher up
        st.subheader("🚀 Actions")
        start_grid = st.button("🟢 Start Demo", use_container_width=True)
        
        if st.session_state.grid_initialized:
            # Record button prominently placed
            if st.button("🎬 Record New Demos", use_container_width=True):
                st.session_state.auto_recorded = False
                st.session_state.game_gif_files = {}
                st.rerun()
            
            stop_grid = st.button("🔴 Stop Demo", use_container_width=True)
        
        st.divider()
        
        # Info
        st.subheader("ℹ️ About")
        st.markdown("""
        **AI Agents**: Random agents with game-specific behaviors
        
        **Demo Length**: Full games until restart
        
        **Layout**: Clean grid with no buttons or clutter
        """)
        
        # Selected games
        if selected_games:
            st.subheader("🎮 Selected")
            for game_id in selected_games:
                config = GAMES_CONFIG[game_id]
                st.markdown(f"{config['emoji']} {config['display_name']}")
    
    # Initialize grid
    if start_grid and selected_games:
        with st.spinner("🎮 Initializing games..."):
            try:
                manager = MultiGameManager(selected_games)
                if manager.initialize_games():
                    if manager.start_all_games():
                        st.session_state.multi_game_manager = manager
                        st.session_state.grid_initialized = True
                        st.session_state.game_gif_files = {}
                        st.session_state.auto_recorded = False
                        st.success(f"✅ Initialized {len(selected_games)} games!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to start games")
                else:
                    st.error("❌ Failed to initialize games")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Stop grid
    if 'stop_grid' in locals() and stop_grid:
        if st.session_state.multi_game_manager:
            st.session_state.multi_game_manager.cleanup()
        st.session_state.multi_game_manager = None
        st.session_state.grid_initialized = False
        st.session_state.game_gif_files = {}
        st.session_state.auto_recorded = False
        st.success("🛑 Demo stopped")
        st.rerun()
    
    # Main content
    if st.session_state.grid_initialized and st.session_state.multi_game_manager:
        show_grid_view()
    
    elif not selected_games:
        # Welcome screen
        st.title("🎮 Atari Multi-Game AI Demo")
        st.markdown("### Watch AI agents master classic Atari games!")
        
        st.info("👈 **Get Started:** Select games from the sidebar and click 'Start Demo'")
        
        # Show available games
        st.subheader("🎯 Available Games")
        cols = st.columns(3)
        for i, (game_id, config) in enumerate(GAMES_CONFIG.items()):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {config['color']}15, transparent);
                    padding: 12px;
                    border-radius: 8px;
                    margin-bottom: 8px;
                    border: 1px solid {config['color']}30;
                    text-align: center;
                ">
                    <div style="font-size: 24px; margin-bottom: 5px;">
                        {config['emoji']}
                    </div>
                    <div style="font-weight: bold; color: {config['color']}; margin-bottom: 3px;">
                        {config['display_name']}
                    </div>
                    <div style="color: #666; font-size: 12px;">
                        {config['description']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # Ready to start
        st.title("🎮 Ready to Start!")
        st.info("👈 Click 'Start Demo' in the sidebar")
        
        # Show selected games preview
        st.subheader(f"Selected Games ({len(selected_games)})")
        cols = st.columns(len(selected_games) if len(selected_games) <= 4 else 4)
        for i, game_id in enumerate(selected_games):
            if i < 4:
                config = GAMES_CONFIG[game_id]
                with cols[i]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px;">
                        <div style="font-size: 32px;">{config['emoji']}</div>
                        <div style="font-weight: bold;">{config['display_name']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        if len(selected_games) > 4:
            st.caption(f"...and {len(selected_games) - 4} more games")

if __name__ == "__main__":
    main()