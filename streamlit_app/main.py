"""
Main Streamlit application for Atari Multi-Game RL Demo - Pure Auto Version
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
    """Initialize Streamlit session state and clean up old files"""
    if 'multi_game_manager' not in st.session_state:
        st.session_state.multi_game_manager = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'game_gif_files' not in st.session_state:
        st.session_state.game_gif_files = {}
    if 'recording_in_progress' not in st.session_state:
        st.session_state.recording_in_progress = False
    if 'demos_ready' not in st.session_state:
        st.session_state.demos_ready = False
    if 'cleanup_done' not in st.session_state:
        # Clean up old GIF files on startup
        cleanup_old_gifs()
        st.session_state.cleanup_done = True

def cleanup_old_gifs():
    """Delete any old GIF files from previous runs"""
    try:
        temp_dir = Path("temp_gifs")
        if temp_dir.exists():
            # Delete all GIF files
            for gif_file in temp_dir.glob("*.gif"):
                gif_file.unlink()
            # Also clean up any other temp files
            for temp_file in temp_dir.glob("*"):
                if temp_file.is_file():
                    temp_file.unlink()
    except Exception as e:
        # Silent cleanup - don't show errors for missing files
        pass

def auto_initialize_all_games():
    """Auto-initialize all available games"""
    if st.session_state.initialized:
        return True
    
    # Use all available games
    all_games = list(GAMES_CONFIG.keys())
    
    try:
        manager = MultiGameManager(all_games)
        if manager.initialize_games():
            if manager.start_all_games():
                st.session_state.multi_game_manager = manager
                st.session_state.initialized = True
                return True
    except Exception as e:
        st.error(f"Failed to initialize games: {e}")
    
    return False

def record_full_games(manager: MultiGameManager):
    """Record full games until completion"""
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
        
        # Record full games - longer duration for complete games
        max_steps = 2000  # Longer recording for full games
        step_count = 0
        
        while step_count < max_steps:
            active_games = manager.step_all_games(deterministic=True)
            step_count += 1
            
            # Collect frames from all games
            for game_id in manager.game_instances.keys():
                frame = manager.get_game_frame(game_id)
                if frame is not None:
                    # Good size for display
                    gif_frame = cv2.resize(frame, (400, 300))
                    all_game_frames[game_id].append(gif_frame)
            
            # Continue until we have substantial gameplay
            if step_count > 500 and active_games == 0:  # Minimum 500 steps
                break
        
        # Save GIFs to files
        game_gif_files = {}
        
        for game_id, frames in all_game_frames.items():
            if len(frames) >= 50:  # Need substantial footage
                try:
                    # Convert to PIL
                    pil_frames = [Image.fromarray(frame) for frame in frames]
                    
                    # Save to file with good quality
                    gif_file = temp_dir / f"{game_id}.gif"
                    pil_frames[0].save(
                        str(gif_file),
                        save_all=True,
                        append_images=pil_frames[1:],
                        duration=60,  # Smooth 16.7 FPS
                        loop=0
                    )
                    
                    game_gif_files[game_id] = str(gif_file)
                        
                except Exception as e:
                    st.error(f"Failed to create GIF for {game_id}: {e}")
        
        st.session_state.game_gif_files = game_gif_files
        st.session_state.recording_in_progress = False
        st.session_state.demos_ready = True
        
        return len(game_gif_files)
        
    except Exception as e:
        st.error(f"Failed to record games: {e}")
        st.session_state.recording_in_progress = False
        return 0

def show_pure_grid():
    """Show just the games - pure and clean"""
    game_gif_files = st.session_state.game_gif_files
    num_games = len(game_gif_files)
    
    # Responsive grid that fits screen nicely
    if num_games <= 2:
        cols = st.columns(2)
    elif num_games <= 4:
        cols = st.columns(2)  # 2x2 grid
    elif num_games <= 6:
        cols = st.columns(3)  # 3x2 grid
    else:
        cols = st.columns(4)  # 4x2 grid for all 6+ games
    
    # Display each game GIF - nothing else
    for i, (game_id, gif_file) in enumerate(game_gif_files.items()):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            # Pure game display - no labels, no buttons, no clutter
            try:
                st.image(gif_file, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying {game_id}: {e}")

def main():
    """Main application - pure auto experience"""
    
    # Initialize session state
    init_session_state()
    
    # Auto-initialize games on first load
    if not st.session_state.initialized:
        with st.spinner("🎮 Initializing AI agents for all Atari games..."):
            if auto_initialize_all_games():
                st.success("✅ All games initialized!")
                st.rerun()
            else:
                st.error("❌ Failed to initialize games")
                st.stop()
    
    # Auto-record full games
    if st.session_state.initialized and not st.session_state.demos_ready and not st.session_state.recording_in_progress:
        with st.spinner("🤖 AI agents are playing full games... Recording complete gameplay sessions..."):
            manager = st.session_state.multi_game_manager
            gifs_created = record_full_games(manager)
            if gifs_created > 0:
                st.success(f"✅ Recorded {gifs_created} complete AI gameplay sessions!")
                st.rerun()
            else:
                st.error("❌ Failed to record games")
    
    # Show recording status
    if st.session_state.recording_in_progress:
        st.info("🎬 AI agents are playing complete games... This may take a few minutes for full gameplay recordings...")
        st.stop()
    
    # Pure game display with informative sidebar
    if st.session_state.demos_ready:
        # Add helpful sidebar
        with st.sidebar:
            st.title("🎮 AI Atari Demo")
            
            st.subheader("🤖 What You're Seeing")
            st.markdown("""
            **Reinforcement Learning agents** are playing classic Atari games in real-time.
            
            Each agent uses a **random policy** with game-specific behaviors to demonstrate gameplay mechanics.
            
            The recordings show complete game sessions captured without preprocessing for authentic visuals.
            """)
            
            st.divider()
            
            st.subheader("🎯 Games Playing")
            # Show which games are currently displayed
            for game_id, gif_file in st.session_state.game_gif_files.items():
                if game_id in GAMES_CONFIG:
                    config = GAMES_CONFIG[game_id]
                    st.markdown(f"""
                    **{config['emoji']} {config['display_name']}**  
                    _{config['description']}_
                    """)
            
            st.divider()
            
            st.subheader("ℹ️ Technical Details")
            st.markdown("""
            **Environment**: Atari 2600 via Gymnasium  
            **Agent**: Random with game heuristics  
            **Rendering**: Raw, unprocessed frames  
            **Recording**: Full gameplay sessions  
            **Display**: Real-time game captures
            """)
        
        # Minimal main title
        st.title("🎮 AI Playing Atari Games")
        st.markdown("---")
        
        # Just the games
        show_pure_grid()
    
    else:
        # Loading state
        st.title("🎮 AI Atari Demo")
        st.info("🤖 Loading AI agents...")

if __name__ == "__main__":
    main()