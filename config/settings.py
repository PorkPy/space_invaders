"""
Configuration settings for Space Invaders RL project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Game settings
GAME_CONFIG = {
    "environment_name": "ALE/SpaceInvaders-v5",  # Use base environment without NoFrameskip
    "render_mode": "rgb_array",
    "frame_skip": 1,  # Reduce frame skip to avoid conflict
    "screen_size": 84,
    "frame_stack": 4,
}

# Model settings
MODEL_CONFIG = {
    "algorithm": "DQN",
    # Using a different model URL that's more compatible
    "model_url": "https://huggingface.co/sb3/dqn-SpaceInvadersNoFrameskip-v4/resolve/main/dqn-SpaceInvadersNoFrameskip-v4.zip",
    "local_model_path": MODEL_DIR / "space_invaders_dqn.zip",
    "device": "cpu",  # Use CPU for demo, GPU if available
    "use_random_agent": True,  # Fallback to random agent if model fails
}

# Streamlit settings
STREAMLIT_CONFIG = {
    "page_title": "Space Invaders RL Demo",
    "page_icon": "👾",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# API settings (for AWS deployment)
API_CONFIG = {
    "timeout": 30,
    "max_retries": 3,
    "base_url": os.getenv("API_BASE_URL", "http://localhost:8000"),
}

# Logging settings
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": LOGS_DIR / "space_invaders_rl.log",
}