"""
Configuration for multiple Atari games - Updated versions with new games
"""

# Available games configuration with better environment versions
GAMES_CONFIG = {
    "space_invaders": {
        "env_name": "SpaceInvaders-v4",
        "display_name": "Space Invaders",
        "emoji": "👾",
        "description": "Classic alien shooter",
        "grid_size": (320, 200),
        "color": "#4CAF50"
    },
    "breakout": {
        "env_name": "Breakout-v4",
        "display_name": "Breakout",
        "emoji": "🧱",
        "description": "Paddle and ball physics",
        "grid_size": (320, 200),
        "color": "#2196F3"
    },
    "pong": {
        "env_name": "Pong-v4",
        "display_name": "Pong", 
        "emoji": "🏓",
        "description": "Classic paddle tennis",
        "grid_size": (320, 200),
        "color": "#FF9800"
    },
    "pacman": {
        "env_name": "MsPacman-v4",
        "display_name": "Ms. Pac-Man",
        "emoji": "👻", 
        "description": "Maze navigation",
        "grid_size": (320, 200),
        "color": "#E91E63"
    },
    "donkey_kong": {
        "env_name": "DonkeyKong-v4",
        "display_name": "Donkey Kong",
        "emoji": "🦍",
        "description": "Climb and jump adventure",
        "grid_size": (320, 200),
        "color": "#8B4513"
    },
    "centipede": {
        "env_name": "Centipede-v4",
        "display_name": "Centipede",
        "emoji": "🐛",
        "description": "Bug shooter defense",
        "grid_size": (320, 200),
        "color": "#9C27B0"
    }
}

# Grid layout configuration
GRID_CONFIG = {
    "columns": 3,  # 3x2 grid
    "rows": 2,
    "update_interval": 0.2,  # Seconds between frame updates
    "max_steps_per_game": 200,  # Steps to record per game
    "auto_restart": True  # Restart games when they end
}