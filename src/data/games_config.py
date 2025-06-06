"""
Configuration for multiple Atari games - Updated versions
"""

# Available games configuration with better environment versions
GAMES_CONFIG = {
    "space_invaders": {
        "env_name": "SpaceInvaders-v4",  # Using v4 like we did before
        "display_name": "Space Invaders",
        "emoji": "👾",
        "description": "Classic alien shooter",
        "grid_size": (320, 200),
        "color": "#4CAF50"
    },
    "breakout": {
        "env_name": "Breakout-v4",  # Using v4 for consistency
        "display_name": "Breakout",
        "emoji": "🧱",
        "description": "Paddle and ball physics",
        "grid_size": (320, 200),
        "color": "#2196F3"
    },
    "pong": {
        "env_name": "Pong-v4",  # Using v4 for consistency
        "display_name": "Pong", 
        "emoji": "🏓",
        "description": "Classic paddle tennis",
        "grid_size": (320, 200),
        "color": "#FF9800"
    },
    "pacman": {
        "env_name": "MsPacman-v4",  # Using v4 for consistency
        "display_name": "Ms. Pac-Man",
        "emoji": "👻", 
        "description": "Maze navigation",
        "grid_size": (320, 200),
        "color": "#E91E63"
    },
    "asteroids": {
        "env_name": "Asteroids-v4",  # Changed to v4 to fix the odd behavior
        "display_name": "Asteroids",
        "emoji": "🚀",
        "description": "Space shooter",
        "grid_size": (320, 200), 
        "color": "#9C27B0"
    },
    "frostbite": {
        "env_name": "Frostbite-v4",  # Using v4 for consistency
        "display_name": "Frostbite", 
        "emoji": "🧊",
        "description": "Arctic adventure",
        "grid_size": (320, 200),
        "color": "#00BCD4"
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