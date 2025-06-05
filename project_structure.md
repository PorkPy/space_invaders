# Space Invaders RL Project Structure

```
space-invaders-rl/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── __init__.py
│   └── settings.py              # Environment configs, model paths
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_loader.py      # Load/initialize RL models
│   │   └── inference.py         # Model prediction logic
│   ├── game/
│   │   ├── __init__.py
│   │   ├── environment.py       # Atari environment setup
│   │   ├── renderer.py          # Game frame rendering
│   │   └── controller.py        # Game state management
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py           # Common utilities
│   │   └── logging_config.py    # Logging setup
│   └── api/
│       ├── __init__.py
│       └── endpoints.py         # API interface (for AWS later)
├── streamlit_app/
│   ├── __init__.py
│   ├── main.py                  # Main Streamlit entry point
│   ├── components/
│   │   ├── __init__.py
│   │   ├── game_display.py      # Game visualization component
│   │   ├── controls.py          # User interaction controls
│   │   └── metrics.py           # Performance metrics display
│   └── pages/
│       ├── __init__.py
│       ├── gameplay.py          # Live gameplay page
│       ├── analysis.py          # Model analysis page
│       └── about.py             # Project info page
├── docker/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── entrypoint.sh
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── lambda.tf
│   ├── s3.tf
│   └── api_gateway.tf
└── tests/
    ├── __init__.py
    ├── test_models.py
    ├── test_game.py
    └── test_api.py
```

## Benefits of This Structure:

### 🎯 **Separation of Concerns:**
- **Models**: RL logic isolated and testable
- **Game**: Environment and rendering separate from AI
- **Streamlit**: UI components reusable and focused
- **API**: Clean interface for AWS deployment

### ⚡ **Easy Updates:**
- Update individual modules without touching others
- Clear dependencies between components
- Each artifact can target specific files

### 🔧 **Reusable Infrastructure:**
- Docker/Terraform setup independent of application code
- Same deployment pipeline for any RL game
- Configuration-driven rather than hardcoded

### 🧪 **Testable:**
- Each module can be unit tested
- Mock interfaces for development
- Clear boundaries for debugging

## Development Workflow:
1. **Start with** `src/models/model_loader.py` - Get RL model working
2. **Add** `src/game/environment.py` - Set up Atari environment  
3. **Build** `streamlit_app/components/game_display.py` - Basic visualization
4. **Integrate** `streamlit_app/main.py` - Tie components together
5. **Deploy** using existing Docker/Terraform pipeline

Ready to start with the first module?