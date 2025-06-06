"""
Debug version with detailed error logging
"""
import streamlit as st
import sys
import traceback
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Debug Multi Test",
    page_icon="🔍",
    layout="wide"
)

def test_basic_imports():
    """Test if basic imports work"""
    st.subheader("🔍 Testing Basic Imports")
    
    try:
        import ale_py
        st.success("✅ ale_py imported")
    except Exception as e:
        st.error(f"❌ ale_py failed: {e}")
        return False
    
    try:
        import gymnasium as gym
        st.success("✅ gymnasium imported")
    except Exception as e:
        st.error(f"❌ gymnasium failed: {e}")
        return False
    
    try:
        from src.game.environment import AtariEnvironment
        st.success("✅ AtariEnvironment imported")
    except Exception as e:
        st.error(f"❌ AtariEnvironment failed: {e}")
        st.code(traceback.format_exc())
        return False
    
    try:
        from src.models.inference import GameInference
        st.success("✅ GameInference imported")
    except Exception as e:
        st.error(f"❌ GameInference failed: {e}")
        st.code(traceback.format_exc())
        return False
    
    return True

def test_single_environment():
    """Test creating a single environment"""
    st.subheader("🎮 Testing Single Environment")
    
    try:
        from src.game.environment import AtariEnvironment
        
        st.info("Creating AtariEnvironment...")
        env = AtariEnvironment("SpaceInvaders-v0")
        st.success("✅ AtariEnvironment created")
        
        st.info("Creating gymnasium environment...")
        try:
            result = env.create_environment()
            if result:
                st.success("✅ Gymnasium environment created successfully")
                
                st.info("Testing environment reset...")
                obs = env.reset()
                if obs is not None:
                    st.success(f"✅ Environment reset successful, observation shape: {obs.shape}")
                    
                    st.info("Testing environment step...")
                    obs2, reward, done, info = env.step(1)  # FIRE action
                    if obs2 is not None:
                        st.success(f"✅ Environment step successful, reward: {reward}")
                    else:
                        st.error("❌ Environment step returned None observation")
                    
                    st.info("Testing environment render...")
                    frame = env.render()
                    if frame is not None:
                        st.success(f"✅ Environment render successful, frame shape: {frame.shape}")
                    else:
                        st.error("❌ Environment render returned None")
                    
                else:
                    st.error("❌ Environment reset returned None")
                
                env.close()
                st.success("✅ Environment closed successfully")
                return True
            else:
                st.error("❌ Failed to create gymnasium environment")
                return False
        except Exception as e:
            st.error(f"❌ Environment creation failed with exception: {e}")
            st.code(traceback.format_exc())
            return False
            
    except Exception as e:
        st.error(f"❌ Environment test failed: {e}")
        st.code(traceback.format_exc())
        return False

def test_game_inference():
    """Test GameInference creation"""
    st.subheader("🤖 Testing GameInference")
    
    try:
        from src.models.inference import GameInference
        
        st.info("Creating GameInference...")
        game = GameInference("SpaceInvaders-v0")
        st.success("✅ GameInference created")
        
        st.info("Initializing GameInference...")
        if game.initialize():
            st.success("✅ GameInference initialized successfully")
            
            st.info("Starting new game...")
            obs = game.start_new_game()
            if obs is not None:
                st.success(f"✅ Game started, observation shape: {obs.shape}")
                
                st.info("Testing game step...")
                new_obs, reward, done, info, action = game.play_step(obs)
                if new_obs is not None:
                    st.success(f"✅ Game step successful, action: {action}, reward: {reward}")
                else:
                    st.error("❌ Game step returned None observation")
                
                st.info("Testing frame rendering...")
                frame = game.get_render_frame()
                if frame is not None:
                    st.success(f"✅ Frame rendered, shape: {frame.shape}")
                    
                    # Display the frame
                    from PIL import Image
                    import cv2
                    import numpy as np
                    
                    display_frame = frame
                    if len(display_frame.shape) == 2:
                        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2RGB)
                    
                    if display_frame.dtype != np.uint8:
                        if display_frame.max() <= 1.0:
                            display_frame = (display_frame * 255).astype(np.uint8)
                        else:
                            display_frame = display_frame.astype(np.uint8)
                    
                    pil_image = Image.fromarray(display_frame)
                    st.image(pil_image, caption="Test Frame", width=300)
                else:
                    st.error("❌ Frame rendering returned None")
                
            else:
                st.error("❌ Game start returned None observation")
            
            game.cleanup()
            st.success("✅ GameInference cleaned up")
            return True
        else:
            st.error("❌ GameInference initialization failed")
            return False
            
    except Exception as e:
        st.error(f"❌ GameInference test failed: {e}")
        st.code(traceback.format_exc())
        return False

def test_available_environments():
    """Test which environments are actually available"""
    st.subheader("🎯 Testing Available Environments")
    
    try:
        import gymnasium as gym
        import ale_py
        
        test_envs = [
            "SpaceInvaders-v0",
            "SpaceInvaders-v4",
            "SpaceInvadersNoFrameskip-v4",
            "Breakout-v0",
            "Pong-v0"
        ]
        
        available = []
        
        for env_name in test_envs:
            try:
                env = gym.make(env_name, render_mode="rgb_array")
                env.close()
                available.append(env_name)
                st.success(f"✅ {env_name} available")
            except Exception as e:
                st.error(f"❌ {env_name} not available: {str(e)[:100]}...")
        
        if available:
            st.info(f"Found {len(available)} available environments")
            return available
        else:
            st.error("No environments available!")
            return []
            
    except Exception as e:
        st.error(f"❌ Environment availability test failed: {e}")
        st.code(traceback.format_exc())
        return []

def main():
    st.title("🔍 Debug Multi-Game Test")
    st.markdown("Let's debug step by step to find what's failing...")
    
    # Test each component
    if st.button("🔍 Run All Debug Tests"):
        
        st.markdown("---")
        if not test_basic_imports():
            st.error("🛑 Basic imports failed - fix imports first!")
            return
        
        st.markdown("---")
        available_envs = test_available_environments()
        if not available_envs:
            st.error("🛑 No environments available - install ale-py ROMs!")
            st.code("pip install ale-py[accept-rom-license]")
            return
        
        st.markdown("---")
        if not test_single_environment():
            st.error("🛑 Single environment failed!")
            return
        
        st.markdown("---")
        if not test_game_inference():
            st.error("🛑 GameInference failed!")
            return
        
        st.markdown("---")
        st.success("🎉 All tests passed! Multi-game should work now.")
        
        # Show next steps
        st.subheader("✅ Next Steps")
        st.markdown("""
        Since all tests passed, the multi-game grid should work. Try:
        1. Go back to the main app
        2. Select games in the sidebar
        3. Click 'Start Multi-Game Grid'
        
        If it still doesn't work, check the console logs where you're running Streamlit.
        """)

if __name__ == "__main__":
    main()