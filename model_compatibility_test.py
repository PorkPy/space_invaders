"""
Model compatibility checker - verifies pre-trained models work with our environment
"""
import streamlit as st
import sys
import numpy as np
import gymnasium as gym
import ale_py
from pathlib import Path
import requests
import json
from stable_baselines3 import DQN
import tempfile
import traceback

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Model Compatibility Checker",
    page_icon="🔍",
    layout="wide"
)

def download_model_metadata(model_url: str) -> dict:
    """Download and parse model metadata from HuggingFace"""
    try:
        # Try to get the model config
        base_url = model_url.replace('/resolve/main/', '/raw/main/')
        config_url = base_url.replace('.zip', '') + '/config.json'
        
        st.write(f"🔍 Checking model metadata: {config_url}")
        response = requests.get(config_url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"Could not fetch model config (HTTP {response.status_code})")
            return {}
    except Exception as e:
        st.warning(f"Could not fetch model metadata: {e}")
        return {}

def test_environment_creation(env_name: str) -> dict:
    """Test creating the environment and get its properties"""
    results = {"success": False, "error": None, "properties": {}}
    
    try:
        st.write(f"🎮 Testing environment: {env_name}")
        
        # Test base environment
        env = gym.make(env_name, render_mode="rgb_array")
        
        results["properties"]["base_action_space"] = env.action_space.n
        results["properties"]["base_obs_space"] = env.observation_space.shape
        
        # Test with standard preprocessing
        env_preprocessed = gym.make(env_name, render_mode=None, frameskip=1)
        env_preprocessed = gym.wrappers.AtariPreprocessing(
            env_preprocessed,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=True
        )
        env_preprocessed = gym.wrappers.FrameStackObservation(env_preprocessed, 4)
        
        results["properties"]["preprocessed_action_space"] = env_preprocessed.action_space.n
        results["properties"]["preprocessed_obs_space"] = env_preprocessed.observation_space.shape
        
        # Test reset and step
        obs, _ = env_preprocessed.reset()
        results["properties"]["actual_obs_shape"] = obs.shape
        results["properties"]["obs_dtype"] = str(obs.dtype)
        results["properties"]["obs_min"] = float(obs.min())
        results["properties"]["obs_max"] = float(obs.max())
        
        # Take a step
        obs2, reward, done, truncated, info = env_preprocessed.step(1)
        results["properties"]["step_successful"] = True
        
        env.close()
        env_preprocessed.close()
        
        results["success"] = True
        
    except Exception as e:
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
    
    return results

def test_model_loading(model_url: str) -> dict:
    """Test downloading and loading the model with aggressive compatibility fixes"""
    results = {"success": False, "error": None, "properties": {}}
    
    try:
        st.write(f"📥 Downloading model: {model_url}")
        
        # Download model
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
            model_path = f.name
        
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        st.write("✅ Model downloaded")
        
        # Load model with aggressive compatibility fixes
        st.write("🤖 Loading model with aggressive compatibility fixes...")
        
        # Custom objects to handle old model compatibility
        custom_objects = {
            "learning_rate": 0.0001,
            "lr_schedule": lambda x: 0.0001,
            "exploration_schedule": lambda x: 0.1,
        }
        
        try:
            # Try loading without setting up the model (for inference only)
            import torch
            from stable_baselines3.common.policies import BasePolicy
            
            # Load just the policy for inference
            st.write("🎯 Attempting policy-only loading...")
            
            # Load the saved data
            data, params, pytorch_variables = DQN._load_from_file(model_path, custom_objects=custom_objects)
            
            # Create a minimal model just for inference
            model = DQN(
                policy="CnnPolicy",
                env=None,  # We'll provide observations directly
                learning_rate=0.0001,
                buffer_size=1,  # Minimal buffer
                learning_starts=1,
                target_update_interval=1,
                train_freq=1,
                gradient_steps=1,
                exploration_fraction=0.1,
                exploration_initial_eps=0.1,
                exploration_final_eps=0.02,
                optimize_memory_usage=False,  # Fix the compatibility issue
                verbose=0
            )
            
            # Set the policy parameters
            model.policy.load_state_dict(pytorch_variables)
            model.policy.eval()  # Set to evaluation mode
            
            results["properties"]["load_method"] = "policy_only_inference"
            results["properties"]["model_type"] = str(type(model))
            results["properties"]["policy_type"] = str(type(model.policy))
            
        except Exception as e1:
            st.write(f"⚠️ Policy-only loading failed: {e1}")
            st.write("🔄 Trying different approach...")
            
            # Fallback: Create new model and load only the policy weights
            try:
                # Create a new model with compatible settings
                from gymnasium.spaces import Box, Discrete
                
                # Mock environment spaces
                observation_space = Box(low=0, high=1, shape=(4, 84, 84), dtype=np.float32)
                action_space = Discrete(6)
                
                model = DQN(
                    policy="CnnPolicy",
                    env=None,
                    learning_rate=0.0001,
                    buffer_size=100,  # Small buffer
                    learning_starts=100,
                    target_update_interval=1000,
                    train_freq=4,
                    gradient_steps=1,
                    exploration_fraction=0.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.01,
                    optimize_memory_usage=False,  # This should fix the issue
                    handle_timeout_termination=False,  # This too
                    verbose=0
                )
                
                # Set the environment spaces manually
                model._setup_lr_schedule()
                model.observation_space = observation_space
                model.action_space = action_space
                
                # Try to load just the policy
                data, params, pytorch_variables = DQN._load_from_file(model_path, custom_objects=custom_objects)
                model.policy.load_state_dict(pytorch_variables)
                model.policy.eval()
                
                results["properties"]["load_method"] = "new_model_with_old_weights"
                results["properties"]["model_type"] = str(type(model))
                
            except Exception as e2:
                st.write(f"⚠️ Fallback loading failed: {e2}")
                raise e2
        
        # Test a simple prediction to make sure model works
        dummy_obs = np.zeros((4, 84, 84), dtype=np.float32)
        try:
            action, _ = model.predict(dummy_obs, deterministic=True)
            results["properties"]["test_prediction_successful"] = True
            results["properties"]["test_action"] = int(action)
            st.write(f"✅ Test prediction successful: action = {action}")
        except Exception as pred_e:
            results["properties"]["test_prediction_successful"] = False
            results["properties"]["test_prediction_error"] = str(pred_e)
            st.write(f"⚠️ Test prediction failed: {pred_e}")
        
        results["success"] = True
        
        # Cleanup
        Path(model_path).unlink()
        
    except Exception as e:
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
    
    return results

def test_model_environment_compatibility(model_url: str, env_name: str) -> dict:
    """Test if model can actually make predictions on environment observations"""
    results = {"success": False, "error": None, "predictions": []}
    
    try:
        st.write("🔗 Testing model-environment compatibility...")
        
        # Download and load model with compatibility fixes
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
            model_path = f.name
        
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Load with compatibility fixes
        custom_objects = {
            "learning_rate": 0.0001,
            "lr_schedule": lambda x: 0.0001,
            "exploration_schedule": lambda x: 0.1,
        }
        
        try:
            model = DQN.load(model_path, custom_objects=custom_objects)
        except ValueError as e:
            if "ReplayBuffer" in str(e):
                model = DQN.load(model_path, custom_objects=custom_objects, 
                                replay_buffer=None)
            else:
                raise e
        
        # Create environment
        env = gym.make(env_name, render_mode=None, frameskip=1)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=True
        )
        env = gym.wrappers.FrameStackObservation(env, 4)
        
        # Test predictions
        obs, _ = env.reset()
        
        for step in range(10):
            try:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                
                results["predictions"].append({
                    "step": step,
                    "action": int(action),
                    "reward": float(reward),
                    "obs_shape": obs.shape,
                    "obs_range": [float(obs.min()), float(obs.max())]
                })
                
                if done or truncated:
                    obs, _ = env.reset()
                    
            except Exception as e:
                results["predictions"].append({
                    "step": step,
                    "error": str(e)
                })
                break
        
        env.close()
        Path(model_path).unlink()
        
        results["success"] = True
        
    except Exception as e:
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
    
    return results

def main():
    """Main compatibility checker application"""
    
    st.title("🔍 Model Compatibility Checker")
    st.markdown("**Verify pre-trained models work with our environment setup**")
    
    # Test configuration
    with st.sidebar:
        st.subheader("🎯 Test Configuration")
        
        model_urls = {
            "SpaceInvaders DQN": "https://huggingface.co/sb3/dqn-SpaceInvadersNoFrameskip-v4/resolve/main/dqn-SpaceInvadersNoFrameskip-v4.zip",
            "Breakout DQN": "https://huggingface.co/sb3/dqn-BreakoutNoFrameskip-v4/resolve/main/dqn-BreakoutNoFrameskip-v4.zip",
            "Pong DQN": "https://huggingface.co/sb3/dqn-PongNoFrameskip-v4/resolve/main/dqn-PongNoFrameskip-v4.zip"
        }
        
        selected_model = st.selectbox("Select Model", list(model_urls.keys()))
        model_url = model_urls[selected_model]
        
        # Corresponding environment names
        env_mapping = {
            "SpaceInvaders DQN": "SpaceInvadersNoFrameskip-v4",
            "Breakout DQN": "BreakoutNoFrameskip-v4", 
            "Pong DQN": "PongNoFrameskip-v4"
        }
        
        env_name = env_mapping[selected_model]
        
        st.write(f"**Model**: {selected_model}")
        st.write(f"**Environment**: {env_name}")
        
        if st.button("🚀 Run Compatibility Test", use_container_width=True):
            st.session_state.run_test = True
            st.session_state.model_url = model_url
            st.session_state.env_name = env_name
    
    # Main content
    if hasattr(st.session_state, 'run_test') and st.session_state.run_test:
        
        model_url = st.session_state.model_url
        env_name = st.session_state.env_name
        
        st.subheader(f"🧪 Testing {selected_model}")
        
        # Test 1: Model Metadata
        with st.expander("📋 Model Metadata", expanded=True):
            metadata = download_model_metadata(model_url)
            if metadata:
                st.json(metadata)
            else:
                st.info("No metadata available")
        
        # Test 2: Environment Creation
        with st.expander("🎮 Environment Test", expanded=True):
            env_results = test_environment_creation(env_name)
            if env_results["success"]:
                st.success("✅ Environment creation successful")
                st.json(env_results["properties"])
            else:
                st.error("❌ Environment creation failed")
                st.code(env_results["error"])
                if "traceback" in env_results:
                    st.code(env_results["traceback"])
        
        # Test 3: Model Loading
        with st.expander("🤖 Model Loading Test", expanded=True):
            model_results = test_model_loading(model_url)
            if model_results["success"]:
                st.success("✅ Model loading successful")
                st.json(model_results["properties"])
            else:
                st.error("❌ Model loading failed")
                st.code(model_results["error"])
                if "traceback" in model_results:
                    st.code(model_results["traceback"])
        
        # Test 4: Compatibility Test
        with st.expander("🔗 Model-Environment Compatibility", expanded=True):
            if env_results["success"] and model_results["success"]:
                compat_results = test_model_environment_compatibility(model_url, env_name)
                if compat_results["success"]:
                    st.success("✅ Model-environment compatibility confirmed!")
                    
                    # Show predictions
                    st.subheader("🎯 Sample Predictions")
                    for pred in compat_results["predictions"]:
                        if "error" not in pred:
                            st.write(f"Step {pred['step']}: Action {pred['action']}, Reward {pred['reward']}")
                        else:
                            st.error(f"Step {pred['step']}: {pred['error']}")
                            
                    # Summary
                    successful_steps = len([p for p in compat_results["predictions"] if "error" not in p])
                    st.metric("Successful Prediction Steps", f"{successful_steps}/10")
                    
                    if successful_steps >= 8:
                        st.success("🎉 Model appears to be working correctly!")
                    else:
                        st.warning("⚠️ Model may have compatibility issues")
                        
                else:
                    st.error("❌ Model-environment compatibility failed")
                    st.code(compat_results["error"])
                    if "traceback" in compat_results:
                        st.code(compat_results["traceback"])
            else:
                st.warning("⏭️ Skipping compatibility test due to previous failures")
        
        # Reset test flag
        st.session_state.run_test = False
    
    else:
        # Instructions
        st.markdown("""
        ### 🎯 How This Works
        
        This tool tests whether pre-trained models from HuggingFace will work with our environment setup:
        
        **🔍 Tests Performed:**
        1. **Model Metadata** - Check what environment the model expects
        2. **Environment Creation** - Verify we can create the environment
        3. **Model Loading** - Test downloading and loading the model
        4. **Compatibility** - Verify model can make predictions on our observations
        
        **✅ What We're Looking For:**
        - Matching observation shapes (4, 84, 84)
        - Compatible action spaces
        - Successful predictions without errors
        - Reasonable action choices
        
        **👈 Select a model and click 'Run Compatibility Test' to begin!**
        """)

if __name__ == "__main__":
    main()