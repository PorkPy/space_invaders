#!/usr/bin/env python3
"""
Quick test script to verify all modules work before running Streamlit
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test all module imports"""
    print("🔍 Testing imports...")
    
    try:
        from config.settings import MODEL_CONFIG, GAME_CONFIG
        print("✅ Config imports successful")
        
        from src.models.model_loader import ModelLoader
        print("✅ Model loader import successful")
        
        from src.game.environment import SpaceInvadersEnvironment
        print("✅ Environment import successful")
        
        from src.models.inference import GameInference
        print("✅ Inference import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test model downloading and loading"""
    print("\n🤖 Testing model loading...")
    
    try:
        from src.models.model_loader import ModelLoader
        
        loader = ModelLoader()
        print(f"📥 Downloading model to: {loader.model_path}")
        
        # This will download the model (might take a moment)
        model = loader.load_model()
        
        if model is not None:
            print("✅ Model loaded successfully!")
            print(f"📊 Model info: {loader.get_model_info()}")
            return True
        else:
            print("❌ Failed to load model")
            return False
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_environment():
    """Test environment creation"""
    print("\n🎮 Testing environment...")
    
    try:
        from src.game.environment import SpaceInvadersEnvironment
        
        env = SpaceInvadersEnvironment()
        
        if env.create_environment():
            print("✅ Environment created successfully!")
            
            # Test reset
            obs = env.reset()
            if obs is not None:
                print(f"✅ Environment reset successful, observation shape: {obs.shape}")
                
                # Test action space
                action_space = env.get_action_space()
                print(f"🎯 Action space: {action_space}")
                
                env.close()
                return True
            else:
                print("❌ Failed to reset environment")
                return False
        else:
            print("❌ Failed to create environment")
            return False
            
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return False

def test_full_system():
    """Test complete inference system"""
    print("\n🧠 Testing full inference system...")
    
    try:
        from src.models.inference import GameInference
        
        game = GameInference()
        
        if game.initialize():
            print("✅ Inference system initialized!")
            
            # Start new game
            obs = game.start_new_game()
            if obs is not None:
                print("✅ Game started successfully!")
                
                # Take a few steps
                for i in range(3):
                    next_obs, reward, done, info, action = game.play_step(obs)
                    print(f"Step {i+1}: Action={action}, Reward={reward}, Done={done}")
                    
                    if done:
                        print("🎮 Game episode completed!")
                        break
                    obs = next_obs
                
                game.cleanup()
                print("✅ Full system test successful!")
                return True
            else:
                print("❌ Failed to start game")
                return False
        else:
            print("❌ Failed to initialize inference system")
            return False
            
    except Exception as e:
        print(f"❌ Full system error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Space Invaders RL - System Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("Environment", test_environment),
        ("Full System", test_full_system)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print("\n⚠️ Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to run Streamlit app:")
        print("streamlit run streamlit_app/main.py")
    else:
        print("\n⚠️ Some tests failed. Please fix issues before running Streamlit app.")

if __name__ == "__main__":
    main()