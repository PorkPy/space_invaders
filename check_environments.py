#!/usr/bin/env python3
"""
Check available Space Invaders environment variants
"""
import gymnasium as gym
import ale_py

def main():
    print("🎮 Checking available Space Invaders environments...")
    print("=" * 60)
    
    # Get all registered environments (different API for newer gymnasium)
    try:
        # Try new API first
        all_env_ids = list(gym.envs.registry.env_specs.keys())
    except:
        try:
            # Try older API
            all_env_ids = list(gym.envs.registry.all())
            all_env_ids = [env.id for env in all_env_ids]
        except:
            # Manual fallback
            all_env_ids = list(gym.envs.registry.keys()) if hasattr(gym.envs.registry, 'keys') else []
    
    # Filter for Space Invaders variants
    space_invaders_envs = [env for env in all_env_ids if 'space' in env.lower() or 'invaders' in env.lower()]
    
    print('Available Space Invaders environments:')
    for env in sorted(space_invaders_envs):
        print(f'  - {env}')
    
    # Test a few common ones manually
    print('\nTesting common Space Invaders variants:')
    common_variants = [
        'SpaceInvaders-v0',
        'SpaceInvaders-v4', 
        'SpaceInvadersDeterministic-v4',
        'SpaceInvadersNoFrameskip-v4',
        'ALE/SpaceInvaders-v5'
    ]
    
    for variant in common_variants:
        try:
            env = gym.make(variant, render_mode='rgb_array')
            print(f'  ✅ {variant} - Available')
            env.close()
        except:
            print(f'  ❌ {variant} - Not available')
    
    # Also check ALE ROM variants
    try:
        from ale_py.roms import get_all_rom_ids
        roms = get_all_rom_ids()
        space_roms = [r for r in roms if 'space' in r.lower() or 'invader' in r.lower()]
        print('\nSpace Invaders ROM variants:')
        for rom in sorted(space_roms):
            print(f'  - {rom}')
    except Exception as e:
        print(f'\nCould not check ROM variants: {e}')
    
    print("\n" + "=" * 60)
    print("Current environment: SpaceInvaders-v4")
    print("Try these alternatives for different alien behavior:")
    print("  - SpaceInvaders-v0 (original version)")
    print("  - SpaceInvadersDeterministic-v4 (deterministic)")
    print("  - SpaceInvadersNoFrameskip-v4 (no frame skip)")

if __name__ == "__main__":
    main()