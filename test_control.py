import pygame
import sys
from lunar_env import LunarEnvironment

def manual_control():
    pygame.init()
    env = LunarEnvironment()
    state = env.reset()
    
    # Render initial state
    env.render()  # Add this line to show initial state
    pygame.display.flip()
    
    # Kecepatan thrust untuk kontrol manual
    thrust_power = 3
    
    print("\nKontrol Lunar Probe:")
    print("←: Thruster kanan (bergerak ke kiri)")
    print("→: Thruster kiri (bergerak ke kanan)")
    print("↑: Thruster bawah (bergerak ke atas)")
    print("↓: Thruster atas (bergerak ke bawah)")
    print("Space: Reset posisi")
    print("Q: Keluar\n")
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Reset actions setiap frame
        actions = [0, 0, 0, 0]
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    state = env.reset()
                    env.render()  # Add render after reset
                    pygame.display.flip()
        
        # Get continuous key states
        keys = pygame.key.get_pressed()
        
        # Mapping keyboard ke actions:
        # actions[0] = left thruster (bergerak ke kanan)
        # actions[1] = right thruster (bergerak ke kiri)
        # actions[2] = top thruster (bergerak ke bawah)
        # actions[3] = bottom thruster (bergerak ke atas)
        
        if keys[pygame.K_LEFT]:
            actions[1] = thrust_power  # Aktifkan right thruster
        if keys[pygame.K_RIGHT]:
            actions[0] = thrust_power  # Aktifkan left thruster
        if keys[pygame.K_UP]:
            actions[3] = thrust_power  # Aktifkan bottom thruster
        if keys[pygame.K_DOWN]:
            actions[2] = thrust_power  # Aktifkan top thruster
            
        # Apply actions and update environment
        state, reward, done, _ = env.step(actions)
        env.render()  # Make sure to render after each step
        pygame.display.flip()
        
        if done:
            state = env.reset()
            env.render()  # Add render after reset on done
            pygame.display.flip()
            
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    manual_control()