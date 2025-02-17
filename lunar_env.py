import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import random

class LunarEnvironment(gym.Env):
    def __init__(self):
        super(LunarEnvironment, self).__init__()
        
        # Definisi space untuk aksi dan observasi
        # Aksi: [thrust_top_left, thrust_top_right, thrust_bottom_left, thrust_bottom_right]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),  # Setiap thruster 0-1
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Tambah target position ke observasi
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0, -np.inf, -np.inf, -np.inf, -np.inf]),  
            # [x, y, fuel, vel_x, vel_y, target_x, target_y]
            high=np.array([np.inf, np.inf, 100, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )
        
        # Inisialisasi pygame
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        
        # Ukuran probe dan thruster
        self.probe_size = 30
        self.thruster_width = 8
        self.thruster_height = 8
        self.thrust_animation_counter = 0
        self.thrust_particles = []  # Untuk animasi gas yang lebih dinamis
        
        # Parameter animasi gas
        self.particle_life = 5  # Durasi partikel gas
        self.particle_speed = 3  # Kecepatan partikel gas
        
        # Warna
        self.probe_color = (128, 128, 128)  # Abu-abu
        self.thrust_color = (255, 165, 0)   # Orange
        self.moon_colors = [
            (100, 100, 100),  # Abu-abu gelap
            (120, 120, 120),  # Abu-abu medium
            (140, 140, 140)   # Abu-abu terang
        ]
        
        # Parameter fisika
        self.gravity = 1.62
        self.dt = 0.05
        self.thrust_force = 2.0
        
        # Landing zones
        self.landing_zones = []
        self.generate_landing_zones()
        
        self.reset()
    
    def generate_landing_zones(self):
        # Buat beberapa landing zone di sepanjang permukaan bulan
        self.landing_zones = []
        zone_width = 60
        possible_x = range(zone_width, self.screen_width - zone_width, zone_width * 2)
        for x in possible_x:
            self.landing_zones.append(x)
    
    def _generate_craters(self):
        craters = []
        for _ in range(20):  # Buat 20 kawah
            x = random.randint(0, self.screen_width)
            y = random.randint(self.screen_height - 100, self.screen_height - 10)
            radius = random.randint(5, 15)
            craters.append((x, y, radius))
        return craters
    
    def reset(self):
        # Pilih posisi awal dan target secara random di bagian atas
        start_x = random.choice(self.landing_zones)
        target_x = random.choice([x for x in self.landing_zones if x != start_x])
        
        self.state = {
            'x': start_x,
            'y': self.screen_height * 0.2,  # 20% dari atas layar
            'fuel': 100.0,
            'vel_x': 0.0,
            'vel_y': 0.0,
            'target_x': target_x,
            'target_y': self.screen_height * 0.2
        }
        
        self.thrust_particles = []  # Reset partikel gas
        self.last_action = np.zeros(4)
        return self._get_observation()
    
    def step(self, action):
        # Simpan action untuk render
        self.last_action = action.copy()
        
        # Hitung total gaya dari semua thruster
        thrust_x = 0
        thrust_y = 0
        
        # Top thrusters (mendorong ke bawah)
        thrust_y += (action[0] + action[1]) * self.thrust_force
        
        # Bottom thrusters (mendorong ke atas)
        thrust_y -= (action[2] + action[3]) * self.thrust_force
        
        # Left thrusters (mendorong ke kanan)
        thrust_x += (action[1] + action[3]) * self.thrust_force
        
        # Right thrusters (mendorong ke kiri)
        thrust_x -= (action[0] + action[2]) * self.thrust_force
        
        # Update kecepatan dan posisi
        self.state['vel_x'] += thrust_x * self.dt
        self.state['vel_y'] += (thrust_y + self.gravity) * self.dt
        
        self.state['x'] += self.state['vel_x'] * self.dt
        self.state['y'] += self.state['vel_y'] * self.dt
        
        # Update fuel
        self.state['fuel'] -= sum(action) * self.dt
        
        # Hitung reward
        reward = self._calculate_reward()
        
        # Cek apakah episode selesai
        done = self._is_done()
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        return np.array([
            self.state['x'],
            self.state['y'],
            self.state['fuel'],
            self.state['vel_x'],
            self.state['vel_y'],
            self.state['target_x'],
            self.state['target_y']
        ])
    
    def _calculate_reward(self):
        reward = 0.0
        
        # Penalti untuk kecepatan tinggi
        speed = np.sqrt(self.state['vel_x']**2 + self.state['vel_y']**2)
        reward -= 0.1 * speed
        
        # Penalti untuk jarak dari target
        distance_to_target = np.sqrt(
            (self.state['x'] - self.state['target_x'])**2 +
            (self.state['y'] - self.state['target_y'])**2
        )
        reward -= 0.01 * distance_to_target
        
        # Bonus untuk mencapai target
        if self._is_at_target():
            reward += 100.0
        
        return reward
    
    def _is_done(self):
        # Keluar arena
        if (self.state['x'] < 0 or 
            self.state['x'] > self.screen_width or 
            self.state['y'] < 0 or 
            self.state['y'] > self.screen_height):
            return True
        
        # Kehabisan bahan bakar
        if self.state['fuel'] <= 0:
            return True
        
        # Mencapai target
        if self._is_at_target():
            return True
        
        return False
    
    def _is_at_target(self):
        distance_to_target = np.sqrt(
            (self.state['x'] - self.state['target_x'])**2 +
            (self.state['y'] - self.state['target_y'])**2
        )
        return (distance_to_target < 10 and 
                abs(self.state['vel_x']) < 2.0 and 
                abs(self.state['vel_y']) < 2.0)
    
    def _update_thrust_particles(self):
        # Update posisi partikel yang ada
        updated_particles = []
        for particle in self.thrust_particles:
            x, y, dx, dy, life, color = particle
            if life > 0:
                updated_particles.append((
                    x + dx,
                    y + dy,
                    dx * 0.95,  # Perlambatan
                    dy * 0.95,
                    life - 1,
                    color
                ))
        self.thrust_particles = updated_particles
    
    def render(self):
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        self.screen.fill((0, 0, 0))
        
        # Gambar permukaan bulan dengan gradasi dan kawah
        moon_surface = pygame.Surface((self.screen_width, 100))
        for y in range(100):
            color_idx = int(y / 33)  # Bagi permukaan menjadi 3 bagian
            if color_idx > 2: color_idx = 2
            pygame.draw.line(moon_surface, self.moon_colors[color_idx],
                           (0, y), (self.screen_width, y))
        
        # Gambar kawah
        for x, y, radius in self._generate_craters():
            pygame.draw.circle(moon_surface, self.moon_colors[0], 
                             (x, y - (self.screen_height - 100)), radius)
        
        self.screen.blit(moon_surface, (0, self.screen_height - 100))
        
        # Gambar landing zones
        for x in self.landing_zones:
            pygame.draw.line(self.screen, (80, 80, 80),
                           (x - 30, self.screen_height - 5),
                           (x + 30, self.screen_height - 5), 4)
        
        # Gambar probe (kotak abu-abu)
        probe_rect = pygame.Rect(
            self.state['x'] - self.probe_size//2,
            self.state['y'] - self.probe_size//2,
            self.probe_size,
            self.probe_size
        )
        pygame.draw.rect(self.screen, self.probe_color, probe_rect)
        
        # Update animation counter lebih cepat
        self.thrust_animation_counter = (self.thrust_animation_counter + 1) % 4
        
        # Update partikel gas
        self._update_thrust_particles()
        
        # Gambar thrusters
        thrusters = [
            # [x_offset, y_offset, action_index, direction_x, direction_y]
            [-self.probe_size//2, 0, 0, -1, 0],  # Left
            [self.probe_size//2, 0, 1, 1, 0],  # Right
            [0, -self.probe_size//2, 2, 0, -1],  # Top
            [0, self.probe_size//2, 3, 0, 1]  # Bottom
        ]
        
        for x_offset, y_offset, action_idx, dir_x, dir_y in thrusters:
            # Posisi thruster
            thruster_x = self.state['x'] + x_offset
            thruster_y = self.state['y'] + y_offset
            
            # Gambar thruster (kotak kecil)
            thruster_rect = pygame.Rect(
                thruster_x - self.thruster_width//2,
                thruster_y - self.thruster_height//2,
                self.thruster_width,
                self.thruster_height
            )
            pygame.draw.rect(self.screen, (100, 100, 100), thruster_rect)
            
            # Animasi gas jika thruster aktif
            if action_idx < len(self.last_action) and self.last_action[action_idx] > 0:
                thrust_power = self.last_action[action_idx]
                
                # Tambah partikel baru
                if self.thrust_animation_counter == 0:
                    for _ in range(int(thrust_power * 3)):  # Jumlah partikel berdasarkan power
                        spread = random.uniform(-0.5, 0.5)  # Penyebaran gas
                        speed = self.particle_speed * thrust_power
                        particle_dx = dir_x * speed + spread
                        particle_dy = dir_y * speed + spread
                        
                        # Warna api dari orange ke kuning
                        color_r = min(255, int(255 * (1 + random.uniform(-0.2, 0.2))))
                        color_g = min(255, int(165 * thrust_power * (1 + random.uniform(-0.2, 0.2))))
                        color_b = 0
                        
                        self.thrust_particles.append((
                            thruster_x + dir_x * self.thruster_width//2,
                            thruster_y + dir_y * self.thruster_height//2,
                            particle_dx,
                            particle_dy,
                            self.particle_life,
                            (color_r, color_g, color_b)
                        ))
        
        # Gambar semua partikel gas
        for particle in self.thrust_particles:
            x, y, _, _, life, color = particle
            size = int(2 + (life / self.particle_life) * 3)  # Partikel mengecil seiring waktu
            pygame.draw.circle(self.screen, color, (int(x), int(y)), size)
        
        # Gambar indikator posisi awal dan target di atas
        target_y = self.screen_height * 0.2
        pygame.draw.line(self.screen, (255, 0, 0),  # Posisi awal (merah)
                        (self.state['x'] - 15, target_y - 20),
                        (self.state['x'] + 15, target_y - 20), 4)
        pygame.draw.line(self.screen, (0, 255, 0),  # Target (hijau)
                        (self.state['target_x'] - 15, target_y - 20),
                        (self.state['target_x'] + 15, target_y - 20), 4)
        
        pygame.display.flip() 