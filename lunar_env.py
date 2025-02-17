import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import random
import math
import time

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
            high=np.array([np.inf, np.inf, 500, np.inf, np.inf, np.inf, np.inf]),  # Ubah dari 100 ke 500
            dtype=np.float32
        )
        
        # Inisialisasi pygame dengan flag yang benar
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Lunar Probe")
        
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
        
        # Parameter fisika yang disesuaikan
        self.gravity = 0.5  # Gravitasi dikurangi dari 1.62 ke 0.5
        self.dt = 0.05
        self.thrust_force = 1.0  # Thrust force juga disesuaikan
        
        # Tinggi permukaan bulan dari bawah layar
        self.moon_height = 100
        self.moon_surface_y = self.screen_height - self.moon_height
        
        # Ketinggian default untuk probe dan target
        self.probe_hover_height = 20  # Jarak dari permukaan bulan
        
        # Landing zones
        self.landing_zones = []
        self.generate_landing_zones()
        
        # Tambahan untuk objek luar angkasa
        self.stars = []
        self.galaxies = [
            {
                "name": "Andromeda",
                "color": [200, 200, 255],
                "position": (self.screen_width * 0.25, self.screen_height * 0.2),
                "size": 100,
                "rotation": 45,
                "rotation_speed": 0.001  # Dari 0.01 ke 0.001
            },
            {
                "name": "Sombrero",
                "color": [255, 200, 200],
                "position": (self.screen_width * 0.75, self.screen_height * 0.3),
                "size": 80,
                "rotation": -30,
                "rotation_speed": 0.0005  # Dari 0.005 ke 0.0005
            }
        ]
        self.asteroids = []
        self.asteroid_velocities = []
        self.asteroid_craters = []  # Menyimpan posisi kawah untuk setiap asteroid
        self.generate_space_objects()
        
        # Simpan posisi awal untuk penanda
        self.initial_x = None
        
        # Generate permukaan bulan sekali di awal
        self.moon_craters = self._generate_craters()
        
        # Tambah properti untuk fuel indicator
        self.fuel_icon_size = 40
        self.fuel_icon_padding = 10
        self.fuel_text_color = (255, 255, 255)  # Putih
        self.fuel_color = (255, 165, 0)  # Orange
        
        self.reset()
    
    def generate_landing_zones(self):
        # Buat beberapa landing zone di sepanjang permukaan bulan
        self.landing_zones = []
        zone_width = 60
        possible_x = range(zone_width, self.screen_width - zone_width, zone_width)
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
    
    def generate_space_objects(self):
        # Generate bintang
        self.stars = []
        for _ in range(100):
            x = random.randint(0, self.screen_width)
            y = random.randint(0, self.moon_surface_y - 50)
            brightness = random.uniform(0.5, 1.0)
            self.stars.append((x, y, brightness))
        
        # Reset galaksi dengan struktur yang benar
        self.galaxies = [
            {
                "name": "Andromeda",
                "color": [200, 200, 255],
                "position": (self.screen_width * 0.25, self.screen_height * 0.2),
                "size": 100,
                "rotation": 45,
                "rotation_speed": 0.001  # Dari 0.01 ke 0.001
            },
            {
                "name": "Sombrero",
                "color": [255, 200, 200],
                "position": (self.screen_width * 0.75, self.screen_height * 0.3),
                "size": 80,
                "rotation": -30,
                "rotation_speed": 0.0005  # Dari 0.005 ke 0.0005
            }
        ]
        
        # Generate asteroid dengan kawah statis
        self.asteroids = []
        self.asteroid_velocities = []
        self.asteroid_craters = []
        for _ in range(5):
            x = random.randint(0, self.screen_width)
            y = random.randint(0, self.moon_surface_y - 100)
            size = random.randint(15, 30)
            
            # Generate kawah statis untuk asteroid ini
            craters = []
            for _ in range(size // 5):  # Jumlah kawah proporsional dengan ukuran
                crater_angle = random.uniform(0, 2 * math.pi)
                crater_dist = random.uniform(0, size * 0.7)
                crater_x = crater_dist * math.cos(crater_angle)
                crater_y = crater_dist * math.sin(crater_angle)
                crater_size = random.uniform(size * 0.1, size * 0.3)
                craters.append((crater_x, crater_y, crater_size))
            
            self.asteroids.append([x, y, size])
            self.asteroid_craters.append(craters)
            # Kecepatan sangat lambat
            vx = random.uniform(-0.02, 0.02)  # dari -0.2 ke -0.02
            vy = random.uniform(-0.01, 0.01)  # dari -0.1 ke -0.01
            self.asteroid_velocities.append([vx, vy])
    
    def reset(self):
        # Generate ulang objek luar angkasa setiap episode baru
        self.generate_space_objects()
        
        # Generate ulang posisi kawah
        self.moon_craters = self._generate_craters()
        
        # Generate ulang posisi galaksi
        for galaxy in self.galaxies:
            galaxy["position"] = (
                random.uniform(100, self.screen_width-100),
                random.uniform(100, self.moon_surface_y-100)
            )
        
        # Pilih posisi awal dan target
        start_x = random.choice(self.landing_zones)
        target_x = random.choice([x for x in self.landing_zones if x != start_x])
        
        # Simpan posisi awal untuk penanda
        self.initial_x = start_x
        
        # Posisi y diatur relatif terhadap permukaan bulan
        probe_y = self.moon_surface_y - self.probe_hover_height
        
        self.state = {
            'x': start_x,
            'y': probe_y,
            'fuel': 500.0,  # Ubah dari 100 ke 500
            'vel_x': 0.0,
            'vel_y': 0.0,
            'target_x': target_x,
            'target_y': probe_y  # Target pada ketinggian yang sama
        }
        
        self.thrust_particles = []
        self.last_action = np.zeros(4)
        return self._get_observation()
    
    def step(self, action):
        self.last_action = action.copy()
        thrust_x = 0
        thrust_y = 0
        
        # Left thruster [0] mendorong ke kanan (+x)
        thrust_x += action[0] * self.thrust_force
        
        # Right thruster [1] mendorong ke kiri (-x) 
        thrust_x -= action[1] * self.thrust_force
        
        # Top thruster [2] mendorong ke bawah (+y)
        thrust_y += action[2] * self.thrust_force
        
        # Bottom thruster [3] mendorong ke atas (-y)
        thrust_y -= action[3] * self.thrust_force
        
        # Update kecepatan dan posisi
        self.state['vel_x'] += thrust_x * self.dt
        self.state['vel_y'] += (thrust_y + self.gravity) * self.dt
        
        self.state['x'] += self.state['vel_x'] * self.dt
        self.state['y'] += self.state['vel_y'] * self.dt
        
        # Update fuel
        self.state['fuel'] -= sum(action) * self.dt
        
        # Update posisi asteroid
        for i, (asteroid, velocity) in enumerate(zip(self.asteroids, self.asteroid_velocities)):
            asteroid[0] += velocity[0]
            asteroid[1] += velocity[1]
            
            # Wrap around screen
            if asteroid[0] < -50: asteroid[0] = self.screen_width + 50
            if asteroid[0] > self.screen_width + 50: asteroid[0] = -50
            if asteroid[1] < -50: asteroid[1] = self.moon_surface_y
            if asteroid[1] > self.moon_surface_y: asteroid[1] = -50
            
            self.asteroids[i] = asteroid
        
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
        
        # Cek tabrakan dengan permukaan bulan
        if self.state['y'] >= self.moon_surface_y - self.probe_size/2:
            done = True
            reward = -100  # Penalti untuk menabrak permukaan
            self.state['y'] = self.moon_surface_y - self.probe_size/2  # Prevent going below surface
            return self._get_observation(), reward, done, {}
        
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
        # Pastikan screen ada
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        
        # Handle pygame events - penting untuk responsivitas window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        # Bersihkan layar dengan warna hitam
        self.screen.fill((0, 0, 0))
        
        # Gambar bintang berkedip
        for x, y, brightness in self.stars:
            current_brightness = brightness * (0.7 + 0.3 * math.sin(time.time() * 5 + x * y))
            color = int(255 * current_brightness)
            pygame.draw.circle(self.screen, (color, color, color), (int(x), int(y)), 1)
        
        # Gambar galaksi dengan detail lebih baik
        for galaxy in self.galaxies:
            surface = pygame.Surface((200, 200), pygame.SRCALPHA)
            center = 100
            
            # Gambar inti galaksi
            color = galaxy["color"]  # Sekarang ini akan bekerja karena struktur data yang benar
            pygame.draw.circle(surface, (*color, 150), (center, center), 30)
            
            # Update rotasi galaksi
            galaxy["rotation"] += galaxy["rotation_speed"]
            
            # Gambar lengan spiral
            for arm in range(5):  # 5 lengan spiral
                start_angle = (2 * math.pi * arm / 5) + galaxy["rotation"]
                for r in range(10, 90, 2):
                    angle = start_angle + (r * 0.1)
                    x = center + r * math.cos(angle)
                    y = center + r * math.sin(angle)
                    
                    # Variasi transparansi berdasarkan radius
                    alpha = int(150 * (1 - r/90))
                    
                    # Gambar kelompok bintang di sepanjang lengan
                    for _ in range(3):
                        offset_x = random.uniform(-5, 5)
                        offset_y = random.uniform(-5, 5)
                        size = random.randint(1, 3)
                        pygame.draw.circle(surface, (*color, alpha),
                                        (int(x + offset_x), int(y + offset_y)), size)
            
            # Posisikan galaksi
            pos = galaxy["position"]
            self.screen.blit(surface, (pos[0] - 100, pos[1] - 100))
        
        # Gambar asteroid dengan kawah statis
        for (x, y, size), craters in zip(self.asteroids, self.asteroid_craters):
            # Gambar asteroid dasar
            pygame.draw.circle(self.screen, (169, 169, 169), (int(x), int(y)), size)
            
            # Gambar kawah statis
            for crater_x, crater_y, crater_size in craters:
                # Translasi posisi kawah relatif terhadap posisi asteroid
                abs_crater_x = x + crater_x
                abs_crater_y = y + crater_y
                pygame.draw.circle(self.screen, (100, 100, 100), 
                                 (int(abs_crater_x), int(abs_crater_y)), 
                                 int(crater_size))
        
        # Gambar permukaan bulan (tidak berubah selama episode)
        moon_surface = pygame.Surface((self.screen_width, self.moon_height))
        for y in range(self.moon_height):
            color_idx = int(y / (self.moon_height/3))
            if (color_idx > 2): color_idx = 2
            pygame.draw.line(moon_surface, self.moon_colors[color_idx],
                           (0, y), (self.screen_width, y))
        
        # Gambar kawah yang sudah di-generate
        for x, y, radius in self.moon_craters:
            pygame.draw.circle(moon_surface, self.moon_colors[0], 
                             (x, y - (self.screen_height - self.moon_height)), radius)
        
        self.screen.blit(moon_surface, (0, self.moon_surface_y))
        
        # Gambar landing zones dan penanda
        for x in self.landing_zones:
            # Landing zone (abu-abu)
            pygame.draw.line(self.screen, (80, 80, 80),
                           (x - 30, self.moon_surface_y - 5),
                           (x + 30, self.moon_surface_y - 5), 4)
        
        # Penanda posisi awal (merah, tetap)
        pygame.draw.line(self.screen, (255, 0, 0),
                        (self.initial_x - 30, self.moon_surface_y - 5),
                        (self.initial_x + 30, self.moon_surface_y - 5), 4)
        
        # Penanda target (hijau)
        pygame.draw.line(self.screen, (0, 255, 0),
                        (self.state['target_x'] - 30, self.moon_surface_y - 5),
                        (self.state['target_x'] + 30, self.moon_surface_y - 5), 4)
        
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
        
        # Gambar fuel indicator
        fuel_percentage = (self.state['fuel'] / 500.0) * 100
        
        # Gambar icon bahan bakar
        icon_x = self.screen_width - self.fuel_icon_size - self.fuel_icon_padding
        icon_y = self.fuel_icon_padding
        
        # Tentukan warna berdasarkan persentase
        current_fuel_color = (255, 0, 0) if fuel_percentage <= 50 else self.fuel_color
        
        # Gambar wadah bahan bakar (outline)
        pygame.draw.rect(self.screen, current_fuel_color, 
                        (icon_x, icon_y, self.fuel_icon_size, self.fuel_icon_size), 2)
        
        # Gambar isi bahan bakar
        fuel_height = int((fuel_percentage / 100) * (self.fuel_icon_size - 4))
        fuel_y = icon_y + (self.fuel_icon_size - 2) - fuel_height
        if fuel_height > 0:
            pygame.draw.rect(self.screen, current_fuel_color,
                           (icon_x + 2, fuel_y, self.fuel_icon_size - 4, fuel_height))
        
        # Gambar label "Fuel"
        font = pygame.font.Font(None, 24)
        fuel_label = font.render("Fuel", True, self.fuel_text_color)
        label_rect = fuel_label.get_rect()
        label_rect.right = icon_x - 5
        label_rect.bottom = icon_y + self.fuel_icon_size//2 - 5  # Posisikan di atas persentase
        self.screen.blit(fuel_label, label_rect)
        
        # Gambar text persentase
        text = font.render(f"{int(fuel_percentage)}%", True, self.fuel_text_color)
        text_rect = text.get_rect()
        text_rect.right = icon_x - 5
        text_rect.top = icon_y + self.fuel_icon_size//2 + 5  # Posisikan di bawah label
        self.screen.blit(text, text_rect)
        
        # Pastikan display diupdate
        pygame.display.update()
        pygame.event.pump()  # Handle events untuk mencegah "not responding"