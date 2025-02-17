# Lunar Probe Reinforcement Learning

Proyek ini mengimplementasikan simulasi lunar probe menggunakan Reinforcement Learning. Probe harus belajar untuk bernavigasi dari satu titik ke titik lain di atas permukaan bulan menggunakan 4 thruster.

## Fitur
- Simulasi fisika sederhana dengan gravitasi dan gaya dorong
- Visualisasi real-time menggunakan Pygame
- 4 thruster independen dengan efek visual gas roket
- Permukaan bulan dengan gradasi dan kawah
- Sistem reward yang mendorong efisiensi bahan bakar dan ketepatan landing

## Instalasi

1. Clone repository
```bash
git clone https://github.com/[username]/lunar-probe-rl.git
cd lunar-probe-rl
```

2. Buat environment conda
```bash
conda env create -f environment.yml
conda activate lunar_landing_rl
```

## Penggunaan

1. Training model
```bash
python train.py --episodes 1000 --render
```

Argumen yang tersedia:
- `--episodes`: Jumlah episode training (default: 1000)
- `--save-interval`: Interval penyimpanan checkpoint (default: 100)
- `--render`: Tampilkan visualisasi training
- `--resume`: Lanjutkan training dari checkpoint terakhir

## Struktur Proyek
```
lunar-probe-rl/
├── environment.yml     # Konfigurasi environment
├── lunar_env.py       # Implementasi environment lunar probe
├── model.py           # Arsitektur model RL (Actor-Critic)
├── train.py           # Script training
├── checkpoints/       # Model checkpoint
└── metrics/           # Grafik hasil training
```

## Implementasi Teknis

### Environment
- State space: [x, y, fuel, vel_x, vel_y, target_x, target_y]
- Action space: [thrust_left, thrust_right, thrust_top, thrust_bottom]
- Reward: Berdasarkan jarak ke target, penggunaan bahan bakar, dan kecepatan

### Model
- Arsitektur: Deep Deterministic Policy Gradient (DDPG)
- Actor network: 2 hidden layer (64 unit) dengan output sigmoid
- Critic network: 2 hidden layer (64 unit)

## License
MIT License