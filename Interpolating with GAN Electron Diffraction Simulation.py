import taichi as ti
import numpy as np
import math
import matplotlib
import scipy.constants
from sympy import symbols, Eq, solve
from math import gcd
import random
import torch
import torch.nn as nn

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

# --- GAN 모델 관련 상수 ---
DATA_DIM = 12  # 위치(3) + 온도(1) + 밀도(1) + 기존속도(3) + 계산된속도(3) + 가속도(3)

# --- 시뮬레이션 상수 및 매개변수 ---
DIM = 2
MAX_ELECTRONS = 50000
MAX_AIR_PARTICLES = 1500000# 최대 공기 입자 수를 합리적인 수준으로 조정했습니다.
ELECTRON_RADIUS = 0.0005
AIR_PARTICLE_RADIUS = 0.0001
ELECTRON_ACTIVATION_RADIUS = 0.05
COLLISION_EPS = 1e-9
DE_BROGLIE_CALC_EPS = 1e-30
DT = 5e-11
DOMAIN_SIZE = 0.35
PLANCK_CONSTANT = scipy.constants.value('Planck constant')
ELECTRON_MASS_PHYSICAL = scipy.constants.value('electron mass')
ELECTRON_CHARGE_PHYSICAL = scipy.constants.value('elementary charge')
NITROGEN_MOLECULE_MASS_PHYSICAL = 28.014 * 1.66053906660e-27
ELECTRONS_PER_EMISSION = 100

# --- 결정 격자 매개변수 (2D에 맞춰 조정) ---
GRID_X_POSITION = 0.20
GRID_THICKNESS = 0.005
GRID_SIZE_Y = 0.08
GRID_Y_MIN = (DOMAIN_SIZE / 2.0) - (GRID_SIZE_Y / 2.0)
GRID_Y_MAX = (DOMAIN_SIZE / 2.0) + (GRID_SIZE_Y / 2.0)
GUI_LATTICE_CONSTANT_FOR_RENDERING = 0.00008
num_cells_y_gui_for_rendering = int(GRID_SIZE_Y / GUI_LATTICE_CONSTANT_FOR_RENDERING)
MAX_LATTICE_ATOMS_FOR_RENDERING = num_cells_y_gui_for_rendering + 1

# --- 브래그 회절 관련 상수 ---
BRAGG_ORDER = 1
BRAGG_ANGLE_TOLERANCE_RADIANS = math.radians(2.0)

# --- Python에서 d-spacing 계산 및 Taichi로 전달 ---
NM = 10 ** (-9)

# --- GAN 모델 클래스 정의 ---
class Generator(nn.Module):
    def __init__(self, data_dim=DATA_DIM):
        super(Generator, self).__init__()
        self.data_dim = data_dim
        
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, data_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

def load_gan_model(model_path='best_generator_particledata4.pth', scaler_path='scaler_particledata4.pth'):
    """GAN 모델과 스케일러를 로드합니다."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 생성기 모델 로드
    generator = Generator(DATA_DIM).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    # 스케일러 로드
    scaler = torch.load(scaler_path, map_location=device)
    
    print(f"GAN 모델 로드 완료: {model_path}")
    print(f"스케일러 로드 완료: {scaler_path}")
    
    return generator, scaler, device

def generate_air_particles_with_gan(generator, scaler, device, num_particles=1000):
    """GAN을 사용하여 공기 입자 데이터를 생성합니다."""
    with torch.no_grad():
        # 랜덤 노이즈 생성
        z = torch.randn(num_particles, 100).to(device)
        
        # GAN으로 데이터 생성
        generated_data = generator(z)
        
        # 스케일러로 역정규화
        generated_data_np = scaler.inverse_transform(generated_data.cpu().numpy())
        
        return generated_data_np

confs = {
    'Nickel' : {'structure': 'ccp', 'a':0.3524*NM, 'b': 0.3524*NM, 'c' : 0.3524*NM, 'alpha' : 90, 'beta' :90, 'gamma':90},
    'Zinc' : {'structure': 'hcp','a':0.26649*NM, 'b': 0.26659*NM, 'c' : 0.49468*NM, 'alpha' : 90, 'beta':90, 'gamma':120},
    'Silicon' : {'structure': 'diamond', 'a':0.54309*NM, 'b': 0.54309*NM, 'c' : 0.54309*NM, 'alpha' : 90, 'beta':90, 'gamma':90}
}

def calculate_interplanar_distances(conf:dict, max_miller:int):
    a, b, c = symbols('a b c')
    d = symbols('d')
    h, k, l = symbols('h k l')

    cubic_expr = Eq(d**2, 1/((h**2 + k**2 + l**2)/a**2))
    hexagonal_expr = Eq(d**2, 1 / (4/3 * (h**2 + h*k + k**2)/ a**2  + l**2 / c**2))
    
    ds = dict()
    ress = []
    
    hkl_combinations = []
    for _h in range(max_miller + 1):
        for _k in range(max_miller + 1):
            for _l in range(max_miller + 1):
                if _h == 0 and _k == 0 and _l == 0:
                    continue
                hkl_combinations.append((_h, _k, _l))

    unique_hkls = []
    for hkl in hkl_combinations:
        h_val, k_val, l_val = hkl
        
        current_gcd = 0
        non_zero_components = [abs(val) for val in [h_val, k_val, l_val] if val != 0]

        if len(non_zero_components) > 0:
            current_gcd = non_zero_components[0]
            for i in range(1, len(non_zero_components)):
                current_gcd = gcd(current_gcd, non_zero_components[i])
        else:
            continue

        normalized_hkl = (h_val // current_gcd, k_val // current_gcd, l_val // current_gcd)

        if normalized_hkl not in unique_hkls:
            unique_hkls.append(normalized_hkl)
            
    for _h, _k, _l in unique_hkls:
        if conf['structure'] == 'ccp' or conf['structure'] == 'diamond':
            expr = cubic_expr.subs([
                                        (a, conf['a']), (b, conf['b']), (c, conf['c']),
                                        (h, _h), (k, _k), (l, _l)
                                    ])
        elif conf['structure'] == 'hcp':
            expr = hexagonal_expr.subs([
                                        (a, conf['a']), (b, conf['b']), (c, conf['c']),
                                        (h, _h), (k, _k), (l, _l)
                                    ])
        
        res_list = solve(expr, d, check=False)
        current_d_val = 0.0
        for r in res_list:
            if r.is_real and r > 0:
                current_d_val = float(r)
                break
        
        if current_d_val > 0:
            is_duplicate = False
            for existing_d in ress:
                if abs(existing_d - current_d_val) < 1e-12:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                ress.append(current_d_val)
                ds[f'{_h}{_k}{_l}'] = current_d_val
    
    sorted_ds = dict(sorted(ds.items(), key=lambda item: item[1]))
    return sorted_ds

dict_nickel_ds = calculate_interplanar_distances(confs['Nickel'], 3)
print("Calculated d-spacing values for Nickel (nm):")
for hkl, d_val in dict_nickel_ds.items():
    print(f"  ({hkl}): {d_val / NM:.3f} nm")

MAX_D_SPACINGS = len(dict_nickel_ds) + 5
nickel_d_spacings = ti.field(dtype=ti.f32, shape=MAX_D_SPACINGS)
num_nickel_d_spacings = ti.field(dtype=ti.i32, shape=())

def load_d_spacings_to_taichi(d_dict):
    idx = 0
    for d_val in d_dict.values():
        if idx < MAX_D_SPACINGS:
            nickel_d_spacings[idx] = d_val
            idx += 1
    num_nickel_d_spacings[None] = idx
    print(f"Number of d-spacings loaded into Taichi field: {num_nickel_d_spacings[None]}")

# --- Taichi Field Declarations (Existing) ---
electron_pos = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_ELECTRONS)
electron_vel = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_ELECTRONS)
electron_traveled = ti.field(dtype=ti.f32, shape=MAX_ELECTRONS)
electron_is_active = ti.field(dtype=ti.i32, shape=MAX_ELECTRONS)
electron_is_particle_state = ti.field(dtype=ti.i32, shape=MAX_ELECTRONS)
electron_has_scattered_bragg = ti.field(dtype=ti.i32, shape=MAX_ELECTRONS)
electron_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_ELECTRONS)
num_active_electrons = ti.field(dtype=ti.i32, shape=())

electron_exit_angles = ti.field(dtype=ti.f32, shape=MAX_ELECTRONS)
electron_exit_count = ti.field(dtype=ti.i32, shape=())
electron_scattered_count = ti.field(dtype=ti.i32, shape=())

lattice_atom_pos_for_rendering = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_LATTICE_ATOMS_FOR_RENDERING)
num_lattice_atoms_for_rendering = ti.field(dtype=ti.i32, shape=())

# --- 공기 입자 관련 필드 ---
air_particle_pos = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_AIR_PARTICLES)
air_particle_vel = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_AIR_PARTICLES)
air_particle_color = ti.Vector.field(3, dtype=ti.f32, shape=MAX_AIR_PARTICLES)
air_particle_active = ti.field(dtype=ti.i32, shape=MAX_AIR_PARTICLES)
num_current_air_particles = ti.field(dtype=ti.i32, shape=())

NUM_DEBUG_ELECTRONS = 10
selected_electron_indices_ti = ti.field(dtype=ti.i32, shape=NUM_DEBUG_ELECTRONS)
debug_electron_pos = ti.Vector.field(DIM, dtype=ti.f32, shape=NUM_DEBUG_ELECTRONS)
debug_electron_vel = ti.Vector.field(DIM, ti.f32, shape=NUM_DEBUG_ELECTRONS)

@ti.func
def is_inside_lattice_region(p):
    return (p[0] >= GRID_X_POSITION and p[0] <= GRID_X_POSITION + GRID_THICKNESS and
            p[1] >= GRID_Y_MIN and p[1] <= GRID_Y_MAX)

@ti.kernel
def initialize_electrons_initial():
    for i in range(MAX_ELECTRONS):
        electron_pos[i] = ti.Vector([0.0, 0.0])
        electron_vel[i] = ti.Vector([0.0, 0.0])
        electron_traveled[i] = 0.0
        electron_is_active[i] = 0
        electron_is_particle_state[i] = 1
        electron_has_scattered_bragg[i] = 0
        electron_colors[i] = ti.Vector([0.0, 0.0, 0.0])
    num_active_electrons[None] = 0
    electron_exit_count[None] = 0
    electron_scattered_count[None] = 0

@ti.kernel
def update_and_reposition_air_particles(activation_radius: ti.f32, max_air_particles_gui_limit: ti.i32, air_molecules_per_m2_actual: ti.f32):
    """
    각 활성 전자 주위에 공기 입자를 무작위로 재배치하고 활성화합니다.
    GUI 렌더링을 위해 최대 입자 수를 제한합니다.
    """
    active_particles_this_frame = 0
    for i in range(MAX_ELECTRONS):
        if electron_is_active[i] == 0:
            continue
        
        e_pos = electron_pos[i]
        # 활성화 반경 내의 이론적인 공기 분자 수 계산
        activation_area = ti.math.pi * activation_radius * activation_radius
        num_molecules_in_activation_area = activation_area * air_molecules_per_m2_actual
        
        # 각 전자가 생성할 최대 입자 수 (GUI 렌더링 제한 고려)
        # MAX_AIR_PARTICLES를 전체 파티클 수로 사용하고, 각 전자가 일정 비율을 차지하도록 함
        # 이 방식으로 여러 전자의 활성화 영역이 겹치더라도, 전체 MAX_AIR_PARTICLES 내에서 관리됩니다.
        max_particles_per_electron_for_rendering = ti.cast(max_air_particles_gui_limit / num_active_electrons[None] + 1, ti.i32) if num_active_electrons[None] > 0 else max_air_particles_gui_limit
        target_particles_for_this_electron = ti.min(ti.cast(num_molecules_in_activation_area, ti.i32), max_particles_per_electron_for_rendering)
        
        for k in range(target_particles_for_this_electron):
            if active_particles_this_frame >= max_air_particles_gui_limit:
                break

            new_pos_found = False
            rand_pos = ti.Vector([0.0, 0.0])
            attempts = 0
            max_attempts = 100 # 유효한 위치를 찾기 위한 최대 시도 횟수

            while not new_pos_found and attempts < max_attempts:
                angle = ti.random(ti.f32) * 2.0 * ti.math.pi
                distance = ti.random(ti.f32) * activation_radius
                rand_pos = e_pos + ti.Vector([distance * ti.cos(angle), distance * ti.sin(angle)])

                # 시뮬레이션 도메인 내부에 있고, 격자 영역 내부가 아닌지 확인
                if rand_pos[0] >= 0.0 and rand_pos[0] <= DOMAIN_SIZE and \
                   rand_pos[1] >= 0.0 and rand_pos[1] <= DOMAIN_SIZE:
                    if not is_inside_lattice_region(rand_pos):
                        new_pos_found = True
                attempts += 1
            
            if new_pos_found:
                # 활성 공기 입자를 현재 인덱스에 저장 (덮어쓰기 방식)
                idx = active_particles_this_frame
                air_particle_pos[idx] = rand_pos
                air_particle_vel[idx] = ti.Vector([0.0, 0.0]) # 공기 입자는 정지
                air_particle_color[idx] = ti.Vector([0.0, 1.0, 0.0]) # 활성 상태 (녹색)
                air_particle_active[idx] = 1 # 활성으로 표시
                active_particles_this_frame += 1
            else:
                # 유효한 위치를 찾지 못한 경우, 해당 입자는 생성하지 않음 (스킵)
                pass
    
    # 남은 공기 입자 슬롯을 비활성화 (이전 프레임에 활성화된 입자 제거)
    for i in range(active_particles_this_frame, max_air_particles_gui_limit):
        air_particle_active[i] = 0
        air_particle_color[i] = ti.Vector([0.0, 0.0, 0.0])
        air_particle_pos[i] = ti.Vector([-1.0, -1.0]) # 화면 밖으로 이동
        air_particle_vel[i] = ti.Vector([0.0, 0.0])

    num_current_air_particles[None] = active_particles_this_frame

def update_air_particles_with_gan(generator, scaler, device, activation_radius, max_air_particles_gui_limit):
    """GAN 모델을 사용하여 공기 입자를 생성하고 배치합니다."""
    # GAN으로 공기 입자 데이터 생성
    gan_particles = generate_air_particles_with_gan(generator, scaler, device, max_air_particles_gui_limit)
    
    # 생성된 데이터를 Taichi 필드에 복사
    active_particles_this_frame = 0
    
    for i in range(min(len(gan_particles), max_air_particles_gui_limit)):
        # GAN 생성 데이터에서 위치 정보 추출 (첫 3개 컬럼이 위치)
        gan_pos = gan_particles[i][:3]  # x, y, z 위치
        
        # 2D 시뮬레이션에 맞게 조정 (z는 무시)
        pos_2d = np.array([gan_pos[0], gan_pos[1]])
        
        # 시뮬레이션 도메인 내부로 정규화
        pos_2d = np.clip(pos_2d, 0.0, DOMAIN_SIZE)
        
        # 격자 영역 내부가 아닌지 확인
        if not (GRID_X_POSITION <= pos_2d[0] <= GRID_X_POSITION + GRID_THICKNESS and 
                GRID_Y_MIN <= pos_2d[1] <= GRID_Y_MAX):
            
            # Taichi 필드에 저장
            air_particle_pos.from_numpy(np.array([pos_2d]))
            air_particle_vel.from_numpy(np.array([[0.0, 0.0]]))
            air_particle_color.from_numpy(np.array([[0.0, 1.0, 0.0]]))  # 녹색
            air_particle_active.from_numpy(np.array([1]))
            
            active_particles_this_frame += 1
    
    # 남은 슬롯 비활성화
    for i in range(active_particles_this_frame, max_air_particles_gui_limit):
        air_particle_active.from_numpy(np.array([0]))
        air_particle_color.from_numpy(np.array([[0.0, 0.0, 0.0]]))
        air_particle_pos.from_numpy(np.array([[-1.0, -1.0]]))
        air_particle_vel.from_numpy(np.array([[0.0, 0.0]]))
    
    num_current_air_particles.from_numpy(np.array([active_particles_this_frame]))


@ti.kernel
def initialize_lattice_atoms_for_rendering():
    """Initializes static lattice points for rendering the crystal structure."""
    current_atom_idx = 0
    x_coord = GRID_X_POSITION
    for y_idx in range(num_cells_y_gui_for_rendering + 1):
        if current_atom_idx < MAX_LATTICE_ATOMS_FOR_RENDERING:
            y_coord = GRID_Y_MIN + y_idx * GUI_LATTICE_CONSTANT_FOR_RENDERING
            if y_coord >= GRID_Y_MIN and y_coord >= 0.0 and y_coord <= DOMAIN_SIZE:
                lattice_atom_pos_for_rendering[current_atom_idx] = ti.Vector([x_coord, y_coord])
                current_atom_idx += 1

    num_lattice_atoms_for_rendering[None] = current_atom_idx
    print(f"Total GUI rendering lattice atoms placed: {num_lattice_atoms_for_rendering[None]}")

@ti.kernel
def emit_new_electrons_kernel(start_speed: ti.f32, incident_angle_radians: ti.f32):
    """Emits new electrons from a defined point with a given speed and angle."""
    emitted_this_call = 0
    for i in range(MAX_ELECTRONS):
        if num_active_electrons[None] >= MAX_ELECTRONS:
            continue

        if emitted_this_call < ELECTRONS_PER_EMISSION and electron_is_active[i] == 0:
            emission_x_start = 0.0

            slit_center_y = (GRID_Y_MIN + GRID_Y_MAX) / 2.0
            slit_width_y = 0.01 

            y_offset = (ti.random(ti.f32) - 0.5) * slit_width_y

            electron_pos[i] = ti.Vector([emission_x_start, slit_center_y + y_offset])

            electron_vel[i] = ti.Vector([
                start_speed * ti.cos(incident_angle_radians),
                start_speed * ti.sin(incident_angle_radians)
            ])

            electron_traveled[i] = 0.0
            electron_is_active[i] = 1
            electron_is_particle_state[i] = 1
            electron_has_scattered_bragg[i] = 0 
            electron_colors[i] = ti.Vector([1.0, 0.0, 0.0]) 
            ti.atomic_add(num_active_electrons[None], 1)
            emitted_this_call += 1

@ti.func
def calculate_de_broglie_wavelength_ti(velocity_magnitude: ti.f32) -> ti.f32:
    """Calculates de Broglie wavelength for a given electron velocity."""
    momentum = ELECTRON_MASS_PHYSICAL * velocity_magnitude
    wavelength_result = 0.0
    if momentum >= DE_BROGLIE_CALC_EPS:
        wavelength_result = PLANCK_CONSTANT / momentum
    return wavelength_result

def calculate_de_broglie_wavelength_python(velocity_magnitude: float) -> float:
    """Python version of de Broglie wavelength calculation."""
    momentum = ELECTRON_MASS_PHYSICAL * velocity_magnitude
    if momentum < DE_BROGLIE_CALC_EPS:
        return 0.0
    return PLANCK_CONSTANT / momentum

@ti.func
def calculate_bragg_scatter_direction_for_polycrystal(
    incident_vel: ti.template(),
    current_speed: ti.f32,
    tolerance: ti.f32
) -> ti.template():
    """
    Calculates the Bragg scattered direction for an electron interacting with a polycrystalline material.
    Assumes random lattice orientations.
    """
    new_velocity = incident_vel 

    wavelength = calculate_de_broglie_wavelength_ti(current_speed)

    should_attempt_diffraction = 1 

    if wavelength <= DE_BROGLIE_CALC_EPS:
        should_attempt_diffraction = 0

    if num_nickel_d_spacings[None] == 0: 
        should_attempt_diffraction = 0
    
    if should_attempt_diffraction == 1:
        random_d_idx = ti.cast(ti.random(ti.f32) * num_nickel_d_spacings[None], ti.i32)
        current_d_spacing = nickel_d_spacings[random_d_idx]

        random_phi = ti.random(ti.f32) * 2.0 * ti.math.pi
        random_lattice_normal = ti.Vector([ti.cos(random_phi), ti.sin(random_phi)]).normalized()
        
        incident_direction = incident_vel.normalized()
        
        # Incident angle for Bragg law (angle between incident ray and lattice plane)
        # For a 2D simulation, this is the angle between the velocity vector and the lattice normal
        # The true Bragg angle theta is typically defined between the incident ray and the plane.
        # Here theta_prime is between incident_direction and lattice_normal (phi).
        # We need the angle between the incident ray and the plane. If the normal is perpendicular to the plane,
        # then the angle between incident ray and plane is pi/2 - theta_prime.
        theta_prime = ti.acos(ti.abs(incident_direction.dot(random_lattice_normal)))
        bragg_theta_incident = ti.abs(ti.math.pi / 2.0 - theta_prime) 

        required_sin_theta_formula = (BRAGG_ORDER * wavelength) / (2.0 * current_d_spacing)

        # Robustness: Clamp input to asin to prevent domain errors
        if required_sin_theta_formula >= -1.0 + COLLISION_EPS and required_sin_theta_formula <= 1.0 + COLLISION_EPS: 
            bragg_theta_from_formula = ti.asin(ti.max(-1.0, ti.min(1.0, required_sin_theta_formula))) 

            if ti.abs(bragg_theta_incident - bragg_theta_from_formula) < tolerance:
                # Reflection off the lattice plane (normal is random_lattice_normal)
                dot_prod_in_normal = incident_direction.dot(random_lattice_normal)
                scattered_direction = incident_direction - 2 * dot_prod_in_normal * random_lattice_normal
                
                new_velocity = scattered_direction * current_speed # Keep the same speed after Bragg scattering
    
    return new_velocity

@ti.kernel
def update_electrons(dt: ti.f32, initial_electron_speed_in_kernel: ti.f32):
    """
    Updates electron positions and velocities, handling boundaries,
    Bragg scattering, and (simplified) air particle collisions.
    """
    for i in range(MAX_ELECTRONS):
        if electron_is_active[i] == 0:
            continue

        prev_pos = electron_pos[i]
        current_vel = electron_vel[i]
        current_speed = current_vel.norm()

        # Update position
        next_pos = electron_pos[i] + current_vel * dt
        electron_traveled[i] += current_vel.norm() * dt

        is_electron_exited_x = 0
        should_stop_electron = 0

        # Boundary checks
        # If electron exits on X-axis (right side), mark as exited and stop.
        # If it hits top/bottom or left wall, reflect.
        for d_idx in ti.static(range(DIM)):
            if d_idx == 0 and next_pos[d_idx] > DOMAIN_SIZE: # Exit on right
                is_electron_exited_x = 1
                electron_pos[i][d_idx] = DOMAIN_SIZE
                should_stop_electron = 1
            elif not is_electron_exited_x: # Reflection for other boundaries
                if next_pos[d_idx] < 0.0:
                    electron_pos[i][d_idx] = 0.0 + COLLISION_EPS
                    electron_vel[i][d_idx] *= -1.0 # Reflect velocity component
                elif next_pos[d_idx] > DOMAIN_SIZE:
                    electron_pos[i][d_idx] = DOMAIN_SIZE - COLLISION_EPS
                    electron_vel[i][d_idx] *= -1.0 # Reflect velocity component
        
        if should_stop_electron == 1:
            electron_vel[i] = ti.Vector([0.0, 0.0]) # Stop completely
            if is_electron_exited_x == 1:
                electron_is_active[i] = 0
                ti.atomic_sub(num_active_electrons[None], 1)
                electron_colors[i] = ti.Vector([0.5, 0.5, 0.5]) # Gray for exited
            continue # Skip further updates for this electron

        # --- Electron interaction with Lattice Region ---
        # State change: Particle State (moving freely) -> Wave State (inside lattice)
        if is_inside_lattice_region(next_pos) and electron_is_particle_state[i] == 1:
            electron_is_particle_state[i] = 0 # Enter wave state
            electron_colors[i] = ti.Vector([0.5, 0.0, 0.5]) # Purple for wave state

        # If in wave state (inside lattice)
        if electron_is_particle_state[i] == 0:
            is_scattered_this_frame = 0

            # Attempt Bragg diffraction
            diffraction_attempt_probability_per_frame = 0.01 
            if ti.random(ti.f32) < diffraction_attempt_probability_per_frame:
                new_scatter_vel = calculate_bragg_scatter_direction_for_polycrystal(
                    current_vel,
                    current_speed,
                    BRAGG_ANGLE_TOLERANCE_RADIANS
                )

                if (new_scatter_vel - current_vel).norm() > COLLISION_EPS: # Check if actual scattering occurred
                    electron_vel[i] = new_scatter_vel
                    is_scattered_this_frame = 1
                    if electron_has_scattered_bragg[i] == 0:
                        electron_has_scattered_bragg[i] = 1
                        ti.atomic_add(electron_scattered_count[None], 1)
            
            # Removed random deviation inside lattice to promote straight movement
            # if is_scattered_this_frame == 0:
            #     random_deviation_factor = 0.005
            #     random_vec = ti.Vector([ti.random(ti.f32) * 2 - 1, ti.random(ti.f32) * 2 - 1])
            #     random_vec_norm = random_vec.norm()
            #     if random_vec_norm > COLLISION_EPS:
            #         new_vel = current_vel + random_vec / random_vec_norm * random_deviation_factor * current_speed
            #         electron_vel[i] = new_vel.normalized() * current_speed


        # State change: Wave State (inside lattice) -> Particle State (exiting lattice)
        if electron_is_particle_state[i] == 0 and not is_inside_lattice_region(next_pos):
            electron_is_particle_state[i] = 1 # Exit wave state, return to particle state
            if electron_has_scattered_bragg[i] == 1:
                electron_colors[i] = ti.Vector([1.0, 0.0, 1.0]) # Magenta for Bragg scattered
            else:
                electron_colors[i] = ti.Vector([0.0, 0.0, 1.0]) # Blue for unscattered (passed through)

            # Record exit angle only when transitioning back to particle state
            if current_speed > COLLISION_EPS:
                angle_rad = ti.atan2(electron_vel[i][1], electron_vel[i][0])
                
                # --- START ANGLE CORRECTION FOR -X AXIS ---
                # 각도를 -X 축을 기준으로 계산 (0 ~ 2*pi)
                # +X 축 기준 각도 (angle_rad)를 -X 축 기준으로 변환
                # -X 축은 +pi (180도)에 해당.
                # 반시계 방향으로 각도가 증가한다고 가정
                
                # atan2(y, x)는 (-pi, pi] 범위의 값을 반환합니다.
                # 이를 [0, 2*pi) 범위로 정규화합니다.
                if angle_rad < 0.0:
                    angle_rad += 2 * ti.math.pi
                
                # 이제 angle_rad는 [0, 2*pi) 범위의 +X축 기준 각도입니다.
                # -X축을 기준으로 하고 싶다면, +X축 기준 각도에서 pi (180도)를 빼거나 더하면 됩니다.
                # 예를 들어, +X축 기준 0도는 -X축 기준 pi (180도).
                # +X축 기준 pi/2 (90도)는 -X축 기준 -pi/2 (270도 또는 -90도).
                # 플롯의 X축 범위를 0-360도로 유지하는 것이 일반적이므로,
                # 다음과 같이 조정하여 -X축을 0으로 하고 반시계 방향으로 증가하도록 합니다.

                # (angle_rad - ti.math.pi)는 -X축을 0으로 하는 각도. 범위는 [-pi, pi)
                # 이를 [0, 2*pi)로 다시 정규화
                adjusted_angle_rad = angle_rad - ti.math.pi
                if adjusted_angle_rad < 0.0:
                    adjusted_angle_rad += 2 * ti.math.pi
                
                # --- END ANGLE CORRECTION ---

                idx = ti.atomic_add(electron_exit_count[None], 1)
                if idx < MAX_ELECTRONS: # Prevent out-of-bounds access
                    electron_exit_angles[idx] = adjusted_angle_rad

        # --- Enhanced Electron interaction with Air Particles (with direction change) ---
        # Both speed reduction and direction change due to air collisions
        if electron_is_particle_state[i] == 1:
            for j in range(num_current_air_particles[None]):
                if air_particle_active[j] == 0:
                    continue 

                air_pos = air_particle_pos[j]
                
                dist_vec = electron_pos[i] - air_pos
                distance = dist_vec.norm()

                min_distance_for_collision = (ELECTRON_RADIUS + AIR_PARTICLE_RADIUS) * DOMAIN_SIZE

                if distance < min_distance_for_collision:
                    # Enhanced collision: both speed reduction and direction change
                    energy_loss_factor = 0.995 # Small energy loss per collision
                    
                    # Calculate collision normal (from air particle to electron)
                    collision_normal = dist_vec.normalized()
                    
                    # Calculate reflection direction using elastic collision
                    incident_vel = current_vel
                    incident_normal = collision_normal
                    
                    # Elastic collision reflection formula: v' = v - 2(v·n)n
                    # where v is incident velocity, n is normal vector
                    dot_product = incident_vel.dot(incident_normal)
                    reflected_vel = incident_vel - 2.0 * dot_product * incident_normal
                    
                    # Apply energy loss to reflected velocity
                    new_speed = current_speed * energy_loss_factor
                    electron_vel[i] = reflected_vel.normalized() * new_speed

                    # Ensure minimal speed after collision, prevent it from stopping completely
                    if electron_vel[i].norm() < initial_electron_speed_in_kernel * 0.005: 
                        electron_vel[i] = reflected_vel.normalized() * initial_electron_speed_in_kernel * 0.005
                    
                    # Repulsion to avoid sticking (simple overlap correction)
                    overlap = min_distance_for_collision - distance
                    if overlap > 0:
                        correction_vec = overlap * dist_vec.normalized() * 0.5 # Move electron away from air particle
                        electron_pos[i] += correction_vec
                        # air_particle_pos[j] -= correction_vec # No need to move air particle for this simplified model

        electron_pos[i] = next_pos # Final position update for this frame


@ti.kernel
def select_and_debug_electrons_kernel(indices: ti.template()):
    """
    선택된 전자의 위치와 속도 정보를 디버그 필드로 복사하는 Taichi 커널.
    indices는 선택된 전자의 인덱스를 담고 있는 ti.field입니다.
    """
    for i in range(NUM_DEBUG_ELECTRONS):
        electron_idx = indices[i]
        debug_electron_pos[i] = electron_pos[electron_idx]
        debug_electron_vel[i] = electron_vel[electron_idx]

def main():
    load_d_spacings_to_taichi(dict_nickel_ds)

    # GAN 모델 로드
    try:
        generator, scaler, device = load_gan_model()
        print("GAN 모델을 사용하여 공기 입자를 생성합니다.")
        use_gan_model = True
    except Exception as e:
        print(f"GAN 모델 로드 실패: {e}")
        print("기존 랜덤 방식으로 공기 입자를 생성합니다.")
        use_gan_model = False

    acceleration_voltage = 3600.0
    initial_electron_speed = ti.sqrt(2 * ELECTRON_CHARGE_PHYSICAL * acceleration_voltage / ELECTRON_MASS_PHYSICAL)
    print(f"Initial electron speed: {initial_electron_speed:.2e} m/s (accelerated by {acceleration_voltage}V)")
    initial_de_broglie_wavelength = calculate_de_broglie_wavelength_python(initial_electron_speed)
    print(f"Initial de Broglie wavelength: {initial_de_broglie_wavelength:.2e} m")

    electron_incident_angle_degrees = 0.0
    electron_incident_angle_radians_for_kernel = math.radians(electron_incident_angle_degrees)
    print(f"Electron emission angle (relative to X-axis, deg): {electron_incident_angle_degrees:.2f}")
    print(f"Bragg angle tolerance (deg): {math.degrees(BRAGG_ANGLE_TOLERANCE_RADIANS):.2f}")

    PRESSURE_ATM = 0.19
    PRESSURE_PA = PRESSURE_ATM * 101325.0 
    TEMPERATURE_K = 298.15 
    GAS_CONSTANT_R = 8.314 
    AVOGADRO_NUMBER = 6.022e23 

    moles_of_air_per_m3 = PRESSURE_PA / (GAS_CONSTANT_R * TEMPERATURE_K)
    molecules_of_air_per_m3 = moles_of_air_per_m3 * AVOGADRO_NUMBER
    
    air_particles_per_m2_actual = molecules_of_air_per_m3
    
    print(f"\n--- Air Particle Calculation ---")
    print(f"Air pressure: {PRESSURE_ATM} atm, Temperature: {TEMPERATURE_K} K")
    print(f"Molecules of air per cubic meter: {molecules_of_air_per_m3:.2e}")
    print(f"Actual air particles per square meter (assuming 1m depth): {air_particles_per_m2_actual:.2e}")
    print(f"Note: Due to extreme density, only a representative subset of particles within the activation radius will be rendered (MAX_AIR_PARTICLES: {MAX_AIR_PARTICLES}).")

    initialize_electrons_initial()
    initialize_lattice_atoms_for_rendering() 

    gui = ti.GUI("Davidson-Germer Experiment Simulation (Polycrystalline Bragg Diffraction)", res=(800, 600), background_color=0x112F41)
    frame = 0
    total_sim_time = 0.0
    total_electrons_emitted = 0 

    max_frames_to_emit = 100

    py_selected_electron_indices = random.sample(range(MAX_ELECTRONS), NUM_DEBUG_ELECTRONS)
    print(f"\n디버그를 위해 선택된 전자 ID (Python 리스트): {py_selected_electron_indices}")

    for i in range(NUM_DEBUG_ELECTRONS):
        selected_electron_indices_ti[i] = py_selected_electron_indices[i]

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False

        if frame < max_frames_to_emit:
            emit_new_electrons_kernel(initial_electron_speed, electron_incident_angle_radians_for_kernel)
            total_electrons_emitted += ELECTRONS_PER_EMISSION

        update_electrons(DT, initial_electron_speed) 
        # 공기 입자 위치는 매 프레임마다 전자 주변에 새로 배치되므로, 이 함수를 호출하여 활성 공기 입자만 렌더링하도록 합니다.
        if use_gan_model:
            update_air_particles_with_gan(generator, scaler, device, ELECTRON_ACTIVATION_RADIUS, MAX_AIR_PARTICLES)
        else:
            update_and_reposition_air_particles(ELECTRON_ACTIVATION_RADIUS, MAX_AIR_PARTICLES, air_particles_per_m2_actual) 

        total_sim_time += DT

        select_and_debug_electrons_kernel(selected_electron_indices_ti) 
        
        debug_pos_np = debug_electron_pos.to_numpy()
        debug_vel_np = debug_electron_vel.to_numpy()

        # Debugging output for selected electrons (print less frequently to avoid flooding console)
        if frame % 100 == 0: # Print every 100 frames
            print(f"\n--- 현재 시간: {total_sim_time:.2e}s (프레임: {frame}) ---")
            for i in range(NUM_DEBUG_ELECTRONS):
                electron_id = py_selected_electron_indices[i] 
                pos = debug_pos_np[i]
                vel = debug_vel_np[i]
                print(f"  전자 {electron_id}:")
                print(f"    위치 (x, y): ({pos[0]:.6f}, {pos[1]:.6f}) m")
                print(f"    속도 (vx, vy): ({vel[0]:.6e}, {vel[1]:.6e}) m/s")
            print("--------------------------------------------------")

        if frame % 2 == 0:  # 2프레임마다 출력
            print(f"\n--- 현재 프레임 {frame}의 활성 대기 보간입자(녹색) 개수: {num_current_air_particles[None]} ---")

        if frame > max_frames_to_emit + 1000 and num_active_electrons[None] == 0:
            print("\nAll electrons have exited the simulation domain. Terminating simulation.")
            break
        
        if frame > 5000: 
            print("\nMaximum simulation frames reached. Force terminating simulation.")
            break

        # --- GUI Rendering ---
        gui.clear(0x112F41)

        scale_factor = 1.0 / DOMAIN_SIZE

        rect_color = 0x555555 
        line_width = 1

        x0_scaled = GRID_X_POSITION * scale_factor
        x1_scaled = (GRID_X_POSITION + GRID_THICKNESS) * scale_factor
        y0_scaled = GRID_Y_MIN * scale_factor
        y1_scaled = GRID_Y_MAX * scale_factor

        gui.line(begin=np.array([x0_scaled, y0_scaled]), end=np.array([x1_scaled, y0_scaled]), color=rect_color, radius=line_width) 
        gui.line(begin=np.array([x0_scaled, y1_scaled]), end=np.array([x1_scaled, y1_scaled]), color=rect_color, radius=line_width) 
        gui.line(begin=np.array([x0_scaled, y0_scaled]), end=np.array([x0_scaled, y1_scaled]), color=rect_color, radius=line_width) 
        gui.line(begin=np.array([x1_scaled, y0_scaled]), end=np.array([x1_scaled, y1_scaled]), color=rect_color, radius=line_width) 
        
        current_lattice_atom_count = num_lattice_atoms_for_rendering[None]
        if current_lattice_atom_count > 0:
            # CORRECTED LINE: Use num_lattice_atoms_for_rendering[None] instead of current_atom_idx
            lattice_pos_np = lattice_atom_pos_for_rendering.to_numpy()[:current_lattice_atom_count] 
            gui.circles(lattice_pos_np * scale_factor, radius=1, color=0xAAAAAA)

        total_air_particle_count = num_current_air_particles[None]
        if total_air_particle_count > 0:
            render_air_positions_np = air_particle_pos.to_numpy()[:total_air_particle_count]
            render_air_colors_np = air_particle_color.to_numpy()[:total_air_particle_count]

            converted_air_colors_np = (render_air_colors_np * 255).astype(np.uint32)
            packed_air_colors = (converted_air_colors_np[:, 0] << 16) | \
                                (converted_air_colors_np[:, 1] << 8) | \
                                converted_air_colors_np[:, 2]
            
            gui.circles(render_air_positions_np * scale_factor,
                        radius=AIR_PARTICLE_RADIUS / DOMAIN_SIZE * 800 * 2,
                        color=packed_air_colors)

        render_electron_positions = []
        render_electron_colors = []

        for i in range(total_electrons_emitted):
            if i < MAX_ELECTRONS:
                render_electron_positions.append(electron_pos[i].to_numpy())
                if electron_is_active[i] == 1:
                    if electron_is_particle_state[i] == 0:
                        render_electron_colors.append(np.array([0.5, 0.0, 0.5]))
                    else:
                        render_electron_colors.append(electron_colors[i].to_numpy())
                else:
                    render_electron_colors.append(electron_colors[i].to_numpy())

        if len(render_electron_positions) > 0:
            render_electron_positions_np = np.array(render_electron_positions)
            render_electron_colors_np = np.array(render_electron_colors)

            converted_electron_colors_np = (render_electron_colors_np * 255).astype(np.uint32)
            packed_electron_colors = (converted_electron_colors_np[:, 0] << 16) | \
                                     (converted_electron_colors_np[:, 1] << 8) | \
                                     converted_electron_colors_np[:, 2]

            gui.circles(render_electron_positions_np * scale_factor,
                        radius=2,
                        color=packed_electron_colors)

        emission_x_gui = 0.0 * scale_factor
        slit_center_y_gui_display = ((GRID_Y_MIN + GRID_Y_MAX) / 2.0) * scale_factor
        gui.circles(np.array([[emission_x_gui, slit_center_y_gui_display]]), radius=7, color=0xFFFFFF)

        gui.text(f"Active Electrons: {num_active_electrons[None]}", pos=(0.05, 0.95), color=0xFFFFFF)
        gui.text(f"Total Emitted: {total_electrons_emitted}", pos=(0.05, 0.90), color=0xFFFFFF)
        gui.text(f"Bragg Scattered: {electron_scattered_count[None]}", pos=(0.05, 0.85), color=0xFFFFFF)
        
        gui.show()
        frame += 1

    # --- Matplotlib 플롯 추가 ---
    # 시뮬레이션 종료 후 전자 출구 각도 데이터 가져오기
    exit_angles_np = electron_exit_angles.to_numpy()[:electron_exit_count[None]]

    if len(exit_angles_np) > 0:
        # 라디안을 도로 변환
        exit_angles_degrees = np.degrees(exit_angles_np)

        plt.figure(figsize=(10, 6))
        
        # 히스토그램 그리기
        # `bins`는 각도 범위, `counts`는 각 빈의 전자 개수
        # `_`는 patches (막대 그래프 객체)인데 여기서는 사용하지 않음
        counts, bins, _ = plt.hist(exit_angles_degrees, bins=90, range=(0, 360), color='skyblue', edgecolor='black', alpha=0.7)
        
        # y축 중간값 계산 및 설정
        # 히스토그램의 y축 데이터(counts)에서 0이 아닌 값들만 고려
        non_zero_counts = counts[counts > 0]
        if len(non_zero_counts) > 0:
            y_median_value = np.median(non_zero_counts)
            # y축 범위를 [0, 중앙값 * 2]로 설정하여 중앙값이 y축의 중간에 오도록 함
            # 이렇게 하면 y축의 최대값이 데이터의 최대값보다 작아서 데이터가 잘릴 위험이 있습니다.
            # 이 방식은 "중간값이 y축 중간에 오도록" 하는 시각적인 요청에 맞추는 것이지만,
            # 데이터의 전체 분포를 제대로 보여주지 못할 수 있음을 인지해야 합니다.
            # 더 안전한 방법은 `plt.axhline(y_median_value, color='red', linestyle='--', label=f'Median: {int(y_median_value)}')`와 같이
            # 중앙값을 선으로 표시하는 것입니다.
            
            # 사용자 요청에 따라 y축 스케일 조정 (데이터 손실 가능성 있음!)
            plt.ylim(0, y_median_value * 2) 
            plt.axhline(y_median_value, color='red', linestyle='--', label=f'Median Count: {int(y_median_value)}')
            print(f"Median y-axis count: {y_median_value}")
        else:
            print("No non-zero counts in histogram bins.")


        plt.title('Distribution of Electron Exit Angles (Y-axis Scaled by Median Count)')
        plt.xlabel('Exit Angle (Degrees)')
        plt.ylabel('Number of Electrons')
        
        # x축 눈금을 30도마다 설정
        plt.xticks(np.arange(0, 361, 30))
        
        plt.grid(True, alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No electrons exited the simulation domain with recorded angles to plot.")

if __name__ == '__main__':

    main()
