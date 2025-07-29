import taichi as ti
import numpy as np
import math
import matplotlib
import scipy.constants
from sympy import symbols, Eq, solve
from math import gcd
import random

# Matplotlib 백엔드 강제 설정 (Matplotlib 창이 뜨도록)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Taichi 아키텍처 설정
ti.init(arch=ti.gpu)

# --- 시뮬레이션 상수 및 매개변수 ---
DIM = 2 # 2차원 공간
MAX_ELECTRONS = 50000 # 시뮬레이션할 최대 전자 수 (충분히 많게)

ELECTRON_RADIUS = 0.0005 # 전자 시각화 반지름 (GUI 스케일)

COLLISION_EPS = 1e-9 # 부동 소수점 비교를 위한 작은 오차 값
DE_BROGLIE_CALC_EPS = 1e-30 # 드브로이 파장 계산을 위한 아주 작은 값 (운동량 0 방지)

DT = 5e-11 # 시뮬레이션 시간 스텝 (초)

DOMAIN_SIZE = 0.35 # 시뮬레이션 도메인의 크기 (미터 단위, GUI 스케일 기준)

PLANCK_CONSTANT = scipy.constants.value('Planck constant') # 플랑크 상수 (Joule-second)
ELECTRON_MASS_PHYSICAL = scipy.constants.value('electron mass') # 전자의 질량 (kg)
ELECTRON_CHARGE_PHYSICAL = scipy.constants.value('elementary charge') # 전자의 전하량 (Coulomb)

ELECTRONS_PER_EMISSION = 50000 # 매번 방출될 때 방출되는 전자 수 (한 번에 모든 전자를 방출하도록 MAX_ELECTRONS와 동일하게 설정)
EMISSION_INTERVAL_FRAMES = 1 # 전자가 방출되는 프레임 간격 (첫 프레임에만 방출)

# --- 결정 격자 매개변수 (2D에 맞춰 조정) ---
GRID_X_POSITION = 0.20 # 결정 영역의 X축 시작 위치 (미터 단위)
GRID_THICKNESS = 0.005 # 결정 영역의 X축 두께 (실제 물리적 두께는 유지)

GRID_SIZE_Y = 0.08 # 시료의 Y축 크기
GRID_Y_MIN = (DOMAIN_SIZE / 2.0) - (GRID_SIZE_Y / 2.0) # Y축 최소 위치
GRID_Y_MAX = (DOMAIN_SIZE / 2.0) + (GRID_SIZE_Y / 2.0) # Y축 최대 위치

# GUI 렌더링용 원자 간격 (겉면에만 표현)
GUI_LATTICE_CONSTANT_FOR_RENDERING = 0.00008

# 격자 원자의 최대 개수를 미리 계산 (메모리 할당용)
num_cells_y_gui_for_rendering = int(GRID_SIZE_Y / GUI_LATTICE_CONSTANT_FOR_RENDERING)
MAX_LATTICE_ATOMS_FOR_RENDERING = num_cells_y_gui_for_rendering + 1

# --- 브래그 회절 관련 상수 ---
BRAGG_ORDER = 1 # 1차 회절 (이 코드는 n=1만 가정, 여러 n을 지원하려면 수정 필요)
BRAGG_ANGLE_TOLERANCE_RADIANS = math.radians(2.0) # 허용 오차 (2도)

# --- 전자 투과율 관련 상수 (새로 추가) ---
# 이 값은 0.0에서 1.0 사이여야 합니다. 1.0이면 투과율이 100% (감소 없음), 0.0이면 0% (모두 제거)
# 이 값을 조정하여 전자빔이 시료를 통과하면서 감소하는 정도를 시뮬레이션할 수 있습니다.
# 3600V 가속 전자에 대한 감쇠계수 α = 3.86×10⁶ m⁻¹을 바탕으로 계산된 투과율
# T = e^(-αd)에서 d = DT (시간 간격)일 때의 투과율
TRANSMITTANCE_PROBABILITY = 0.999807 # 3600V 감쇠계수 기반 투과율 (99.981%)

# --- Python에서 d-spacing 계산 및 Taichi로 전달 ---
NM = 10 ** (-9) # 나노미터 상수

confs = {
    'Nickel' : {'structure': 'ccp', 'a':0.3524*NM, 'b': 0.3524*NM, 'c' : 0.3524*NM, 'alpha' : 90, 'beta':90, 'gamma':90},
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
    ress = [] # d 값을 저장하여 중복을 피하기 위함
    
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
        # 0이 아닌 성분들만 GCD 계산에 포함
        non_zero_components = [abs(val) for val in [h_val, k_val, l_val] if val != 0]

        if len(non_zero_components) > 0:
            # 첫 번째 non-zero 값을 시작 GCD로 설정
            current_gcd = non_zero_components[0]
            # 나머지 non-zero 값들에 대해 GCD 계산
            for i in range(1, len(non_zero_components)):
                current_gcd = gcd(current_gcd, non_zero_components[i])
        else: # 모든 성분이 0인 경우는 이미 제외됨, 하지만 안전을 위해
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

# Calculate d-spacings for Nickel using the corrected function
dict_nickel_ds = calculate_interplanar_distances(confs['Nickel'], 3)
print("Calculated d-spacing values for Nickel (nm):")
for hkl, d_val in dict_nickel_ds.items():
    print(f"  ({hkl}): {d_val / NM:.3f} nm")

# Taichi field declaration (to store d-spacing values calculated in Python)
MAX_D_SPACINGS = len(dict_nickel_ds) + 5 # Allocate slightly more space than strictly needed
nickel_d_spacings = ti.field(dtype=ti.f32, shape=MAX_D_SPACINGS)
num_nickel_d_spacings = ti.field(dtype=ti.i32, shape=())

# Function to load Python d-spacing data into Taichi fields
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
electron_is_particle_state = ti.field(dtype=ti.i32, shape=MAX_ELECTRONS) # 1: particle, 0: wave
electron_has_scattered_bragg = ti.field(dtype=ti.i32, shape=MAX_ELECTRONS) # Has this electron experienced Bragg diffraction? (1: Yes, 0: No)
electron_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_ELECTRONS)
num_active_electrons = ti.field(dtype=ti.i32, shape=())

electron_exit_angles = ti.field(dtype=ti.f32, shape=MAX_ELECTRONS) # Radians
electron_exit_count = ti.field(dtype=ti.i32, shape=())
electron_scattered_count = ti.field(dtype=ti.i32, shape=()) # Counter for total Bragg-scattered electrons

lattice_atom_pos_for_rendering = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_LATTICE_ATOMS_FOR_RENDERING)
num_lattice_atoms_for_rendering = ti.field(dtype=ti.i32, shape=())

# --- Lattice-related functions (adapted for 2D) ---
@ti.func
def is_inside_lattice_region(p):
    """Checks if a given coordinate p is within the region where lattice atoms are placed (2D)."""
    return (p[0] >= GRID_X_POSITION and p[0] <= GRID_X_POSITION + GRID_THICKNESS and
            p[1] >= GRID_Y_MIN and p[1] <= GRID_Y_MAX)

# --- Initialization Kernels ---
@ti.kernel
def initialize_electrons_initial():
    """Resets all electrons to their initial state (inactive, initial position etc.)."""
    for i in range(MAX_ELECTRONS):
        electron_pos[i] = ti.Vector([0.0, 0.0])
        electron_vel[i] = ti.Vector([0.0, 0.0])
        electron_traveled[i] = 0.0
        electron_is_active[i] = 0
        electron_is_particle_state[i] = 1 # Initially in particle state
        electron_has_scattered_bragg[i] = 0 # Bragg scattering flag initialized to No
        electron_colors[i] = ti.Vector([0.0, 0.0, 0.0]) # Initial color (black)
    num_active_electrons[None] = 0
    electron_exit_count[None] = 0
    electron_scattered_count[None] = 0 # Reset scattered electron counter

@ti.kernel
def initialize_lattice_atoms_for_rendering():
    """
    Kernel to place atoms only on the visible surface of the crystal lattice for GUI rendering (2D).
    Actual physical interaction is determined by 'is_inside_lattice_region'.
    """
    current_atom_idx = 0
    # Place atoms only on the leftmost surface (X_POSITION) of the sample
    x_coord = GRID_X_POSITION
    for y_idx in range(num_cells_y_gui_for_rendering + 1):
        if current_atom_idx < MAX_LATTICE_ATOMS_FOR_RENDERING:
            y_coord = GRID_Y_MIN + y_idx * GUI_LATTICE_CONSTANT_FOR_RENDERING
            if y_coord >= GRID_Y_MIN and y_coord <= GRID_Y_MAX:
                lattice_atom_pos_for_rendering[current_atom_idx] = ti.Vector([x_coord, y_coord])
                current_atom_idx += 1

    num_lattice_atoms_for_rendering[None] = current_atom_idx
    print(f"Total GUI rendering lattice atoms placed: {num_lattice_atoms_for_rendering[None]}")


@ti.kernel
def emit_new_electrons_kernel(start_speed: ti.f32, incident_angle_radians: ti.f32, num_to_emit: ti.i32):
    """Kernel to emit new electrons at a specified speed and angle (adapted for 2D)."""
    emitted_this_call = 0 
    for i in range(MAX_ELECTRONS): 
        # 이 조건을 만족할 때만 전자 방출 로직을 실행합니다.
        # 즉, 목표 전자 수를 채웠거나, 활성 전자 슬롯이 더 이상 없으면 아무것도 하지 않고 다음 i로 넘어갑니다.
        if emitted_this_call < num_to_emit and num_active_electrons[None] < MAX_ELECTRONS:
            if electron_is_active[i] == 0: # 현재 전자 슬롯 'i'가 비활성 상태(사용 가능)인지 확인.
                emission_x_start = 0.0

                slit_center_y = (GRID_Y_MIN + GRID_Y_MAX) / 2.0
                slit_width_y = 0.01 # 입자용으로 넓은 방출 영역

                y_offset = (ti.random(ti.f32) - 0.5) * slit_width_y

                electron_pos[i] = ti.Vector([emission_x_start, slit_center_y + y_offset])

                electron_vel[i] = ti.Vector([
                    start_speed * ti.cos(incident_angle_radians),
                    start_speed * ti.sin(incident_angle_radians)
                ])

                electron_traveled[i] = 0.0
                electron_is_active[i] = 1
                electron_is_particle_state[i] = 1 # 처음에는 입자 상태
                electron_has_scattered_bragg[i] = 0 # 새로 방출된 전자는 브래그 산란되지 않음
                electron_colors[i] = ti.Vector([1.0, 0.0, 0.0]) # 초기 전자 색상 (빨강)
                ti.atomic_add(num_active_electrons[None], 1)
                emitted_this_call += 1


# --- De Broglie Wavelength Calculation Functions ---
@ti.func
def calculate_de_broglie_wavelength_ti(velocity_magnitude: ti.f32) -> ti.f32:
    """De Broglie wavelength calculation function for use within Taichi kernels."""
    momentum = ELECTRON_MASS_PHYSICAL * velocity_magnitude
    wavelength_result = 0.0
    if momentum >= DE_BROGLIE_CALC_EPS:
        wavelength_result = PLANCK_CONSTANT / momentum
    return wavelength_result

def calculate_de_broglie_wavelength_python(velocity_magnitude: float) -> float:
    """De Broglie wavelength calculation function for use in Python script."""
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
    Simulates Bragg diffraction for a polycrystalline material.
    Checks Bragg condition against a random lattice plane orientation and d-spacing.
    """
    new_velocity = incident_vel # Default to incident velocity (if no diffraction)

    wavelength = calculate_de_broglie_wavelength_ti(current_speed)

    should_attempt_diffraction = 1 # Flag to control if diffraction logic runs

    if wavelength <= DE_BROGLIE_CALC_EPS:
        should_attempt_diffraction = 0

    if num_nickel_d_spacings[None] == 0: # If no d-spacings are loaded
        should_attempt_diffraction = 0
    
    if should_attempt_diffraction == 1:
        # 1. Randomly select a d-spacing from the pre-calculated list
        random_d_idx = ti.cast(ti.random(ti.f32) * num_nickel_d_spacings[None], ti.i32)
        current_d_spacing = nickel_d_spacings[random_d_idx]

        # 2. Generate a random lattice plane normal vector (for 2D)
        random_phi = ti.random(ti.f32) * 2.0 * ti.math.pi
        random_lattice_normal = ti.Vector([ti.cos(random_phi), ti.sin(random_phi)]).normalized()

        # 3. Check Bragg condition
        incident_direction = incident_vel.normalized()
        
        theta_prime = ti.acos(ti.abs(incident_direction.dot(random_lattice_normal)))
        bragg_theta_incident = ti.abs(ti.math.pi / 2.0 - theta_prime) # Angle between plane and beam

        required_sin_theta_formula = (BRAGG_ORDER * wavelength) / (2.0 * current_d_spacing)

        if required_sin_theta_formula >= -1.0 + COLLISION_EPS and required_sin_theta_formula <= 1.0 - COLLISION_EPS:
            bragg_theta_from_formula = ti.asin(required_sin_theta_formula)

            if ti.abs(bragg_theta_incident - bragg_theta_from_formula) < tolerance:
                # Bragg condition is met, diffraction occurs
                dot_prod_in_normal = incident_direction.dot(random_lattice_normal)
                scattered_direction = incident_direction - 2 * dot_prod_in_normal * random_lattice_normal
                
                new_velocity = scattered_direction * current_speed
    
    return new_velocity


@ti.kernel
def update_electrons(dt: ti.f32):
    """Updates the position, velocity, and state of all active electrons (adapted for 2D)."""
    for i in range(MAX_ELECTRONS):
        if electron_is_active[i] == 0:
            continue

        prev_pos = electron_pos[i]
        current_vel = electron_vel[i]
        current_speed = current_vel.norm()

        next_pos = electron_pos[i] + current_vel * dt
        electron_traveled[i] += current_vel.norm() * dt

        is_electron_exited_x = 0
        should_reflect = 0 # Use a separate flag for reflection

        # Boundary checks
        # If electron goes beyond X_max (DOMAIN_SIZE), deactivate
        if next_pos[0] > DOMAIN_SIZE:
            is_electron_exited_x = 1
            electron_pos[i][0] = DOMAIN_SIZE # Snap to boundary for exit angle calc
            electron_vel[i] = ti.Vector([0.0, 0.0]) # Stop movement for exited electrons
            electron_is_active[i] = 0 # Deactivate
            ti.atomic_sub(num_active_electrons[None], 1)
            electron_colors[i] = ti.Vector([0.5, 0.5, 0.5]) # Grey (inactive)
            continue # Skip further processing for deactivated electrons

        # If electron goes beyond X_min (0.0), or Y_min/Y_max, reflect
        if next_pos[0] < 0.0:
            electron_pos[i][0] = 0.0 + COLLISION_EPS
            electron_vel[i][0] *= -1.0 # Reflect X velocity
            should_reflect = 1
        
        if next_pos[1] < 0.0:
            electron_pos[i][1] = 0.0 + COLLISION_EPS
            electron_vel[i][1] *= -1.0 # Reflect Y velocity
            should_reflect = 1
        elif next_pos[1] > DOMAIN_SIZE:
            electron_pos[i][1] = DOMAIN_SIZE - COLLISION_EPS
            electron_vel[i][1] *= -1.0 # Reflect Y velocity
            should_reflect = 1

        if should_reflect == 1:
            # If reflection happened, recalculate next_pos with the reflected velocity
            next_pos = prev_pos + electron_vel[i] * dt 

        # --- Wave-Particle Transition and Atom Interaction Logic ---
        # If electron enters lattice region, transition to wave state
        if is_inside_lattice_region(next_pos) and electron_is_particle_state[i] == 1:
            electron_is_particle_state[i] = 0 # Transition to wave state (0)
            electron_colors[i] = ti.Vector([0.5, 0.0, 0.5]) # Purple (Wave state)

        # Only interact with lattice atoms (attempt diffraction) if in wave state
        if electron_is_particle_state[i] == 0:
            is_scattered_this_frame = 0

            # Probability of attempting diffraction in a given frame
            # 파동 상태일 때만 Bragg 회절을 시도 (기존 로직 유지)
            diffraction_attempt_probability_per_frame = 0.01

            if ti.random(ti.f32) < diffraction_attempt_probability_per_frame:
                new_scatter_vel = calculate_bragg_scatter_direction_for_polycrystal(
                    current_vel,
                    current_speed,
                    BRAGG_ANGLE_TOLERANCE_RADIANS
                )

                if (new_scatter_vel - current_vel).norm() > COLLISION_EPS: # If velocity vector changed (diffraction occurred)
                    electron_vel[i] = new_scatter_vel
                    is_scattered_this_frame = 1
                    if electron_has_scattered_bragg[i] == 0:
                        electron_has_scattered_bragg[i] = 1
                        ti.atomic_add(electron_scattered_count[None], 1)

            if is_scattered_this_frame == 0:
                # Small random deviation if no Bragg scattering occurs, simulating amorphous scattering
                random_deviation_factor = 0.005
                random_vec = ti.Vector([ti.random(ti.f32) * 2 - 1, ti.random(ti.f32) * 2 - 1])
                random_vec_norm = random_vec.norm()
                if random_vec_norm > COLLISION_EPS:
                    new_vel = current_vel + random_vec / random_vec_norm * random_deviation_factor * current_speed
                    electron_vel[i] = new_vel.normalized() * current_speed


        # If electron exits lattice region, transition back to particle state
        if electron_is_particle_state[i] == 0 and not is_inside_lattice_region(next_pos):
            electron_is_particle_state[i] = 1
            if electron_has_scattered_bragg[i] == 1:
                electron_colors[i] = ti.Vector([1.0, 0.0, 1.0]) # Magenta (Bragg-scattered electron)
            else:
                electron_colors[i] = ti.Vector([0.0, 0.0, 1.0]) # Blue (Non-Bragg-scattered electron)

            # Record exit angle relative to the NEGATIVE X-axis (180 degrees or pi radians)
            if current_speed > COLLISION_EPS:
                # Get the angle relative to the +X axis (0 to 2*pi)
                angle_rad_plus_x = ti.atan2(electron_vel[i][1], electron_vel[i][0])
                if angle_rad_plus_x < 0.0:
                    angle_rad_plus_x += 2 * ti.math.pi # Normalize to [0, 2*pi)

                # Convert to angle relative to -X axis (180 degrees or pi radians)
                # An angle of 0 deg (+X) becomes 180 deg (relative to -X)
                # An angle of 180 deg (-X) becomes 0 deg (relative to -X)
                # Basically, we shift the origin by pi radians
                angle_rad_neg_x_base = angle_rad_plus_x - ti.math.pi
                
                # Normalize to [0, 2*pi) for consistent histogram range
                if angle_rad_neg_x_base < 0.0:
                    angle_rad_neg_x_base += 2 * ti.math.pi

                idx = ti.atomic_add(electron_exit_count[None], 1)
                if idx < MAX_ELECTRONS:
                    electron_exit_angles[idx] = angle_rad_neg_x_base # Store angle relative to -X axis

        # --- Electron Transmittance / Removal Logic ---
        # 전자가 격자 영역 내부에 있는 경우 (입자 상태든 파동 상태든 관계없이) 감소 확률 적용
        if electron_is_active[i] == 1 and is_inside_lattice_region(next_pos):
            if ti.random(ti.f32) > TRANSMITTANCE_PROBABILITY:
                # Electron is removed (absorbed/scattered out of detection)
                electron_is_active[i] = 0
                electron_vel[i] = ti.Vector([0.0, 0.0])
                ti.atomic_sub(num_active_electrons[None], 1)
                electron_colors[i] = ti.Vector([0.2, 0.2, 0.2]) # Dark grey for removed electrons
                continue # Skip further updates for this electron in this frame

        electron_pos[i] = next_pos


# --- 랜더링 함수 ---
def print_random_electron_info():
    active_electrons_np = electron_is_active.to_numpy()
    active_indices = np.where(active_electrons_np == 1)[0]
    
    if len(active_indices) == 0:
        print(f"[{total_sim_time:.2e}s] No active electrons to print.")
        return

    num_to_print = min(10, len(active_indices))
    
    # 활성 전자 중에서 무작위로 10개 선택 (또는 그 이하)
    selected_indices = random.sample(list(active_indices), num_to_print)

    print(f"\n--- Random {num_to_print} Active Electron Info (at {total_sim_time:.2e}s) ---")
    for idx in selected_indices:
        pos = electron_pos[idx].to_numpy()
        vel = electron_vel[idx].to_numpy()
        is_particle = electron_is_particle_state[idx] == 1
        has_scattered = electron_has_scattered_bragg[idx] == 1
        
        state_str = "Particle" if is_particle else "Wave"
        scatter_str = "Bragg Scattered" if has_scattered else "Not Bragg Scattered"

        print(f"  Electron {idx: <5}: Pos=({pos[0]:.4f}, {pos[1]:.4f}), Vel=({vel[0]:.2e}, {vel[1]:.2e}), State={state_str}, Scattered={scatter_str}")
    print("--------------------------------------------------")


# --- Main Simulation Loop ---
def main():
    # Load d-spacing data calculated in Python to Taichi
    load_d_spacings_to_taichi(dict_nickel_ds)

    # 가속 전압을 54.0V로 변경
    acceleration_voltage = 54.0
    initial_electron_speed = ti.sqrt(2 * ELECTRON_CHARGE_PHYSICAL * acceleration_voltage / ELECTRON_MASS_PHYSICAL)
    print(f"Initial electron speed: {initial_electron_speed:.2e} m/s (accelerated by {acceleration_voltage}V)")
    initial_de_broglie_wavelength = calculate_de_broglie_wavelength_python(initial_electron_speed)
    print(f"Initial de Broglie wavelength: {initial_de_broglie_wavelength:.2e} m")

    # Electron incident angle 0 degrees (along X-axis)
    electron_emission_angle_radians_for_kernel = math.radians(0.0) # Electrons are emitted along +X
    print(f"Electron emission angle (relative to +X-axis, deg): {math.degrees(electron_emission_angle_radians_for_kernel):.2f}")
    print(f"Bragg angle tolerance (deg): {math.degrees(BRAGG_ANGLE_TOLERANCE_RADIANS):.2f}")

    initialize_electrons_initial()
    initialize_lattice_atoms_for_rendering() # Initialize lattice atoms for GUI rendering

    gui = ti.GUI("Davidson-Germer Experiment Simulation (Polycrystalline Bragg Diffraction)", res=(800, 600), background_color=0x112F41)
    frame = 0
    global total_sim_time 
    total_sim_time = 0.0
    total_electrons_emitted = 0 # Track total emitted electrons

    MAX_SIMULATION_FRAMES = 100000 # 시뮬레이션 최대 프레임 수 늘림

    prev_active_electrons = num_active_electrons[None] # 이전 프레임의 활성 전자 수 저장

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        # 전자 방출: 첫 프레임에만 한 번 방출
        if frame == 0: 
            # ELECTRONS_PER_EMISSION을 MAX_ELECTRONS와 동일하게 설정하여 한 번에 모두 방출
            emit_new_electrons_kernel(initial_electron_speed, electron_emission_angle_radians_for_kernel, ELECTRONS_PER_EMISSION)
            total_electrons_emitted = ELECTRONS_PER_EMISSION # total_electrons_emitted를 실제 방출된 MAX_ELECTRONS로 설정

        update_electrons(DT)
        total_sim_time += DT

        current_active_electrons = num_active_electrons[None]
        electrons_removed_this_frame = prev_active_electrons - current_active_electrons

        # --- 로깅 시작 ---
        if frame % 1 == 0: 
            if electrons_removed_this_frame > 0:
                print(f"[{total_sim_time:.2e}s] Electrons Removed This Frame: {electrons_removed_this_frame}, Current Active: {current_active_electrons}")
            
        if frame % 100 == 0:
            print_random_electron_info()
            print(f"[{total_sim_time:.2e}s] Active Electrons: {current_active_electrons}")

        # --- 로깅 끝 ---

        prev_active_electrons = current_active_electrons 

        # 시뮬레이션 종료 조건
        if num_active_electrons[None] == 0 and total_electrons_emitted > 0: # total_electrons_emitted가 0이 아닌 경우에만 종료
            print("\nAll emitted electrons have exited the simulation domain or been removed. Terminating simulation.")
            break
        
        if frame >= MAX_SIMULATION_FRAMES:
            print(f"\nMaximum simulation frames ({MAX_SIMULATION_FRAMES}) reached. Force terminating simulation.")
            break


        # --- GUI Rendering ---
        gui.clear(0x112F41)

        scale_factor = 1.0 / DOMAIN_SIZE

        # Render sample region (border lines)
        rect_color = 0x555555 # Grey border
        line_width = 1

        x0_scaled = GRID_X_POSITION * scale_factor
        x1_scaled = (GRID_X_POSITION + GRID_THICKNESS) * scale_factor
        y0_scaled = GRID_Y_MIN * scale_factor
        y1_scaled = GRID_Y_MAX * scale_factor

        gui.line(begin=np.array([x0_scaled, y0_scaled]), end=np.array([x1_scaled, y0_scaled]), color=rect_color, radius=line_width) # Top
        gui.line(begin=np.array([x0_scaled, y1_scaled]), end=np.array([x1_scaled, y1_scaled]), color=rect_color, radius=line_width) # Bottom
        gui.line(begin=np.array([x0_scaled, y0_scaled]), end=np.array([x0_scaled, y1_scaled]), color=rect_color, radius=line_width) # Left
        gui.line(begin=np.array([x1_scaled, y0_scaled]), end=np.array([x1_scaled, y1_scaled]), color=rect_color, radius=line_width) # Right


        # Visualize electron emission origin (large white circle)
        emission_x_gui = 0.0 * scale_factor
        slit_center_y_gui_display = ((GRID_Y_MIN + GRID_Y_MAX) / 2.0) * scale_factor
        gui.circles(np.array([[emission_x_gui, slit_center_y_gui_display]]), radius=7, color=0xFFFFFF)

        # It's more efficient to convert the entire Taichi field to NumPy
        # than looping and appending individual elements.
        all_electron_pos_np = electron_pos.to_numpy()
        all_electron_colors_np = electron_colors.to_numpy()
        all_electron_active_np = electron_is_active.to_numpy()
        all_electron_particle_state_np = electron_is_particle_state.to_numpy()

        # Filter only active electrons for rendering to avoid drawing deactivated ones
        active_indices = np.where(all_electron_active_np == 1)[0]
        
        if len(active_indices) > 0:
            current_render_positions = all_electron_pos_np[active_indices]
            current_render_colors = np.copy(all_electron_colors_np[active_indices]) # Use copy to modify colors

            # Apply wave state color for rendering
            wave_state_indices = np.where(all_electron_particle_state_np[active_indices] == 0)[0]
            if len(wave_state_indices) > 0:
                current_render_colors[wave_state_indices] = np.array([0.5, 0.0, 0.5]) # Purple (Wave state)

            converted_colors_np = (current_render_colors * 255).astype(np.uint32)
            packed_colors = (converted_colors_np[:, 0] << 16) | \
                            (converted_colors_np[:, 1] << 8) | \
                            converted_colors_np[:, 2]

            gui.circles(current_render_positions * scale_factor,
                        radius=2,
                        color=packed_colors)

        # Display simulation info
        gui.text(f"Active Electrons: {num_active_electrons[None]}", pos=(0.05, 0.95), color=0xFFFFFF)
        gui.text(f"Total Emitted: {total_electrons_emitted}", pos=(0.05, 0.90), color=0xFFFFFF)
        gui.text(f"Bragg Scattered: {electron_scattered_count[None]}", pos=(0.05, 0.85), color=0xFFFFFF)


        gui.show()
        frame += 1

    # --- Post-Simulation Analysis (Histogram) ---
    gui.close()

    active_exit_angles_rad = electron_exit_angles.to_numpy()[:electron_exit_count[None]]
    total_scattered_electrons = electron_scattered_count[None]
    print(f"\n--- Simulation Summary ---")
    print(f"Total electrons emitted: {total_electrons_emitted}")
    print(f"Total Bragg-scattered electrons: {total_scattered_electrons}")
    if total_electrons_emitted > 0:
        print(f"Bragg scattering ratio: {total_scattered_electrons / total_electrons_emitted * 100:.2f}%")

    if len(active_exit_angles_rad) > 0:
        plt.figure(figsize=(10, 6))

        # Convert to degrees and ensure range is 0-360
        scattered_angles_deg_normalized = np.degrees(active_exit_angles_rad)
        
        # Calculate histogram data
        counts, bins, patches = plt.hist(scattered_angles_deg_normalized, bins=72, range=(0, 360), edgecolor='black')
        
        # Calculate the median of the counts
        non_zero_counts = counts[counts > 0] # 0이 아닌 빈만 고려
        if len(non_zero_counts) > 0:
            median_count = np.median(non_zero_counts) # Median for plotting reference line
            plt.ylim(0, median_count * 2.0) # Y-axis limit set to 1.5 times the median count
            plt.axhline(median_count, color='green', linestyle=':', label=f'Median Count: {int(median_count)}')
            print(f"Median y-axis count: {median_count}")
        else:
            print("No non-zero counts in histogram bins.")


        plt.title('Distribution of Electron Scattering Angles (Relative to -X Axis)')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Number of Electrons')
        plt.xticks(np.arange(0, 361, 50))
        plt.grid(True)
        
        # Incident Beam Direction: Electron emitted along +X, so incident beam is +X.
        # But we are plotting relative to -X axis (180 degrees). So incident beam is 180 deg.
        # plt.axvline(x=180, color='r', linestyle='--', label='Incident Beam Direction (-X axis)') 
        
        # Placeholder for expected Bragg peaks based on d-spacing
        # Recalculate expected Bragg angles relative to -X axis
        for hkl_key, d_val in dict_nickel_ds.items():
            required_sin_theta = (BRAGG_ORDER * initial_de_broglie_wavelength) / (2.0 * d_val)
            if required_sin_theta >= -1.0 and required_sin_theta <= 1.0:
                bragg_theta_rad = math.asin(required_sin_theta) 
                
                # The expected deflection angle from the incident *beam* (which is along +X)
                # is 2 * bragg_theta_rad in either positive or negative Y direction (relative to X).
                # Example: If incident is at 0 degrees (+X), Bragg peaks appear at +2*theta and -2*theta
                
                expected_scattering_angle_plus_y_from_plus_x = math.degrees(2 * bragg_theta_rad)
                expected_scattering_angle_minus_y_from_plus_x = -math.degrees(2 * bragg_theta_rad)

                # Convert these +X relative angles to -X relative angles for plotting
                # Angle relative to -X is (Angle_from_+X + 180) % 360
                
                # For positive Y deflection (from +X), which corresponds to (0 + 2*theta) from +X
                bragg_peak_angle_pos = (expected_scattering_angle_plus_y_from_plus_x + 180) % 360
                plt.axvline(x=bragg_peak_angle_pos, color='c', linestyle=':', label=f'Exp. Bragg Peak ({hkl_key}) at {bragg_peak_angle_pos:.1f}°') 
                
                # For negative Y deflection (from +X), which corresponds to (0 - 2*theta) from +X
                bragg_peak_angle_neg = (expected_scattering_angle_minus_y_from_plus_x + 180) % 360
                # Avoid plotting duplicate if angle is 0 or 180 (straight through or backscatter)
                # Using a small epsilon for float comparison
                if abs(bragg_peak_angle_pos - bragg_peak_angle_neg) > 1e-3: # Use a small float tolerance
                    # plt.axvline(x=bragg_peak_angle_neg, color='c', linestyle=':') # No label for second line
                    pass

        plt.tight_layout()
        plt.show() 
    else:
        print("Not enough electron data to plot histogram (electrons might not have exited the lattice or count is too low).")

if __name__ == "__main__":
    main()