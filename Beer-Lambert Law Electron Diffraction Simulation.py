import taichi as ti
import numpy as np
import math
import matplotlib
import scipy.constants
from sympy import symbols, Eq, solve
from math import gcd
import random

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

DIM = 2
MAX_ELECTRONS = 50000

ELECTRON_RADIUS = 0.0005

COLLISION_EPS = 1e-9
DE_BROGLIE_CALC_EPS = 1e-30

DT = 5e-11

DOMAIN_SIZE = 0.35

PLANCK_CONSTANT = scipy.constants.value('Planck constant')
ELECTRON_MASS_PHYSICAL = scipy.constants.value('electron mass')
ELECTRON_CHARGE_PHYSICAL = scipy.constants.value('elementary charge')

ELECTRONS_PER_EMISSION = 50000
EMISSION_INTERVAL_FRAMES = 1

GRID_X_POSITION = 0.20
GRID_THICKNESS = 0.005

GRID_SIZE_Y = 0.08
GRID_Y_MIN = (DOMAIN_SIZE / 2.0) - (GRID_SIZE_Y / 2.0)
GRID_Y_MAX = (DOMAIN_SIZE / 2.0) + (GRID_SIZE_Y / 2.0)

GUI_LATTICE_CONSTANT_FOR_RENDERING = 0.00008

num_cells_y_gui_for_rendering = int(GRID_SIZE_Y / GUI_LATTICE_CONSTANT_FOR_RENDERING)
MAX_LATTICE_ATOMS_FOR_RENDERING = num_cells_y_gui_for_rendering + 1

BRAGG_ORDER = 1
BRAGG_ANGLE_TOLERANCE_RADIANS = math.radians(2.0)

TRANSMITTANCE_PROBABILITY = 0.999807

SPH_SMOOTHING_RADIUS = 0.01
SPH_KERNEL_CONSTANT = 315.0 / (64.0 * math.pi * SPH_SMOOTHING_RADIUS**9)
SPH_PRESSURE_CONSTANT = 2000.0
SPH_VISCOSITY_CONSTANT = 0.1

NM = 10 ** (-9)

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

electron_density = ti.field(dtype=ti.f32, shape=MAX_ELECTRONS)
electron_pressure = ti.field(dtype=ti.f32, shape=MAX_ELECTRONS)
electron_acceleration = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_ELECTRONS)

@ti.func
def sph_kernel(r: ti.f32, h: ti.f32) -> ti.f32:
    q = r / h
    # Taichi에서는 조건부 반환 대신 조건부 곱셈을 사용
    condition = ti.cast(q >= 0.0 and q <= 1.0, ti.f32)
    kernel_value = (1.0 - q) * (1.0 - q) * (1.0 - q) * (1.0 - q) * (1.0 + 4.0 * q)
    return condition * kernel_value

@ti.func
def sph_kernel_gradient(r: ti.f32, h: ti.f32) -> ti.f32:
    q = r / h
    # Taichi에서는 조건부 반환 대신 조건부 곱셈을 사용
    condition = ti.cast(q >= 0.0 and q <= 1.0, ti.f32)
    gradient_value = -20.0 * q * (1.0 - q) * (1.0 - q) * (1.0 - q) / h
    return condition * gradient_value

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
        electron_density[i] = 0.0
        electron_pressure[i] = 0.0
        electron_acceleration[i] = ti.Vector([0.0, 0.0])
    num_active_electrons[None] = 0
    electron_exit_count[None] = 0
    electron_scattered_count[None] = 0

@ti.kernel
def initialize_lattice_atoms_for_rendering():
    current_atom_idx = 0
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
def calculate_electron_density():
    for i in range(MAX_ELECTRONS):
        if electron_is_active[i] == 0 or electron_is_particle_state[i] == 0:
            electron_density[i] = 0.0
            continue
            
        density = 0.0
        pos_i = electron_pos[i]
        
        for j in range(MAX_ELECTRONS):
            if electron_is_active[j] == 0 or electron_is_particle_state[j] == 0:
                continue
                
            pos_j = electron_pos[j]
            r = (pos_i - pos_j).norm()
            
            if r < SPH_SMOOTHING_RADIUS:
                density += sph_kernel(r, SPH_SMOOTHING_RADIUS)
        
        electron_density[i] = density

@ti.kernel
def calculate_electron_pressure():
    for i in range(MAX_ELECTRONS):
        if electron_is_active[i] == 0 or electron_is_particle_state[i] == 0:
            electron_pressure[i] = 0.0
            continue
            
        electron_pressure[i] = SPH_PRESSURE_CONSTANT * (electron_density[i] - 1.0)

@ti.kernel
def calculate_electron_acceleration():
    for i in range(MAX_ELECTRONS):
        if electron_is_active[i] == 0 or electron_is_particle_state[i] == 0:
            electron_acceleration[i] = ti.Vector([0.0, 0.0])
            continue
            
        acceleration = ti.Vector([0.0, 0.0])
        pos_i = electron_pos[i]
        density_i = electron_density[i]
        pressure_i = electron_pressure[i]
        
        for j in range(MAX_ELECTRONS):
            if electron_is_active[j] == 0 or electron_is_particle_state[j] == 0 or i == j:
                continue
                
            pos_j = electron_pos[j]
            r_vec = pos_i - pos_j
            r = r_vec.norm()
            
            if r < SPH_SMOOTHING_RADIUS and r > COLLISION_EPS:
                density_j = electron_density[j]
                pressure_j = electron_pressure[j]
                
                pressure_grad = sph_kernel_gradient(r, SPH_SMOOTHING_RADIUS)
                pressure_acc = -pressure_grad * (pressure_i + pressure_j) / (2.0 * density_i) * r_vec / r
                acceleration += pressure_acc
                
                vel_i = electron_vel[i]
                vel_j = electron_vel[j]
                viscosity_acc = SPH_VISCOSITY_CONSTANT * pressure_grad * (vel_j - vel_i) / density_i
                acceleration += viscosity_acc
        
        electron_acceleration[i] = acceleration


@ti.kernel
def emit_new_electrons_kernel(start_speed: ti.f32, incident_angle_radians: ti.f32, num_to_emit: ti.i32):
    emitted_this_call = 0 
    for i in range(MAX_ELECTRONS):
        if emitted_this_call < num_to_emit and num_active_electrons[None] < MAX_ELECTRONS:
            if electron_is_active[i] == 0:
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
    momentum = ELECTRON_MASS_PHYSICAL * velocity_magnitude
    wavelength_result = 0.0
    if momentum >= DE_BROGLIE_CALC_EPS:
        wavelength_result = PLANCK_CONSTANT / momentum
    return wavelength_result

def calculate_de_broglie_wavelength_python(velocity_magnitude: float) -> float:
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
        
        theta_prime = ti.acos(ti.abs(incident_direction.dot(random_lattice_normal)))
        bragg_theta_incident = ti.abs(ti.math.pi / 2.0 - theta_prime)

        required_sin_theta_formula = (BRAGG_ORDER * wavelength) / (2.0 * current_d_spacing)

        if required_sin_theta_formula >= -1.0 + COLLISION_EPS and required_sin_theta_formula <= 1.0 - COLLISION_EPS:
            bragg_theta_from_formula = ti.asin(required_sin_theta_formula)

            if ti.abs(bragg_theta_incident - bragg_theta_from_formula) < tolerance:
                dot_prod_in_normal = incident_direction.dot(random_lattice_normal)
                scattered_direction = incident_direction - 2 * dot_prod_in_normal * random_lattice_normal
                
                new_velocity = scattered_direction * current_speed
    
    return new_velocity


@ti.kernel
def update_electrons(dt: ti.f32):
    for i in range(MAX_ELECTRONS):
        if electron_is_active[i] == 0:
            continue

        prev_pos = electron_pos[i]
        current_vel = electron_vel[i]
        current_speed = current_vel.norm()

        next_pos = electron_pos[i] + current_vel * dt
        electron_traveled[i] += current_vel.norm() * dt

        is_electron_exited_x = 0
        should_reflect = 0

        if next_pos[0] > DOMAIN_SIZE:
            is_electron_exited_x = 1
            electron_pos[i][0] = DOMAIN_SIZE
            electron_vel[i] = ti.Vector([0.0, 0.0])
            electron_is_active[i] = 0
            ti.atomic_sub(num_active_electrons[None], 1)
            electron_colors[i] = ti.Vector([0.5, 0.5, 0.5])
            continue

        if next_pos[0] < 0.0:
            electron_pos[i][0] = 0.0 + COLLISION_EPS
            electron_vel[i][0] *= -1.0
            should_reflect = 1
        
        if next_pos[1] < 0.0:
            electron_pos[i][1] = 0.0 + COLLISION_EPS
            electron_vel[i][1] *= -1.0
            should_reflect = 1
        elif next_pos[1] > DOMAIN_SIZE:
            electron_pos[i][1] = DOMAIN_SIZE - COLLISION_EPS
            electron_vel[i][1] *= -1.0
            should_reflect = 1

        if should_reflect == 1:
            next_pos = prev_pos + electron_vel[i] * dt 

        if is_inside_lattice_region(next_pos) and electron_is_particle_state[i] == 1:
            electron_is_particle_state[i] = 0
            electron_colors[i] = ti.Vector([0.5, 0.0, 0.5])

        if electron_is_particle_state[i] == 0:
            is_scattered_this_frame = 0

            diffraction_attempt_probability_per_frame = 0.01

            if ti.random(ti.f32) < diffraction_attempt_probability_per_frame:
                new_scatter_vel = calculate_bragg_scatter_direction_for_polycrystal(
                    current_vel,
                    current_speed,
                    BRAGG_ANGLE_TOLERANCE_RADIANS
                )

                if (new_scatter_vel - current_vel).norm() > COLLISION_EPS:
                    electron_vel[i] = new_scatter_vel
                    is_scattered_this_frame = 1
                    if electron_has_scattered_bragg[i] == 0:
                        electron_has_scattered_bragg[i] = 1
                        ti.atomic_add(electron_scattered_count[None], 1)

            if is_scattered_this_frame == 0:
                random_deviation_factor = 0.005
                random_vec = ti.Vector([ti.random(ti.f32) * 2 - 1, ti.random(ti.f32) * 2 - 1])
                random_vec_norm = random_vec.norm()
                if random_vec_norm > COLLISION_EPS:
                    new_vel = current_vel + random_vec / random_vec_norm * random_deviation_factor * current_speed
                    electron_vel[i] = new_vel.normalized() * current_speed


        if electron_is_particle_state[i] == 0 and not is_inside_lattice_region(next_pos):
            electron_is_particle_state[i] = 1
            if electron_has_scattered_bragg[i] == 1:
                electron_colors[i] = ti.Vector([1.0, 0.0, 1.0])
            else:
                electron_colors[i] = ti.Vector([0.0, 0.0, 1.0])

            if current_speed > COLLISION_EPS:
                angle_rad_plus_x = ti.atan2(electron_vel[i][1], electron_vel[i][0])
                if angle_rad_plus_x < 0.0:
                    angle_rad_plus_x += 2 * ti.math.pi

                angle_rad_neg_x_base = angle_rad_plus_x - ti.math.pi
                
                if angle_rad_neg_x_base < 0.0:
                    angle_rad_neg_x_base += 2 * ti.math.pi

                idx = ti.atomic_add(electron_exit_count[None], 1)
                if idx < MAX_ELECTRONS:
                    electron_exit_angles[idx] = angle_rad_neg_x_base

        if electron_is_particle_state[i] == 1 and electron_is_active[i] == 1:
            electron_vel[i] += electron_acceleration[i] * dt

        if electron_is_active[i] == 1 and is_inside_lattice_region(next_pos):
            if ti.random(ti.f32) > TRANSMITTANCE_PROBABILITY:
                electron_is_active[i] = 0
                electron_vel[i] = ti.Vector([0.0, 0.0])
                ti.atomic_sub(num_active_electrons[None], 1)
                electron_colors[i] = ti.Vector([0.2, 0.2, 0.2])
                continue

        electron_pos[i] = next_pos


def print_random_electron_info():
    active_electrons_np = electron_is_active.to_numpy()
    active_indices = np.where(active_electrons_np == 1)[0]
    
    if len(active_indices) == 0:
        print(f"[{total_sim_time:.2e}s] No active electrons to print.")
        return

    num_to_print = min(10, len(active_indices))
    
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


def main():
    load_d_spacings_to_taichi(dict_nickel_ds)

    acceleration_voltage = 54.0
    initial_electron_speed = ti.sqrt(2 * ELECTRON_CHARGE_PHYSICAL * acceleration_voltage / ELECTRON_MASS_PHYSICAL)
    print(f"Initial electron speed: {initial_electron_speed:.2e} m/s (accelerated by {acceleration_voltage}V)")
    initial_de_broglie_wavelength = calculate_de_broglie_wavelength_python(initial_electron_speed)
    print(f"Initial de Broglie wavelength: {initial_de_broglie_wavelength:.2e} m")

    electron_emission_angle_radians_for_kernel = math.radians(0.0)
    print(f"Electron emission angle (relative to +X-axis, deg): {math.degrees(electron_emission_angle_radians_for_kernel):.2f}")
    print(f"Bragg angle tolerance (deg): {math.degrees(BRAGG_ANGLE_TOLERANCE_RADIANS):.2f}")

    initialize_electrons_initial()
    initialize_lattice_atoms_for_rendering()

    gui = ti.GUI("Davidson-Germer Experiment Simulation (Polycrystalline Bragg Diffraction)", res=(800, 600), background_color=0x112F41)
    frame = 0
    global total_sim_time 
    total_sim_time = 0.0
    total_electrons_emitted = 0

    MAX_SIMULATION_FRAMES = 100000

    prev_active_electrons = num_active_electrons[None]

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        if frame == 0: 
            emit_new_electrons_kernel(initial_electron_speed, electron_emission_angle_radians_for_kernel, ELECTRONS_PER_EMISSION)
            total_electrons_emitted = ELECTRONS_PER_EMISSION

        calculate_electron_density()
        calculate_electron_pressure()
        calculate_electron_acceleration()
        
        update_electrons(DT)
        total_sim_time += DT

        current_active_electrons = num_active_electrons[None]
        electrons_removed_this_frame = prev_active_electrons - current_active_electrons

        if frame % 1 == 0: 
            if electrons_removed_this_frame > 0:
                print(f"[{total_sim_time:.2e}s] Electrons Removed This Frame: {electrons_removed_this_frame}, Current Active: {current_active_electrons}")
            
        if frame % 100 == 0:
            print_random_electron_info()
            print(f"[{total_sim_time:.2e}s] Active Electrons: {current_active_electrons}")


        prev_active_electrons = current_active_electrons 

        if num_active_electrons[None] == 0 and total_electrons_emitted > 0:
            print("\nAll emitted electrons have exited the simulation domain or been removed. Terminating simulation.")
            break
        
        if frame >= MAX_SIMULATION_FRAMES:
            print(f"\nMaximum simulation frames ({MAX_SIMULATION_FRAMES}) reached. Force terminating simulation.")
            break


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


        emission_x_gui = 0.0 * scale_factor
        slit_center_y_gui_display = ((GRID_Y_MIN + GRID_Y_MAX) / 2.0) * scale_factor
        gui.circles(np.array([[emission_x_gui, slit_center_y_gui_display]]), radius=7, color=0xFFFFFF)

        all_electron_pos_np = electron_pos.to_numpy()
        all_electron_colors_np = electron_colors.to_numpy()
        all_electron_active_np = electron_is_active.to_numpy()
        all_electron_particle_state_np = electron_is_particle_state.to_numpy()

        active_indices = np.where(all_electron_active_np == 1)[0]
        
        if len(active_indices) > 0:
            current_render_positions = all_electron_pos_np[active_indices]
            current_render_colors = np.copy(all_electron_colors_np[active_indices])

            wave_state_indices = np.where(all_electron_particle_state_np[active_indices] == 0)[0]
            if len(wave_state_indices) > 0:
                current_render_colors[wave_state_indices] = np.array([0.5, 0.0, 0.5])

            converted_colors_np = (current_render_colors * 255).astype(np.uint32)
            packed_colors = (converted_colors_np[:, 0] << 16) | \
                            (converted_colors_np[:, 1] << 8) | \
                            converted_colors_np[:, 2]

            gui.circles(current_render_positions * scale_factor,
                        radius=2,
                        color=packed_colors)

        gui.text(f"Active Electrons: {num_active_electrons[None]}", pos=(0.05, 0.95), color=0xFFFFFF)
        gui.text(f"Total Emitted: {total_electrons_emitted}", pos=(0.05, 0.90), color=0xFFFFFF)
        gui.text(f"Bragg Scattered: {electron_scattered_count[None]}", pos=(0.05, 0.85), color=0xFFFFFF)
        gui.text(f"SPH Enabled (Particle State Only)", pos=(0.05, 0.80), color=0x00FF00)


        gui.show()
        frame += 1

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

        scattered_angles_deg_normalized = np.degrees(active_exit_angles_rad)
        
        counts, bins, patches = plt.hist(scattered_angles_deg_normalized, bins=72, range=(0, 360), edgecolor='black')
        
        non_zero_counts = counts[counts > 0]
        if len(non_zero_counts) > 0:
            median_count = np.median(non_zero_counts)
            plt.ylim(0, median_count * 2.0)
            plt.axhline(median_count, color='green', linestyle=':', label=f'Median Count: {int(median_count)}')
            print(f"Median y-axis count: {median_count}")
        else:
            print("No non-zero counts in histogram bins.")


        plt.title('Distribution of Electron Scattering Angles (Relative to -X Axis)')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Number of Electrons')
        plt.xticks(np.arange(0, 361, 50))
        plt.grid(True)
        
        
        for hkl_key, d_val in dict_nickel_ds.items():
            required_sin_theta = (BRAGG_ORDER * initial_de_broglie_wavelength) / (2.0 * d_val)
            if required_sin_theta >= -1.0 and required_sin_theta <= 1.0:
                bragg_theta_rad = math.asin(required_sin_theta) 
                
                
                expected_scattering_angle_plus_y_from_plus_x = math.degrees(2 * bragg_theta_rad)
                expected_scattering_angle_minus_y_from_plus_x = -math.degrees(2 * bragg_theta_rad)

                
                bragg_peak_angle_pos = (expected_scattering_angle_plus_y_from_plus_x + 180) % 360
                plt.axvline(x=bragg_peak_angle_pos, color='c', linestyle=':', label=f'Exp. Bragg Peak ({hkl_key}) at {bragg_peak_angle_pos:.1f}°') 
                
                bragg_peak_angle_neg = (expected_scattering_angle_minus_y_from_plus_x + 180) % 360
                if abs(bragg_peak_angle_pos - bragg_peak_angle_neg) > 1e-3:
                    pass

        plt.tight_layout()
        plt.show() 
    else:
        print("Not enough electron data to plot histogram (electrons might not have exited the lattice or count is too low).")

if __name__ == "__main__":
    main()