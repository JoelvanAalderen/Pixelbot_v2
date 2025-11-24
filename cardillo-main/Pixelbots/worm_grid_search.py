
## Import libraries
import numpy as np
import time
from vtk import VTK_VERTEX
from pathlib import Path
from multiprocessing import Pool

from cardillo import System
from cardillo.solver import Moreau, SolverOptions, Solution, save_solution
from Multiplebots import MultiPixelbot
from Pixelbot_contact import Pixelbot_contact



## input for parallel sim
def sim_objective(args):
    sim_id, bot_part, pixel_size, Placement, mass_values, stiffness_values, A_values, n_pixel_types, period, dt, t_final, start_p_position, start_p_angle = args

    return run_bot_part(sim_id, bot_part, pixel_size, Placement, mass_values,
                        stiffness_values, A_values, n_pixel_types, period, dt,
                        t_final, start_p_position, start_p_angle)


## get placement function
def get_placement():
    return np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 2, 3, 4, 5]
    ], dtype=int)


## Build repeatable ground
def ground_func(mode="walker"):
    if mode == "walker":
        x_vals = np.linspace(-0.1, 2, 4)
        angle = 0.0 #16.3 degrees is static for single pixel
        y_vals = np.tan(np.deg2rad(angle))*x_vals # 0.02 * np.sin(200 * x_vals)-0.02 # 0.0006 * np.sin(200 * x_vals)-0.0006 #
        
        points = [np.array([x, y]) for x, y in zip(x_vals, y_vals)]
        vectors = [points[i+1] - points[i] for i in range(len(points)-1)]

    elif mode == "climber":
        points = [
        np.array([0, 2]),
        np.array([0, -0.1]),
        np.array([0.05, -0.1]),
        np.array([0.05, 2])
        ]
        vectors = [
            points[1]-points[0],
            points[2]-points[1],
            points[3]-points[2]
        ]
    
    return points, vectors


## sim function for a part of bots
def run_bot_part(sim_id, bot_part, pixel_size, Placement, mass_values, stiffness_values, A_values, n_pixel_types, period, dt, t_final, start_p_position, start_p_angle, save_folder="solutions/worm_grid_search/flat_ground/loop_omega_phi/"):
    if not bot_part:
        print("Empty bot part, skipping simulation")
        return []
    print(f"\nSimulating {len(bot_part)} bots in this part")


    ## create system
    system = System()
    all_bots = []


    ## create folder to save results
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)


    # nr of columns and rows for plotting
    cols = int(np.ceil(np.sqrt(len(bot_part))))
    rows = int(np.ceil(len(bot_part)/cols))

    bot_width = max(Placement.shape)*pixel_size
    x_offset = max(bot_width+0.025, 0.15)
    y_offset = max(bot_width+0.025, 0.15)


    # define ground mode
    mode = "walker"


    # loop assign bots
    for n, (bot_id, omega_sim, phi_sim) in enumerate(bot_part):
        row = n // cols
        col = n % cols

        pts, vecs = ground_func(mode)

        offset_x = col*x_offset
        offset_y = row*y_offset

        bot_offset_x = start_p_position[0] + offset_x
        bot_offset_y = start_p_position[1] + offset_y
        shifted_points = [p+np.array([offset_x, offset_y]) for p in pts]
        shifted_vectors = [v.copy() for v in vecs]

        def ground_func_for_bot():
            return shifted_points, shifted_vectors

        bot_start_pos = (bot_offset_x,
                         bot_offset_y)
        
        bot_name = f"bot_{bot_id}"

        n_total_pixels = n_pixel_types + 2
        n_actuated = n_pixel_types

        pixel_prop = {}
        for j in range(n_total_pixels):
            if j < n_actuated: 
                pixel_prop[j+1] = {
                "mass": mass_values[j], 
                "stiffness": stiffness_values[j],
                "A": A_values[j],
                "omega": omega_sim,
                "phi": phi_sim * j * (2*np.pi/5)
            }
            else:
                pixel_prop[j+1] = {
                    "mass": mass_values[j],
                    "stiffness": stiffness_values[j],
                    "A": 0.0,
                    "omega": 0.0,
                    "phi": 0.0
                    }  

        bot = MultiPixelbot(bot_name, Placement, pixel_size, pixel_prop,
                            period, bot_start_pos, start_p_angle)
        
        Ground_contact = Pixelbot_contact(bot, ground_func_for_bot, name=f"contact_{bot_name}")


        bot.global_id = bot_id
        system.add(bot, Ground_contact)
        all_bots.append(bot)


    ## assemble and run sim
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False)) 
    solver = Moreau(system, t_final, dt)    
    sol = solver.solve()


    bot_numbers = sorted([bot.global_id for bot in all_bots])
    if len(bot_numbers) > 1:
        bot_range_str = f"{bot_numbers[0]}-{bot_numbers[-1]}"
    else:
        bot_range_str = f"{bot_numbers[0]}"

    filename = save_path / f"full_solution_of_sim_{sim_id}_bots_{bot_range_str}.pkl"

    full_sol = Solution(system, sol.t, sol.q.copy())
    full_sol.u = sol.u.copy()
    save_solution(full_sol, filename)

    t = sol.t                                                               
    q = sol.q                                                           
    u = sol.u    


    ## get results for each bot
    results = []

    for bot in all_bots:
        com_x, com_y = [], []

        for frame in range(len(t)):
            bot_dofs = np.unique(bot.pixelDOF.flatten())
            q_frame = q[frame]
            q_bot = q_frame[bot_dofs]
            cx, cy = bot.center_of_mass(t[frame], q_bot)
            com_x.append(cx)
            com_y.append(cy)
        
        comx = np.array(com_x)
        comy = np.array(com_y)  

        print("bot_id =", bot.global_id, "start_pos =", bot_start_pos)
        print("com initial =", [comx[0], comy[0]])

        results.append({
            "global_id": bot.global_id,
            "omega": [bot.pixel_prop[j + 1]["omega"] for j in range(n_pixel_types)],
            "phi": [bot.pixel_prop[j + 1]["phi"] for j in range(n_pixel_types)],
            "final_com_dx": comx[-1] - comx[0],
            "final_com_x": comx[-1],
            "final_com_y": comy[-1],
            "start_com_x": comx[0],
            "start_com_y": comy[0]
        })
    
    return results


## main code
if __name__ == "__main__":
    ## define sim parameters
    placement = get_placement()
    all_pixel_types = np.unique(placement)
    actuated_pixel_types = [1, 2, 3, 4, 5]
    n_pixel_types = len([pt for pt in all_pixel_types if pt in actuated_pixel_types])
    total_pixel_types = len(np.unique(get_placement())) - 1


    n_cores = 8 #nr of parallel sim
    n_omega_options = 10
    n_phi_options = 10


    ## Main setup
    pixel_size = 0.01
    Placement = get_placement()
    start_p_position = (0.0, 0.0)
    start_p_angle = np.deg2rad(0.0)
    stiffness_values = [3400]*n_pixel_types + [4200, 9200]
    mass_values = [0.000852]*n_pixel_types + [0.000972, 0.000989]
    A_values = [0.6]*n_pixel_types + [0.0, 0.0]

    natural_omega = [np.sqrt(s / m) for s, m in zip(stiffness_values, mass_values)]
    period = 2*np.pi/min(natural_omega)
    dt = (2*np.pi/max(natural_omega))/20
    t_final = 400*period


    
    bot_configs = []
    global_id = 0
    ## make all bot configurations grid_search
    omega_values = np.linspace(0.001*max(natural_omega), 0.05*max(natural_omega), n_omega_options)
    phi_values = np.linspace(0, 5, n_phi_options)
    for omega in omega_values: #itertools.product(omega_values, repeat=n_pixel_types):
        for phi in phi_values: #itertools.product(phi_values, repeat=n_pixel_types):
            bot_configs.append((global_id, omega, phi))
            global_id += 1
    print(f"Total bots in simulation: {len(bot_configs)}")


    ## split bot_config in parts for parallel sim
    parts, remainder = divmod(len(bot_configs), n_cores)
    bot_parts = [bot_configs[i*(parts)+min(i,remainder):(i+1)*parts+min(i+1,remainder)] for i in range(n_cores)]

    
    ## run parallel sim
    param_combo = [(sim_id, part, pixel_size, Placement, mass_values, stiffness_values, A_values,
              n_pixel_types, period, dt, t_final, start_p_position, start_p_angle)
             for sim_id, part in enumerate(bot_parts)]

    start_time = time.perf_counter()
    results = []

    with Pool(processes=n_cores) as pool:
        for i, part_results in enumerate(pool.imap_unordered(sim_objective, param_combo), start=1):
            results.extend(part_results)
            elapsed = time.perf_counter()- start_time
            print(f"\n{i}/{n_cores} parts finished in {elapsed:.2f} sec ({(elapsed/60):.2f} min).")
    finish_time = time.perf_counter() - start_time
    print(f"All simulations finished in {finish_time:.2f} seconds ({finish_time/60:.2f} min)")

    
    ## filter results to show top parameter combinations
    filtered_results = [r for r in results if r["final_com_y"] < 0.01]
    # print top results in terminal
    top_n = 10
    top_simulations = sorted(filtered_results, key=lambda r: r['final_com_dx'], reverse=True)#[:top_n]

    print(f"\n--- Top {top_n} Simulations (based on max COM dx) ---")
    print(f"{'Bot nr':>7} | {'Omega':>25} | {'Phi':>25} | {'COM dx':>10} | {'COM x':>10} | {'COM y':>10}")
    print("-" * 110)
    for r in top_simulations:
        omega_str = str([round(float(p), 2) for p in r['omega']])
        phi_str = str([round(float(p), 2) for p in r['phi']])
        print(f"{r['global_id']:7} | {omega_str:25} | {phi_str:25} | {r['final_com_dx']:10.5f} | {r['final_com_x']:10.5f} | {r['final_com_y']:10.5f}")

    invalid_results = [r for r in results if r["final_com_y"] >= 0.03]
    print(*[r['global_id'] for r in invalid_results], sep=', ')
    print(f"\nTotal invalid results: {int(len(invalid_results))}.")

