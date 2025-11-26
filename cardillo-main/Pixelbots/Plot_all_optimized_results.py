 
## Import libraries
from matplotlib import animation, pyplot as plt
import numpy as np
from pathlib import Path
from cardillo.solver import load_solution
import re, mplcursors



def plot_all_sim():
    # Load the solution
    cwd = Path.cwd()
    filename = "full_solution_of_sim_4_bots_52-63.pkl"
    pkl_path = cwd / "solutions" / "worm_grid_search" / "flat_ground" / "0001_010max(natural_omega)" / filename

    match = re.search(r'bots_(\d+)-(\d+)\.pkl', filename)
    if not match:
        raise ValueError("Filename does not match expected pattern.")
        
    # load solutions
    full_sol = load_solution(pkl_path)
    system = full_sol.system
    t = full_sol.t
    q = full_sol.q
    u = full_sol.u
    t_final = t[-1]

    # find bots in solution, sorted by name
    bot_names = sorted([n for n in system.contributions_map.keys() if n.startswith("bot_")])
    bots = [system.contributions_map[n] for n in bot_names]
    n_bots = len(bots)

    # color map for pixel types
    bot_type_to_color = {1: '#4477AA', 2: '#CCBB44', 3: '#228833', 4: '#EE6677',
                        5: '#AA3377', 6: '#BBBBBB', 7: '#000000', 8: '#66CCEE'}

    # precompute per-bot data
    per_bot = []
    for bot in bots:
        bot_dofs = np.unique(bot.pixelDOF.flatten())
        q_bot_all = q[:, bot_dofs]
        X_all = bot.start_p_position[0] + q_bot_all[:, 0::2]
        Y_all = bot.start_p_position[1] + q_bot_all[:, 1::2]

        spring_pairs = []
        spring_colors = []

        for p_idx, dof in enumerate(bot.pixelDOF):
            btype = bot.bot_types[p_idx]
            for (i_local, j_local, k, stype) in bot.springs[p_idx]:
                gi = dof[2*i_local] // 2
                gj = dof[2*j_local] // 2
                if any(set([gi, gj]) == set([a, b]) for a, b in spring_pairs):
                    continue
                spring_pairs.append((gi, gj))
                spring_colors.append(btype)

        contact_name = f"contact_{bot.name}"
        contact = system.contributions_map.get(contact_name, None)
        
        per_bot.append({
            "bot": bot,
            "bot_dofs": bot_dofs,
            "X_all": X_all,
            "Y_all": Y_all,
            "spring_pairs": spring_pairs,
            "spring_colors": spring_colors,
            "contact": contact
        })

    # create figure with grid of subplots
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("All Bots")

    Xg = np.concatenate([data["X_all"] for data in per_bot], axis=1)
    Yg = np.concatenate([data["Y_all"] for data in per_bot], axis=1)
    PAD = 0.05

    ax.set_xlim(np.min(Xg) - PAD, np.max(Xg) + PAD)
    ax.set_ylim(np.min(Yg) - PAD, np.max(Yg) + PAD)


    # create artist for the plot
    artists = []
    bot_art = []

    for data in per_bot:
        bot = data["bot"]

        scatter = ax.scatter([], [], s=20)
        spring_lines = [ax.plot([],[], '-', lw=1)[0] for _ in data["spring_pairs"]]
        com_marker, = ax.plot([],[],'o',color='#66CCEE',markersize=5)

        if data["contact"] is not None:
            pts = data["contact"].points
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, 'k-', lw=1.2)

        bot_art.append(dict(
            scatter = scatter,
            spring_lines = spring_lines,
            com_marker = com_marker,
            data = data
        ))

    # text with time
    time_text = fig.text(0.02, 0.97, '', fontsize=10)

    # precompute com
    bot_coms = []
    for b in per_bot:
        bot = b["bot"]
        bot_dofs = b["bot_dofs"]
        cx_list = []
        cy_list = []
        for k in range(len(t)):
            q_bot_frame = q[k, bot_dofs]
            cx, cy = bot.center_of_mass(t[k], q_bot_frame)
            cx += bot.start_p_position[0]
            cy += bot.start_p_position[1]
            cx_list.append(cx)
            cy_list.append(cy)
        bot_coms.append((np.array(cx_list), np.array(cy_list)))


    def init():
        out = []
        for ba in bot_art:
            ba["scatter"].set_offsets(np.c_[[], []])
            for ln in ba["spring_lines"]:
                ln.set_data([], [])
                out.append(ln)
            ba["com_marker"].set_data([], [])
            out.append(ba["scatter"])
            out.append(ba["spring_lines"])
            out.append(ba["com_marker"])
        time_text.set_text("")
        out.append(time_text)
        return out

    # animate
    def animate(frame):
        out = []
        for i, ba in enumerate(bot_art):
            data = ba["data"]
            X = data["X_all"][frame]
            Y = data["Y_all"][frame]
            
            gaps_bot = data["contact"].g_full(t[frame], q[frame, bot_dofs])
            

            colors = []
            for j in range(len(X)):
                gap_val = gaps_bot[j]
                if j in data["contact"].outer_masses and gap_val < 1e-3:
                    colors.append('#66CCEE')
                else:
                    colors.append('#4477AA')

        

            # update scatter
            ba["scatter"].set_offsets(np.c_[X, Y])
            ba["scatter"].set_color(colors)
            out.append(ba["scatter"])

            for (gi, gj), ln, ctype in zip(data["spring_pairs"], ba["spring_lines"], data["spring_colors"]):
                ln.set_data([X[gi], X[gj]], [Y[gi], Y[gj]])
                ln.set_color(bot_type_to_color.get(ctype, 'k'))
                out.append(ln)

            # update COM
            cx, cy = bot_coms[i][0][frame], bot_coms[i][1][frame]
            ba["com_marker"].set_data([cx], [cy])
            out.append(ba["com_marker"])

        time_text.set_text(f"t = {t[frame]:.3f} s")
        return out

    # Create animation
    desired_fps = 30
    animation_duration = t_final if t_final > 1.0 else t_final*2.5

    n_frames = int(animation_duration * desired_fps)
    n_frames = max(n_frames, 15)
    frame_indices = np.linspace(0, len(q)-1, n_frames, dtype=int)
    interval_ms = 1000/desired_fps
    ani = animation.FuncAnimation(
        fig, animate, frames=frame_indices, init_func=init,
        blit=False, interval=interval_ms
    )

    return plt.show()



def plot_omega_phi_over_comdx():
    # plot omega and phi over com dx

    # load all files
    results_folder = Path.cwd() / "solutions" / "worm_grid_search" / "flat_ground" / "loop_omega_phi"
    pkl_files = list(results_folder.glob("full_solution_of_sim_*_bots_*.pkl"))

    all_bot_id = []
    invalid_bot_id = []

    all_com_dx = []
    all_phi = []
    all_omega = []
    phi_invalid = []
    omega_invalid = []
    com_dx_invalid = []

    for pkl_file in pkl_files:
        full_sol_i = load_solution(pkl_file)
        system_i = full_sol_i.system
        t_i = full_sol_i.t
        q_i = full_sol_i.q

        bot_names_i = sorted([n for n in system_i.contributions_map if n.startswith("bot_")])
        bots_i = [system_i.contributions_map[n] for n in bot_names_i]


        for bot in bots_i:
            bot_id = bot.global_id

            bot_dofs = np.unique(bot.pixelDOF.flatten())
            comx_list = []
            comy_list = []
            for k in range(len(t_i)):
                q_frame = q_i[k, bot_dofs]
                cx, cy = bot.center_of_mass(t_i[k], q_frame)
                cx += bot.start_p_position[0]
                cy += bot.start_p_position[1]
                comx_list.append(cx)
                comy_list.append(cy)

            com_dx = comx_list[-1] - comx_list[0]

            phi_val = bot.pixel_prop[2]["phi"]
            omega_val = bot.pixel_prop[1]["omega"]

            final_y = comy_list[-1]

            if final_y >= 0.03:
                phi_invalid.append(phi_val)
                omega_invalid.append(omega_val)
                com_dx_invalid.append(com_dx)
                invalid_bot_id.append(bot_id)
            else:
                all_phi.append(phi_val)
                all_omega.append(omega_val)
                all_com_dx.append(com_dx)
                all_bot_id.append(bot_id)


    print(f"Loaded {len(all_com_dx)+len(com_dx_invalid)} bot results from {len(pkl_files)} simulations.")


    fig2, (ax_phi, ax_omega) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    sc_phi_valid = ax_phi.scatter(all_phi, all_com_dx, c='#4477AA', label='Valid')
    sc_phi_invalid = ax_phi.scatter(phi_invalid, com_dx_invalid, c='#EE6677', label='Invalid')
    ax_phi.set_ylabel('COM dx [m]')
    ax_phi.set_title('Phi vs COM displacement')
    ax_phi.legend()
    ax_phi.grid(True)

    sc_omega_valid = ax_omega.scatter(all_omega, all_com_dx, c='#4477AA', label='Valid')
    sc_omega_invalid = ax_omega.scatter(omega_invalid, com_dx_invalid, c='#EE6677', label='Invalid')
    ax_omega.set_xlabel('Omega value [rad/s]')
    ax_omega.set_title('Omega vs COM displacement')
    ax_omega.legend()
    ax_omega.grid(True)

    cursor = mplcursors.cursor([sc_phi_valid, sc_phi_invalid,
                                sc_omega_valid, sc_omega_invalid],
                                hover=True)
    
    @cursor.connect("add")
    def on_add(sel):
        ind = sel.index

        if sel.artist in [sc_phi_valid, sc_omega_valid]:
            bot_id = all_bot_id[ind]
            comdx  = all_com_dx[ind]
        else:
            bot_id = invalid_bot_id[ind]
            comdx  = com_dx_invalid[ind]

        sel.annotation.set_text(f"Bot ID: {bot_id}\nCOM dx: {comdx:.4f}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

    plt.tight_layout()
    return plt.show()


def results_table():

    flat_folder = Path.cwd() / "solutions" / "worm_grid_search" / "flat_ground" / "loop_omega_phi"
    other_folders = [
        # Path.cwd() / "solutions" / "worm_grid_search" / "slanted_down_ground" / "0001_010max(natural_omega)",
        # Path.cwd() / "solutions" / "worm_grid_search" / "slanted_up_ground" / "0001_010max(natural_omega)",
    ]

    # flat ground filtering
    valid_bots = {}  # bot_id -> dict with omega, phi, dx list
    flat_files = list(flat_folder.glob("full_solution_of_sim_0_bots_0-3.pkl"))

    comy_threshold = 0.02
    top_n = 30

    for pkl_file in flat_files:
        full_sol = load_solution(pkl_file)
        system = full_sol.system
        t = full_sol.t
        q = full_sol.q

        bot_names = sorted([n for n in system.contributions_map if n.startswith("bot_")])
        bots = [system.contributions_map[n] for n in bot_names]

        for bot in bots:
            bot_id = bot.global_id
            bot_dofs = np.unique(bot.pixelDOF.flatten())

            comx_list, comy_list = [], []
            for k in range(len(t)):
                q_bot = q[k, bot_dofs]
                cx, cy = bot.center_of_mass(t[k], q_bot)
                cx += bot.start_p_position[0]
                cy += bot.start_p_position[1]
                comx_list.append(cx)
                comy_list.append(cy)

            final_dx = comx_list[-1] - comx_list[0]

            comy_array = np.array(comy_list)

            # Reject bot if COM_y ever exceeds threshold
            if np.max(comy_array) > comy_threshold:
                continue

            

            # Omega and Phi for all actuated pixels
            pixel_keys = sorted(bot.pixel_prop.keys())
            phi_val = [bot.pixel_prop[k]["phi"] for k in pixel_keys]
            omega_val = [bot.pixel_prop[k]["omega"] for k in pixel_keys]

            if bot_id not in valid_bots:
                valid_bots[bot_id] = {"dx_list": [], "omega": omega_val, "phi": phi_val}
            valid_bots[bot_id]["dx_list"].append(final_dx)

    # other folders
    for folder in other_folders:
        pkl_files = list(folder.glob("full_solution_of_sim_*_bots_*.pkl"))
        for pkl_file in pkl_files:
            full_sol = load_solution(pkl_file)
            system = full_sol.system
            t = full_sol.t
            q = full_sol.q

            bot_names = sorted([n for n in system.contributions_map if n.startswith("bot_")])
            bots = [system.contributions_map[n] for n in bot_names]

            for bot in bots:
                bot_id = bot.global_id
                if bot_id not in valid_bots:
                    continue  # only consider flat-ground validated bots

                bot_dofs = np.unique(bot.pixelDOF.flatten())
                comx_list, comy_list = [], []
                for k in range(len(t)):
                    q_bot = q[k, bot_dofs]
                    cx, cy = bot.center_of_mass(t[k], q_bot)
                    cx += bot.start_p_position[0]
                    cy += bot.start_p_position[1]
                    comx_list.append(cx)
                    comy_list.append(cy)

                final_dx = comx_list[-1] - comx_list[0]
                final_comy = comy_list[-1]

                if final_comy < comy_threshold:
                    valid_bots[bot_id]["dx_list"].append(final_dx)

    # compute average dx
    avg_dx_info = []
    for bot_id, data in valid_bots.items():
        avg_dx = np.mean(data["dx_list"])
        avg_dx_info.append({
            "Bot ID": bot_id,
            "avg_dx": avg_dx,
            "omega": data["omega"],
            "phi": data["phi"]
        })

    # Sort descending by avg_dx
    avg_dx_info_sorted = sorted(avg_dx_info, key=lambda x: x['avg_dx'], reverse=True)#[:top_n]

    # print table
    header = f"{'Bot ID':>7} | {'Avg dx':>10} | {'Omega':>25} | {'Phi':>25}"
    print(header)
    print("-" * len(header))

    for entry in avg_dx_info_sorted:
        omega_str = ", ".join([f"{o:.2f}" for o in entry['omega']])
        phi_str = ", ".join([f"{p:.2f}" for p in entry['phi']])
        print(f"{entry['Bot ID']:7} | {entry['avg_dx']:10.5f} | {omega_str:25} | {phi_str:25}")



    return avg_dx_info_sorted


    
def heatmap():
    terrain_folders = {
        "Flat": Path.cwd() / "solutions" / "worm_grid_search" / "flat_ground" / "0001_010max(natural_omega)",
        "Inclined": Path.cwd() / "solutions" / "worm_grid_search" / "slanted_up_ground" / "0001_010max(natural_omega)",
        "Downhill": Path.cwd() / "solutions" / "worm_grid_search" / "slanted_down_ground" / "0001_010max(natural_omega)",
    }
    terrain_data = {}

    dy_threshold = 0.2


    for terrain_name, folder in terrain_folders.items():
        pkl_files = list(folder.glob("full_solution_of_sim_*_bots_*.pkl"))
        omega_list, phi_list, dx_list, valid_list = [], [], [], []

        for pkl_file in pkl_files:
            full_sol = load_solution(pkl_file)
            system = full_sol.system
            t = full_sol.t
            q = full_sol.q

            bot_names = sorted([n for n in system.contributions_map if n.startswith("bot_")])
            bots = [system.contributions_map[n] for n in bot_names]

            for bot in bots:
                bot_dofs = np.unique(bot.pixelDOF.flatten())
                comx_list, comy_list = [], []
                for k in range(len(t)):
                    q_bot = q[k, bot_dofs]
                    cx, cy = bot.center_of_mass(t[k], q_bot)
                    cx += bot.start_p_position[0]
                    cy += bot.start_p_position[1]
                    comx_list.append(cx)
                    comy_list.append(cy)

                comx_array = np.array(comx_list)
                comy_array = np.array(comy_list)

                dx = comx_array[-1] - comx_array[0]

                dt = np.diff(t)
                dy = np.diff(comy_array)/dt

                valid = not np.any(np.abs(dy) > dy_threshold)


                phi_val = bot.pixel_prop[2]["phi"]                                                        # CHANGE THIS WHEN LOOKING AT DIFFERENT BOT
                omega_val = bot.pixel_prop[1]["omega"]

                omega_list.append(omega_val)
                phi_list.append(phi_val)
                dx_list.append(dx)
                valid_list.append(valid)

        terrain_data[terrain_name] = {
            "omega": np.array(omega_list),
            "phi": np.array(phi_list),
            "dx": np.array(dx_list),
            "valid": np.array(valid_list)
        }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    worst_dx_grid = None

    for i, (terrain_name, data) in enumerate(terrain_data.items()):
        omega_unique = np.unique(data["omega"])
        phi_unique = np.unique(data["phi"])
        dx_grid = np.zeros((len(phi_unique), len(omega_unique)))

        for ix, omega_val in enumerate(omega_unique):
            for iy, phi_val in enumerate(phi_unique):
                # average dx for bots matching omega, phi
                mask = (data["omega"] == omega_val) & (data["phi"] == phi_val)
                if np.any(mask):
                    if np.any(~data["valid"][mask]):
                        dx_grid[iy, ix] = np.nan
                    else:
                        dx_grid[iy, ix] = np.mean(data["dx"][mask])
                else:
                    dx_grid[iy, ix] = np.nan  # empty

        if worst_dx_grid is None:
            worst_dx_grid = dx_grid.copy()
        else:
            worst_dx_grid = np.minimum(worst_dx_grid, dx_grid)
        
        dx_grid = np.ma.masked_invalid(dx_grid)

        im = axes[i].imshow(dx_grid, origin='lower', aspect='auto',
                            extent=[omega_unique.min(), omega_unique.max(),
                                    phi_unique.min(), phi_unique.max()],
                            cmap='autumn_r')
        axes[i].set_title(f"{terrain_name} terrain")
        axes[i].set_xlabel("Omega [rad/s]")
        axes[i].set_ylabel("Phi [rad]")
        fig.colorbar(im, ax=axes[i], label="COM dx [m]")

    # Worst-case heatmap
    dx_grid = np.ma.masked_invalid(dx_grid)
    im = axes[3].imshow(worst_dx_grid, origin='lower', aspect='auto',
                        extent=[omega_unique.min(), omega_unique.max(),
                                phi_unique.min(), phi_unique.max()],
                        cmap='autumn_r')
    axes[3].set_title("Worst-case COM dx")
    axes[3].set_xlabel("Omega [rad/s]")
    axes[3].set_ylabel("Phi [rad]")
    fig.colorbar(im, ax=axes[3], label="COM dx [m]")

    plt.tight_layout()
    plt.show()



    return

    



if __name__ == "__main__":
    # plot_all_sim()
    # plot_omega_phi_over_comdx()

    # results_table()
    heatmap()

