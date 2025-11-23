
## Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from vtk import VTK_VERTEX
from pathlib import Path

from cardillo import System
from cardillo.solver import Moreau, SolverOptions, save_solution

from Multiplebots import MultiPixelbot
from Pixelbot_contact import Pixelbot_contact, mass_mass_contact


## Main code block, initializing code
if __name__ == "__main__":

    ## Ground points and vectors initialized
    def ground_func(xmin=-1.0, xmax=10.0, nrpoints=10):
        x_vals = np.linspace(xmin, xmax, nrpoints)
        angle = -10 #16.3 degrees is static for single pixel
        y_vals =  np.tan(np.deg2rad(angle))*x_vals # 0.0006 * np.sin(200 * x_vals)-0.0006 #
        points = [np.array([x, y]) for x, y in zip(x_vals, y_vals)]

        vectors = [points[i+1] - points[i] for i in range(len(points)-1)]

        #add two vertical contact surfaces (for climbing)
        # points = [
        #     np.array([0, 10]),
        #     np.array([0, -0.005]),
        #     np.array([0.05, -0.005]),
        #     np.array([0.05, 10])
        # ]
        # vectors = [
        #     points[1]-points[0],
        #     points[2]-points[1],
        #     points[3]-points[2]
        # ]
        
        return points, vectors


    ## Define grid of pixels
    # Placement = np.array([
    #     [7, 7, 2, 7, 7],
    #     [7, 7, 1, 7, 7],
    #     [6, 0, 0, 0, 6],
    #     [3, 0, 0, 0, 4],
    #     [7, 0, 0, 0, 7]
    # ], dtype=int)
    Placement = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 7, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 7, 7, 0]
    ], dtype=int)
    # Placement = np.array([
    #     [7, 8, 6, 2, 7],
    #     [7, 5, 6, 1, 7],
    #     [6, 0, 6, 0, 6],
    #     [3, 0, 4, 0, 3],
    #     [7, 0, 7, 0, 7]
    # ], dtype=int)
    # Placement = np.array([
    #     [0, 0, 0, 0, 0],
    #     [3, 6, 2, 6, 3],
    #     [3, 6, 1, 6, 3],
    #     [0, 0, 7, 0, 0],
    #     [4, 6, 1, 6, 4]
    # ], dtype=int)
    # Placement = np.array([
    #     [0, 6, 1, 6, 1],
    #     [6, 6, 2, 6, 1],
    #     [0, 7, 7, 0, 0],
    #     [3, 6, 7, 6, 2],
    #     [3, 6, 4, 6, 2],
    # ], dtype=int)


    ## Define pixel properties
    pixel_size = 0.01

    pixel_prop = {
        1: {"mass": 0.000852, "stiffness": 3400, "A": 0.6, "omega":  0.01*np.sqrt(3400/0.000852), "phi": 0}, #0.01*np.sqrt(3400/0.000852)
        2: {"mass": 0.000852, "stiffness": 3400, "A": 0.6, "omega":  0.01*np.sqrt(3400/0.000852), "phi": 1*np.pi},
        3: {"mass": 0.000852, "stiffness": 3400, "A": 0.6, "omega":  0.01*np.sqrt(3400/0.000852), "phi": 0.25*np.pi},
        4: {"mass": 0.000852, "stiffness": 3400, "A": 0.6, "omega":  0.01*np.sqrt(3400/0.000852), "phi": 1.25*np.pi},
        5: {"mass": 0.000852, "stiffness": 3400, "A": 0.6, "omega":  0.01*np.sqrt(3400/0.000852), "phi": 0.4*np.pi},
        8: {"mass": 0.000852, "stiffness": 3400, "A": 0.6, "omega":  0.01*np.sqrt(3400/0.000852), "phi": 1.4*np.pi},
        6: {"mass": 0.000972, "stiffness": 4200, "A": 0.0, "omega": 2*np.pi*0.4, "phi": 0},
        7: {"mass": 0.000989, "stiffness": 9200, "A": 0.0, "omega": 2*np.pi*0.4, "phi": 0},  
    }
    stiffness = [value["stiffness"] for value in pixel_prop.values()]
    mass = [value["mass"] for value in pixel_prop.values()]
    omega = [np.sqrt(s / m) for s, m in zip(stiffness, mass)]
    period = 2*np.pi/min(omega)    

    dt = (2*np.pi/max(omega))/20
    print(dt)
    t_final = 200*(2*np.pi/min(omega)) # change back to 400 after testing
    print(t_final)

    start_p_position = (0.0, 0.0)
    start_p_angle = np.deg2rad(0)
    bot_name = "bot1"

    ## Run combined system
    MultiPixel = MultiPixelbot(
        bot_name,
        Placement,
        pixel_size,
        pixel_prop,
        period,
        start_p_position,
        start_p_angle
        )
    
    Ground_contact = Pixelbot_contact(
        MultiPixel, 
        ground_func,
        offset=[0,0],
        name="contact_ground"
    )

    mass_mass_contact = mass_mass_contact(
        MultiPixel,
        name="contact_mass_mass"
    )
    
    print("Global masses (index: [x,y]):")
    for idx, coord in enumerate(MultiPixel.global_coords):
        print(f"{idx} : {coord}")
    print("\nBot-to-global mapping (counter-clockwise starting bottom-left):")
    for i, indices in enumerate(MultiPixel.bot_global_mass):
        print(f"Bot {i+1}: {indices}")
    print("pixelDOF shape:", MultiPixel.pixelDOF.shape)
    print(MultiPixel.pixelDOF)
    M_global = MultiPixel.M(0, np.zeros(MultiPixel.nq))
    print("Global mass matrix M (shape {}):\n".format(M_global.shape), M_global) 


    print("\n=== MASS TRIOS ===")
    print("Total:", len(mass_mass_contact.mass_trios))
    print("\n=== MASS–MASS CONTACT GAP VALUES AT t = 0 ===")

    q0 = MultiPixel.q0.copy()

    gaps = mass_mass_contact.g_N(0.0, q0)

    for k, (i, j1, j2) in enumerate(mass_mass_contact.mass_trios):
        gi = i
        gj1 = j1
        gj2 = j2
        print(f"{k:3d} : trio({gi},{gj1},{gj2})   gap = {gaps[k]: .6e}")

    print("================================================\n")

    ## set up system and solver
    system = System()
    system.add(MultiPixel, Ground_contact)#, mass_mass_contact)
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))   
                                                       
    solver = Moreau(system, t_final, dt)                                  
    sol = solver.solve()                                                    

    # read solution
    t = sol.t                                                               
    q = sol.q                                                           
    u = sol.u        

    path = Path(__file__)
    save_solution(sol, Path(path.parent, "testfile.pkl"))

    ## Animation for MultiPixelbot
    # Identify unique spring pairs
    spring_pairs = []
    spring_colors = []
    for p_idx, dof in enumerate(MultiPixel.pixelDOF):
        bot_type = MultiPixel.bot_types[p_idx]
        for (i, j, k, spring_type) in MultiPixel.springs[p_idx]:
            gi, gj = dof[2*i]//2, dof[2*j]//2
            if any(set([gi, gj]) == set([g0, g1]) for g0, g1, *_ in spring_pairs):
                continue
            spring_pairs.append((gi, gj))
            spring_colors.append(bot_type)

    bot_type_to_color = {1: '#4477AA', 2: '#CCBB44', 3: '#228833', 4: '#EE6677', 5: '#AA3377', 6: '#BBBBBB', 7: '#000000', 8: 'pink'}


    X0 = MultiPixel.q0[0::2]
    Y0 = MultiPixel.q0[1::2]
    x_center, y_center = np.mean(X0), np.mean(Y0)
    bot_radius = max(np.ptp(X0), np.ptp(Y0)) / 2 + MultiPixel.pixel_size

    ## Collect com position over the sim time
    com_x = []
    com_y = []

    for frame in range(len(t)):
        q_frame = q[frame]
        cx, cy = MultiPixel.center_of_mass(t[frame], q_frame)
        com_x.append(cx)
        com_y.append(cy)

    com_x = np.array(com_x)
    com_y = np.array(com_y)


    # Plot setup
    fig, axes = plt.subplots(
    3, 1, 
    figsize=(4, 12),           
    gridspec_kw={'height_ratios': [3, 1, 1]}
    )
    ax_main = axes[0]
    ax_com = axes[1]
    ax_actuation = axes[2]

    ax_main.set_aspect("equal")
    ax_main.set_xlabel("x")
    ax_main.set_ylabel("y")
    ax_main.set_xlim(x_center - bot_radius*2-min(com_x), x_center + bot_radius*2+max(com_x))
    ax_main.set_ylim(y_center - bot_radius*2-min(com_y), y_center + bot_radius*2+max(com_y))

    ax_com.plot(t, com_x, '-', color='#4477AA', label='COM x')
    ax_com.plot(t, com_y, '-', color='#AA3377', label='COM y')
    ax_com.set_xlabel("Time [s]")
    ax_com.set_ylabel("COM position [m]")
    ax_com.set_title('COM trajectory over time')
    ax_com.legend()
    ax_com.grid(True)

    L_avg = {}
    for ptype, L_list in MultiPixel.L_history.items():
        n_pixels = sum(1 for p in MultiPixel.bot_types if p == ptype)
        values_per_step = n_pixels * 2
        if len(L_list) == 0:
            continue
        L_array = np.array(L_list).reshape(-1, values_per_step)
        L_avg[ptype] = L_array.mean(axis=1)
        t_values = t[:len(L_avg[ptype])]
        ax_actuation.plot(t_values, L_avg[ptype], label=f"Pixel type {ptype}")
    ax_actuation.set_xlabel("Time [s]")
    ax_actuation.set_ylabel("Horizontal length of a pixel L(t) [m]")
    ax_actuation.set_title('COM trajectory over time')
    ax_actuation.legend()
    ax_actuation.grid(True)

    plt.tight_layout()

    # Plot ground
    xmin, xmax = ax_main.get_xlim()
    points, vectors = ground_func(xmin, xmax)
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    ax_main.plot(x_vals, y_vals, "k-", linewidth=1.5)

    # Points and springs for plotting
    points_plot = ax_main.scatter([], [], s=20)
    spring_lines = [ax_main.plot([], [], "r-", lw=1.0)[0] for _ in spring_pairs]

    time_text = ax_main.text(
        0.02, 0.95, '', transform=ax_main.transAxes,
        fontsize=10, color='#000000', ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
    )

    com_point, = ax_main.plot([], [], 'o', color='#4477AA', markersize='4', label='COM')

    # Initialization
    def init():
        points_plot.set_offsets(np.c_[[], []])
        for line in spring_lines:
            line.set_data([], [])
        return [points_plot, *spring_lines]


    # Animation function
    def animate(frame):
        q_frame = q[frame]  # shape (nq,)
        
        # Extract global positions from q_frame
        X = q_frame[0::2]
        Y = q_frame[1::2]
        
        gaps = Ground_contact.g_full(t[frame], q_frame)
        gaps_mass = mass_mass_contact.g_full(t[frame], q_frame)
        colors = ['#4477AA']*len(X)
        
        for idx, g in enumerate(gaps_mass):
            if g < 1e-3:
                    colors[idx] = '#66CCEE'
                    # colors[j1] = colors[j2] = "#FF0000"

         
        for idx, g in enumerate(gaps):
            if g < 1e-3:
                colors[idx] = '#66CCEE'

        points_plot.set_offsets(np.c_[X, Y])
        points_plot.set_color(colors)

        # Draw springs
        for line, (gi, gj), btype in zip(spring_lines, spring_pairs, spring_colors):
            line.set_data([X[gi], X[gj]], [Y[gi], Y[gj]])
            line.set_color(bot_type_to_color[btype])

        comx, comy = MultiPixel.center_of_mass(t[frame], q_frame)
        com_point.set_data([comx], [comy])

        time_text.set_text(f't = {t[frame]:.3f} s')

           # ===== DEBUG: Mass trio 4 =====
        i, j1, j2 = mass_mass_contact.mass_trios[4]

        pi  = np.array([X[i],  Y[i]])
        pj1 = np.array([X[j1], Y[j1]])
        pj2 = np.array([X[j2], Y[j2]])

        # draw the three points
        ax_main.scatter([pi[0], pj1[0], pj2[0]],
                        [pi[1], pj1[1], pj2[1]],
                        s=80, c=["red","orange","yellow"])

        # draw the edge j1–j2
        ax_main.plot([pj1[0], pj2[0]],
                    [pj1[1], pj2[1]],
                    "g-", lw=2)

        # closest point projection
        d = pj2 - pj1
        s = np.dot(pi - pj1, d) / (np.dot(d, d) + 1e-12)
        s = np.clip(s, 0, 1)
        p_cl = pj1 + s * d

        # line from i to closest point
        ax_main.plot([pi[0], p_cl[0]],
                    [pi[1], p_cl[1]],
                    "m--", lw=1.5)

        # normal
        n = np.array([-d[1], d[0]])
        n /= (np.linalg.norm(n) + 1e-12)
        nlen = 0.05 * MultiPixel.pixel_size
        ax_main.plot([p_cl[0], p_cl[0] + n[0]*nlen],
                    [p_cl[1], p_cl[1] + n[1]*nlen],
                    "c-", lw=2)
        # ===============================

        return [points_plot, *spring_lines, time_text, com_point]

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

    plt.show()