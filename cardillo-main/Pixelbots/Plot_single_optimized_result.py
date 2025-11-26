 
## Import libraries
from matplotlib import animation, pyplot as plt
import numpy as np
from pathlib import Path
from cardillo.solver import load_solution


# Load the solution
cwd = Path.cwd()
pkl_path = cwd / "solutions" / "worm_grid_search" / "flat_ground" / "loop_omega_phi" / "full_solution_of_sim_6_bots_76-87.pkl"

full_sol = load_solution(pkl_path)
system = full_sol.system
MultiPixel = full_sol.system.contributions_map["bot_80"]
contact_name = f"contact_{MultiPixel.name}"
contact = full_sol.system.contributions_map[contact_name]

bot_dofs = np.unique(MultiPixel.pixelDOF.flatten()) 
q_bot = full_sol.q[:, bot_dofs]

t = full_sol.t
q = full_sol.q
u = full_sol.u
t_final = t[-1]


## Animation for MultiPixelbot
X0 = MultiPixel.q0[0::2][None, :] + q[:, bot_dofs][:, 0::2]
Y0 = MultiPixel.q0[1::2][None, :] + q[:, bot_dofs][:, 1::2]
bot_radius = max(np.ptp(X0), np.ptp(Y0)) / 2 + MultiPixel.pixel_size
x_min, x_max = np.min(X0), np.max(X0)
y_min, y_max = np.min(Y0), np.max(Y0)

## Collect com position over the sim time
com_x = []
com_y = []

for frame in range(len(t)):
    q_frame = q[frame]
    q_bot = q_frame[bot_dofs]
    cx, cy = MultiPixel.center_of_mass(t[frame], q_bot)
    com_x.append(cx)
    com_y.append(cy)

com_x = np.array(com_x)
com_y = np.array(com_y)


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
ax_main.set_xlim(x_min - bot_radius - min(com_x), x_max + bot_radius + max(com_x))
ax_main.set_ylim(y_min - bot_radius - min(com_y), y_max + bot_radius + max(com_y))

ax_com.plot(t, com_x, '-', color='#4477AA', label='COM x')
ax_com.plot(t, com_y, '-', color='#AA3377', label='COM y')
ax_com.set_xlabel("Time [s]")
ax_com.set_ylabel("Relative COM position [m]")
ax_com.set_title('Relative COM trajectory over time')
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
points, vectors = contact.points, contact.vectors
start_x, start_y = MultiPixel.start_p_position
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

com_point, = ax_main.plot([], [], 'o', color='#66CCEE', markersize='4', label='COM')

# Initialization
def init():
    points_plot.set_offsets(np.c_[[], []])
    for line in spring_lines:
        line.set_data([], [])
    return [points_plot, *spring_lines]


# Animation function
def animate(frame):
    q_frame = q[frame]  # shape (nq,)
    q_bot = q_frame[bot_dofs]
    # Extract global positions from q_frame
    X = MultiPixel.start_p_position[0] + q_bot[0::2]
    Y = MultiPixel.start_p_position[1] + q_bot[1::2]
    
    gaps = contact.g_full(t[frame], q_bot)
    # print(f"Frame {frame}: gaps shape = {gaps.shape}, min gap = {gaps.min()}, max gap = {gaps.max()}")
    colors = ['#66CCEE' if g < 1e-3 else '#4477AA' for g in gaps]

    points_plot.set_offsets(np.c_[X, Y])
    points_plot.set_color(colors)

    # Draw springs
    for line, (gi, gj), btype in zip(spring_lines, spring_pairs, spring_colors):
        line.set_data([X[gi], X[gj]], [Y[gi], Y[gj]])
        line.set_color(bot_type_to_color[btype])

    comx, comy = MultiPixel.center_of_mass(t[frame], q_bot)
    comx+= MultiPixel.start_p_position[0]
    comy+= MultiPixel.start_p_position[1]
    com_point.set_data([comx], [comy])

    time_text.set_text(f't = {t[frame]:.3f} s')

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


omega_values = [round(int(MultiPixel.pixel_prop[j]["omega"])) for j in MultiPixel.pixel_prop]
print("Omega values:", omega_values)

phi_values = [round(int(MultiPixel.pixel_prop[j]["phi"])) for j in MultiPixel.pixel_prop]
print("Phi values:", phi_values)


plt.show()