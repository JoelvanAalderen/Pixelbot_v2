#############
# description
#############
# Planar model of a Point Mass subject to gravity. 
# The Point Mass can come into contact with x-y-plane. help

from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete import PointMass, Box, Frame
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane
from cardillo.solver import Moreau

if __name__ == "__main__":
    ############
    # parameters
    ############

    # Mass of pointmass
    mass = 1.0

    # contact parameters
    e_N = 0.75  # restitution coefficient in normal direction
    e_F = 0.0  # restitution coefficient in tangent direction
    mu = 0  # frictional coefficient (no friction for point mass as it has a very small contact area)

    # gravitational acceleration
    g = np.array([0, 0, -9.81])

    # initial conditions
    r_OC0 = np.array([0, 0, 0.5])  # initial position of c.o.m.
    v_C0 = np.array([1, 0, 0])  # initial velocity of c.o.m.

    # simulation parameters
    t_final = 3  # final time

    # initialize system
    system = System()

    #################
    # assemble system
    #################

    # create pointmass
    q0 = r_OC0
    u0 = v_C0

    pointmass = PointMass(
        mass = mass,
        q0 = q0,
        u0 = u0,
        name = "pointmass",
    )

    # gravitational force for pointmass
    gravity = Force(pointmass.mass * g, pointmass, name="gravity")
    # add pointmass and gravitational force to system
    system.add(pointmass, gravity)

    # create floor (Box only for visualization purposes)
    floor = Box(Frame)(
        dimensions=[2, 2, 0.0001],
        name="floor",
    )

    # add contact between pointmass and floor
    pointmass2plane = Sphere2Plane(floor, pointmass, mu=mu, r=0.0, e_N=e_N, e_F=e_F)
    system.add(floor, pointmass2plane)

    # assemble system
    system.assemble()

    ############
    # simulation
    ############
    dt = 2.0e-3  # time step
    solver = Moreau(system, t_final, dt)  # create solver
    sol = solver.solve()  # simulate system

    # read solution
    t = sol.t  # time
    q = sol.q  # position coordinates
    u = sol.u  # velocity coordinates
    P_N = sol.P_N  # discrete percussions in normal direction
    P_F = sol.P_F  # discrete percussions of friction

    #################
    # post-processing
    #################

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))
    # plot time evolution for x-coordinate
    x = [pointmass.r_OP(ti, qi)[0] for ti, qi in zip(t, q[:, pointmass.qDOF])]
    ax[0, 0].plot(t, x, "-r", label="$x$")
    ax[0, 0].set_title("Evolution of horizontal position")
    ax[0, 0].set_xlabel("t")
    ax[0, 0].set_ylabel("x")
    ax[0, 0].grid()

    # plot time evolution for z-coordinate
    z = [pointmass.r_OP(ti, qi)[2] for ti, qi in zip(t, q[:, pointmass.qDOF])]
    ax[0, 1].plot(t, z, "-g", label="$z$")
    ax[0, 1].set_title("Evolution of height")
    ax[0, 1].set_xlabel("t")
    ax[0, 1].set_ylabel("z")
    ax[0, 1].grid()

    # plot time evolution of x-velocity
    v_x = [
        pointmass.v_P(ti, qi, ui)[0]
        for ti, qi, ui in zip(t, q[:, pointmass.qDOF], u[:, pointmass.uDOF])
    ]
    ax[1, 0].plot(t, v_x, "-r", label="$v_x$")
    ax[1, 0].set_title("Evolution of horizontal velocity")
    ax[1, 0].set_xlabel("t")
    ax[1, 0].set_ylabel("v_x")
    ax[1, 0].grid()

    # plot time evolution of z-velocity
    v_z = [
        pointmass.v_P(ti, qi, ui)[2]
        for ti, qi, ui in zip(t, q[:, pointmass.qDOF], u[:, pointmass.uDOF])
    ]
    ax[1, 1].plot(t, v_z, "-g", label="$z$")
    ax[1, 1].set_title("Evolution of vertical velocity")
    ax[1, 1].set_xlabel("t")
    ax[1, 1].set_ylabel("v_z")
    ax[1, 1].grid()

    # second figure plotting percussions
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    # plot time evolution for x-coordinate
    x = [pointmass.r_OP(ti, qi)[0] for ti, qi in zip(t, q[:, pointmass.qDOF])]
    ax[0, 0].plot(t, x, "-r", label="$x$")
    ax[0, 0].set_title("Evolution of horizontal position")
    ax[0, 0].set_xlabel("t")
    ax[0, 0].set_ylabel("x")
    ax[0, 0].grid()

    # plot time evolution for z-coordinate
    z = [pointmass.r_OP(ti, qi)[2] for ti, qi in zip(t, q[:, pointmass.qDOF])]
    ax[0, 1].plot(t, z, "-g", label="$z$")
    ax[0, 1].set_title("Evolution of height")
    ax[0, 1].set_xlabel("t")
    ax[0, 1].set_ylabel("z")
    ax[0, 1].grid()

    # plot time evolution of discrete normal percussion
    # TODO: How do we name this thing? incremental normal percussion?
    ax[1, 1].plot(t, P_N, "-g", label="$P_N$")
    ax[1, 1].set_title("Evolution of discrete normal percussion")
    ax[1, 1].set_xlabel("t")
    ax[1, 1].set_ylabel("P_N")
    ax[1, 1].grid()

    plt.tight_layout()
    plt.show()

    # vtk-export
    dir_name = Path(__file__).parent
    system.export(dir_name, "vtk", sol)