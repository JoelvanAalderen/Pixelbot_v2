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
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.interactions import TwoPointInteraction
from cardillo.contacts import Sphere2Plane
from cardillo.solver import ScipyIVP

if __name__ == "__main__":
    ############
    # parameters
    ############

    # Mass of pointmass
    mass = 1.0
    l0 = 1
    k = 100
    d = 0
    stretch = 1.5


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
    t0 = 0 # initial time
    t_final = 3  # final time

    # initialize system
    system = System(t0 = t0)

    #################
    # assemble system
    #################

    # point mass 1
    q10 = np.array([-0.5 * (stretch * l0), 0, 0])
    u10 = np.zeros(3)
    pointmass1 = PointMass(
        mass = mass,
        q0 = q10,
        u0 = u10,
        name = "pointmass 1",
    )

    # point mass 2
    q20 = np.array([0.5 * (stretch * l0), 0, 0])
    u20 = np.zeros(3)
    pointmass2 = PointMass(
        mass = mass,
        q0 = q20,
        u0 = u20,
        name = "pointmass 2",
    )

    spring = SpringDamper(
        TwoPointInteraction(pointmass1, pointmass2),
        k,
        d,
        l_ref = l0,
        name = "spring",
    )
    system.add(spring)

    

    # gravitational force for pointmass
    gravity1 = Force(pointmass1.mass * g, pointmass1, name="gravity1")
    gravity2 = Force(pointmass2.mass * g, pointmass2, name="gravity2")
    # add pointmass and gravitational force to system
    system.add(pointmass1, pointmass2, gravity1, gravity2)

    # create floor (Box only for visualization purposes)
    floor = Box(Frame)(
        dimensions=[5, 0.5, 0.0001],
        name="floor",
    )

    # add contact between pointmass and floor
    pointmass2plane1 = Sphere2Plane(floor, pointmass1, mu=mu, r=0.0, e_N=e_N, e_F=e_F)
    pointmass2plane2 = Sphere2Plane(floor, pointmass2, mu=mu, r=0.0, e_N=e_N, e_F=e_F)
    system.add(floor, pointmass2plane1, pointmass2plane2)

    # assemble system
    system.assemble()

    ############
    # simulation
    ############
    dt = 1.0e-2  # time step
    solver = ScipyIVP(system, t_final, dt)  # create solver
    sol = solver.solve()  # simulate system

    # read solution
    t = sol.t  # time
    q = sol.q  # position coordinates
    u = sol.u  # velocity coordinates

    #################
    # post-processing
    #################

      ##############
    # plot results
    ##############
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    # plot time evolution for x-coordinates
    x1 = [pointmass1.r_OP(ti, qi)[0] for ti, qi in zip(t, q[:, pointmass1.qDOF])]
    x2 = [pointmass2.r_OP(ti, qi)[0] for ti, qi in zip(t, q[:, pointmass2.qDOF])]
    ax[0, 0].plot(t, x1, "-r", label="$x_1$")
    ax[0, 0].plot(t, x2, "-g", label="$x_2$")
    ax[0, 0].set_title("Evolution of positions")
    ax[0, 0].set_xlabel("t")
    ax[0, 0].set_ylabel("x")
    ax[0, 0].legend()
    ax[0, 0].grid()

    # plot time evolution of elongation of SD-element
    l = [spring.l(ti, qi) for ti, qi in zip(t, q[:, spring.qDOF])]
    ax[0, 1].plot(t, l)
    ax[0, 1].set_title("Evolution of elongation of SD-element")
    ax[0, 1].set_xlabel("t")
    ax[0, 1].set_ylabel("length")
    ax[0, 1].grid()

    # plot time evolution of force of SD-element
    f = [
        spring.force(ti, qi, ui)
        for ti, qi, ui in zip(t, q[:, spring.qDOF], u[:, spring.uDOF])
    ]
    ax[1, 0].plot(t, f)
    ax[1, 0].set_title("Evolution of scalar force of SD-element")
    ax[1, 0].set_xlabel("t")
    ax[1, 0].set_ylabel("force")
    ax[1, 0].grid()

    # plot time evolution of energy
    # potential energy
    E_pot = np.array([system.E_pot(ti, qi) for ti, qi in zip(t, q)])
    # kinetic energy
    E_kin = np.array([system.E_kin(ti, qi, ui) for ti, qi, ui in zip(t, q, u)])
    ax[1, 1].plot(t, E_pot, label="$E_{pot}$")
    ax[1, 1].plot(t, E_kin, label="$E_{kin}$")
    ax[1, 1].plot(t, E_kin + E_pot, label="$E_{tot}$")
    ax[1, 1].set_title("Evolution of energies")
    ax[1, 1].set_xlabel("t")
    ax[1, 1].set_ylabel("energy")
    ax[1, 1].legend()
    ax[1, 1].grid()

    plt.tight_layout()
    plt.show()

    ############
    # VTK export
    ############
    path = Path(__file__)
    system.export(path.parent, "vtk", sol)
