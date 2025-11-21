
## Import libraries
import numpy as np
from Pixelbot_contact import Pixelbot_contact


## Multiple bots class
class MultiPixelbot:
    def __init__(self, bot_name, Placement, pixel_size, pixel_prop, period, start_p_position=None, start_p_angle=None):
        # add arguments Placement, angle, pixel_prop, position(from ground)

        # define variables
        self.name = bot_name

        # save inputs
        self.Placement = np.array(Placement)
        self.pixel_size = pixel_size
        self.pixel_prop = pixel_prop
        self.start_p_position = np.zeros(2, dtype=float) if start_p_position is None else np.asarray(start_p_position, dtype=float)
        if start_p_angle is None:
            self.start_p_angle = 0.0
        else:
            self.start_p_angle = float(start_p_angle) #in radians

        # count pixel types
        values, counts = np.unique(self.Placement, return_counts=True)
        count_dict = dict(zip(values, counts))

        self.period = period

        # build mapping from placement to global_coords, bot_global_mass and bot_types
        self.global_coords, self.bot_global_mass, self.bot_types = self._pixel_mapping(self.Placement, self.pixel_size, self.start_p_position, self.start_p_angle)

        # global masses
        self.global_masses = [np.array(p) for p in self.global_coords]
        self.total_masses = len(self.global_masses)
        self.nq = 2*self.total_masses
        self.nq_local = 8

        # q_g and initial states
        self.q_g = self.global_q(self.global_coords, self.nq)
        self.q0 = self.q_g.copy()
        self.u0 = np.zeros_like(self.q_g)
        self.nu = self.nq

        self.gravity = np.array([0, -9.81])

        # build pixelDOF (attribute)
        self.pixelDOF = self._build_pixelDOF(self.bot_global_mass)
        self.nr_pixels = len(self.bot_global_mass)   

        # per-pixel parameters arrays (indexed per pixel index)
        self.A = [self.pixel_prop[t]["A"] for t in self.bot_types]
        self.omega = [self.pixel_prop[t]["omega"] for t in self.bot_types]
        self.phi = [self.pixel_prop[t]["phi"] for t in self.bot_types]               
        self.pixel_m = [[self.pixel_prop[t]["mass"]/4] * 4 for t in self.bot_types]
        self.pixel_k = [self.pixel_prop[t]["stiffness"] for t in self.bot_types]
        self.pixel_type = list(self.bot_types)

        # build C_list (local-to-global characterization) and q_l_list and M_l_list
        self.C_list = self._build_C_list(self.bot_global_mass, self.nq)
        self.q_l_list = [C @ self.q_g for C in self.C_list]
        self.M_l_list = []
        for p_idx, ptype in enumerate(self.bot_types):
            props = self.pixel_prop[ptype]
            masses = [props["mass"]/4] * 4
            M_l = np.diag(np.repeat(masses, 2))
            self.M_l_list.append(M_l)

        # define reference spring connectivity template (same for all pixel types)
        springs_template = [
            (0, 1, None, "straight"),   # stiffness filled later
            (1, 2, None, "straight"),
            (2, 3, None, "straight"),
            (3, 0, None, "straight"),
            (0, 2, None, "diagonal"),
            (1, 3, None, "diagonal")
        ]

        # fill stiffness per pixel
        self.springs = []
        for p_idx, ptype in enumerate(self.bot_types):
            k = self.pixel_prop[ptype]["stiffness"]
            this_pixel_springs = [(i, j, k, stype) for (i, j, _, stype) in springs_template]
            self.springs.append(this_pixel_springs)

        # build pixel_springs (local C matrices + parameters) and global_springs (unique pairs)
        self.pixel_springs = [[] for _ in range(self.nr_pixels)]
        self.global_springs = []

        def local_C_spring(i_local, j_local):
            C_spring_local = np.zeros((2, 8))
            C_spring_local[:, 2*i_local:2*i_local+2] = -np.eye(2)
            C_spring_local[:, 2*j_local:2*j_local+2] = np.eye(2)
            return C_spring_local   

        for p_idx, dof in enumerate(self.pixelDOF):     
            # dof is length-8 array of global DOF indices
            for (i_local, j_local, k, spring_type) in self.springs[p_idx]:
                gi = dof[2*i_local] // 2
                gj = dof[2*j_local] // 2
                # compute initial distance from q_g
                pos_i = self.q_g[2*gi:2*gi+2]
                pos_j = self.q_g[2*gj:2*gj+2]
                l0 = np.linalg.norm(pos_j - pos_i)

                self.global_springs.append((gi, gj, k, spring_type, l0))

                # local C matrix (maps local 8 -> spring relative 2)
                C_local = local_C_spring(i_local, j_local)
                self.pixel_springs[p_idx].append((C_local, k, spring_type, l0))
 

        # storing spring length for plotting
        self.L_history = {ptype: [] for ptype in np.unique(self.bot_types)}     


    ## Builder functions for attributes
    def _pixel_mapping(self, Placement, pixel_size, spawn_position, spawn_angle):
        global_coords = []
        coord_to_index = {}
        bot_global_mass = []
        bot_types = []

        rows, cols = Placement.shape
        cosA, sinA = np.cos(spawn_angle), np.sin(spawn_angle)
        R = np.array([[cosA, -sinA],
                      [sinA, cosA]])

        for r in range(rows):
            for c in range(cols):
                if Placement[r, c] == 0:
                    continue

                x = c * pixel_size
                y = (rows - 1 - r) * pixel_size
                corners = [
                    np.array([x, y]),
                    np.array([x + pixel_size, y]),
                    np.array([x + pixel_size, y + pixel_size]),
                    np.array([x, y + pixel_size])
                ]

                corners = [(R @ corner) + spawn_position for corner in corners]

                for corner in corners:
                    key = tuple(np.round(corner, 8))
                    if key not in coord_to_index:
                        coord_to_index[key] = len(global_coords)
                        global_coords.append(np.array(corner))

        global_coords = np.array(global_coords)
        if len(global_coords) == 0:
            return global_coords, [], []

        # Sort for determinism
        sorted_indices = np.lexsort((global_coords[:, 0], -global_coords[:, 1]))
        global_coords = global_coords[sorted_indices]
        coord_to_index = {tuple(np.round(coord, 8)): idx for idx, coord in enumerate(global_coords)}

        # iterate again to build bot-to-global mapping
        for r in range(rows):
            for c in range(cols):
                if Placement[r, c] == 0:
                    continue
                x = c * pixel_size
                y = (rows - 1 - r) * pixel_size
                corners = [
                    np.array([x, y]),
                    np.array([x + pixel_size, y]),
                    np.array([x + pixel_size, y + pixel_size]),
                    np.array([x, y + pixel_size])
                ]

                corners = [(R @ corner) + spawn_position for corner in corners]

                indices = [coord_to_index[tuple(np.round(corner, 8))] for corner in corners]
                bot_global_mass.append(indices)
                bot_types.append(Placement[r, c])

        return global_coords, bot_global_mass, bot_types
    

    def global_q(self, global_coords, nq):
        q_g = np.zeros(nq)
        for i in range(len(global_coords)):
            q_g[2*i:2*i+2] = global_coords[i]
        return q_g


    def _build_pixelDOF(self, bot_global_mass):
        nr_pixels = len(bot_global_mass)
        pixelDOF = np.zeros((nr_pixels, 8), dtype=int)
        for p_idx, global_masses in enumerate(bot_global_mass):
            DOF_indices = []
            for mass_idx in global_masses:
                DOF_indices.extend([2*mass_idx, 2*mass_idx + 1])
            pixelDOF[p_idx, :] = DOF_indices
        return pixelDOF
    

    def _build_C_list(self, bot_global_mass, nq):
        C_list = []
        for b in range(len(bot_global_mass)):
            C = np.zeros((8, nq))
            for local_idx in range(4):
                global_idx = bot_global_mass[b][local_idx]
                C[2*local_idx:2*local_idx+2, 2*global_idx:2*global_idx+2] = np.eye(2)
            C_list.append(C)
        return C_list


    ## Solver interface methods
    def M(self, t, q):
        M_g = np.zeros((self.nq, self.nq))

        for p_idx, M_l in enumerate(self.M_l_list):
            dof = self.pixelDOF[p_idx,:]
            M_g[np.ix_(dof,dof)] += M_l
    
        return M_g


    def q_dot(self, t, q, u): 
        return u                                                             


    def q_dot_u(self, t, q):
        return np.eye(self.nq)                                             
    

    ## Define kinematics for the solver
    def E_kin(self, t, q, u):
        M_g = self.M(t,q)
        return 0.5 * u @ (M_g @ u)
    
    def smooth_actuation(self, t, period):
        if t < 25*period:
            return t/(25*period)
        else:
            return 1.0

    ## Define potential energy for the solver
    def E_pot(self, t, q, pixel=None):
        # gravitational potential
        gy = self.gravity[1]
        M_g = self.M(t,q)
        masses = np.diag(M_g)[::2]
        ys = q[1::2]
        E_pot_gravity = (-gy) * np.sum(masses * ys)


        # elastic potential
        E_pot_elastic = 0
        A_pixel = self.A[pixel]
        omega_pixel = self.omega[pixel]
        phi_pixel = self.phi[pixel]
        mass_pixel = self.pixel_m[pixel]

        for (gi, gj, k, spring_type, l0) in self.global_springs:
            d = q[2*gj:2*gj+2] - q[2*gi:2*gi+2]  # displacement vector between the two masses connected by the spring
            l = np.linalg.norm(d)
            if l <= 1e-12: 
                continue

            period = self.period
            smooth_act = self.smooth_actuation(t,period)

            if spring_type == 'straight':
                l_t = l0
            elif spring_type == 'diagonal':
                l_t = l0 * (1 - smooth_act * A_pixel * np.sin(omega_pixel * t - phi_pixel))
            else:
                raise ValueError("Unknown spring type: {}".format(spring_type))
        
            E_pot_elastic += 0.5 * k * (l - l_t)**2

        return E_pot_gravity + E_pot_elastic
    

    ## Define h for a single pixel
    def h_pixel(self, t, q_pixel, u_pixel, pixel):

        h_loc = np.zeros((8,))
        m_pixel = self.pixel_prop[self.bot_types[pixel]]["mass"]
        # masses = self.pixel_m[pixel]

        A_pixel = self.A[pixel]
        omega_pixel = self.omega[pixel]
        phi_pixel = self.phi[pixel]
        mass_pixel = self.pixel_m[pixel]

        for i_local in range(4):
            h_loc[2*i_local + 1] += (m_pixel/4) * self.gravity[1]

        # spring forces
        for (C_local, k, spring_type, l0) in self.pixel_springs[pixel]: # in range of 6
            d = C_local @ q_pixel
            l = np.linalg.norm(d)
            if l <= 1e-12:
                continue
            dir_vec = d / l

            period = self.period
            smooth_act = self.smooth_actuation(t,period)

            if spring_type == "straight":
                l_t = l0
            elif spring_type == "diagonal":
                l_t = l0 * (1 - smooth_act * A_pixel * np.sin(omega_pixel * t - phi_pixel))
                ptype = self.bot_types[pixel]
                self.L_history[ptype].append(l_t)
            else:
                raise ValueError(f"Unknown spring type: {spring_type}")

            F = -k * (l - l_t) * dir_vec
            h_loc += C_local.T @ F

        return h_loc
    

    ## Define global h by summing local h
    def h(self, t, q, u):
        h = np.zeros((self.nq))

        for pixel in range(self.nr_pixels):
            pixelDOF = self.pixelDOF[pixel] # matrix of nr of pixels x 8  dtype=int
            q_pixel = q[pixelDOF] # pixelDOF is a matrix 
            u_pixel = u[pixelDOF] # u_pixel 
            h[pixelDOF] += self.h_pixel(t,q_pixel,u_pixel,pixel)
        return h


    ## Define h_q for a single pixel
    def h_q_pixel(self, t, q_pixel, pixel):
        H = np.zeros((8, 8))  # local 8x8
        A_pixel = self.A[pixel]
        omega_pixel = self.omega[pixel]
        phi_pixel = self.phi[pixel]
        mass_pixel = self.pixel_m[pixel]

        for (C_local, k, spring_type, l0) in self.pixel_springs[pixel]:
            d = C_local @ q_pixel
            l = np.linalg.norm(d)
            if l <= 1e-12:
                continue
            dir_vec = d / l

            period = self.period
            smooth_act = self.smooth_actuation(t,period)

            # diagonal spring actuation
            if spring_type == "straight":
                l_t = l0
            elif spring_type == "diagonal":
                l_t = l0 * (1 - smooth_act * A_pixel * np.sin(omega_pixel * t - phi_pixel))
            else:
                raise ValueError(f"Unknown spring type: {spring_type}")

            # Jacobian formula
            I = np.eye(2)
            outer = np.outer(dir_vec, dir_vec)
            coeff = (l - l_t) / l if l != 0 else 0
            J = -k * (outer + coeff * (I - outer)) @ C_local  # 2x8

            # assemble into local H (2x8 into 8x8)
            H += C_local.T @ J

        return H


    ## Define global h_q by summing local h_q
    def h_q(self, t, q, u=None):
        H_global = np.zeros((self.nq, self.nq))

        for pixel in range(self.nr_pixels):
            pixel_dof = self.pixelDOF[pixel]
            q_pixel = q[pixel_dof]
            H_local = self.h_q_pixel(t, q_pixel, pixel)
            # map to global
            for i_local, i_global in enumerate(pixel_dof):
                for j_local, j_global in enumerate(pixel_dof):
                    H_global[i_global, j_global] += H_local[i_local, j_local]

        return H_global
    
        
    # define mass-mass contact
    def same_pixel(self, i, j):
        for pixel_masses in self.bot_global_mass:
            if i in pixel_masses and j in pixel_masses:
                return True
        return False
    
    def neighbor_pixel(self, i, j):
        pixels_i = [set(p) for p in self.bot_global_mass if i in p]
        pixels_j = [set(p) for p in self.bot_global_mass if j in p]

        for pi in pixels_i:
            for pj in pixels_j:
                if pi is pj:
                    continue
                if len(pi & pj) >= 2:
                    return True
        return False
    
    def outer_mass(self):
        count = np.zeros(self.total_masses, dtype=int)
        for pixel_masses in self.bot_global_mass:
            for m in pixel_masses:
                count[m] += 1
        outer = np.where(count<=3)[0]
        return outer
    

    def center_of_mass(self, t, q):
        M_g = self.M(t,q)
        masses = np.diag(M_g)[::2]
        xs = q[0::2]
        ys = q[1::2]

        comx = (np.sum(masses*xs)/np.sum(masses))
        comy = (np.sum(masses*ys)/np.sum(masses))
        return comx, comy
