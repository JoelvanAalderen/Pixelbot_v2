
## Import libraries
import numpy as np
from cardillo.math.prox import Sphere
from scipy.spatial import ConvexHull


## Pixelbot contact class
class Pixelbot_contact:   
    def __init__(self, MultiPixelbot, ground_func, name):
        self.name = name

        # define necessary variables from pixelbot class
        self.MultiPixelbot = MultiPixelbot
        self.nr_of_masses = MultiPixelbot.total_masses
        self._nq = MultiPixelbot.nq
        self._nu = MultiPixelbot.nu

        # outer masses
        self.outer_masses = MultiPixelbot.outer_mass()
        self.nr_contact = len(self.outer_masses)

        # define contact variables for solver
        self.nla_N = self.nr_contact
        self.nla_F = self.nr_contact
        self.mu = 0.3 # 0 < mu < 1, 0 no friction
        self.e_N = np.full(self.nr_contact, 0.5) # 0 < eN < 1, 1 perfectly elastic (no energy loss)
        self.e_F = np.full(self.nr_contact, 0.0) # 0 < eF < 1, 1 perfectly elastic (no energy loss)

        # coulomb friction laws
        self.friction_laws = [
            ([i],[i], Sphere(self.mu)) for i in range(self.nr_contact)
            ]
        
        # define ground contact
        points, vectors = ground_func()
        self.points = [np.array(p) for p in points]
        self.vectors = [np.array(v) for v in vectors]
        self.closest_normals = [np.array([0.0, 1.0]) for _ in range(self.nr_contact)]

        self.contact_radius = 5e-2 * MultiPixelbot.pixel_size   # radius of the contact sphere around each mass

    def assembler_callback(self):
        self.qDOF = self.MultiPixelbot.qDOF
        self.uDOF = self.MultiPixelbot.uDOF


    ## Gap function
    def g_N(self, t, q):                                                      
        gaps = np.full(self.nla_N, np.inf, dtype=float)

        for k, i in enumerate(self.outer_masses):
            p_mass = np.asarray(q[2*i:2*i+2], dtype=float)
            best_dist = np.inf
            best_gap = np.inf
            best_normal = np.array([0.0, 1.0], dtype=float)

            for p0, v in zip(self.points, self.vectors):
                seg_len = np.linalg.norm(v)
                v_unit = v / seg_len              # tangent to the line
                normal = np.array([-v_unit[1], v_unit[0]], dtype=float)              # normal to the line
                normal /= np.linalg.norm(normal)

                proj = np.dot((p_mass - p0), v_unit)
                proj_clamped = np.clip(proj, 0.0, seg_len)               # projection of mass on segment
                p_closest = p0 + proj_clamped * v_unit                # closest point on segment 

                d_vec = p_mass - p_closest              # direction vector from mass to segment
                dist = np.linalg.norm(d_vec)                # distance to segment
                gap_val = np.dot(d_vec, normal) - self.contact_radius

                if dist < best_dist:
                    best_gap = gap_val           # gap along normal
                    best_normal = normal
                    best_dist = dist

            gaps[k] = best_gap
            self.closest_normals[k] = best_normal
        return gaps
    

    ## Define jacobian matrix for solver
    def W_N(self, t, q):                                                       
        WN = np.zeros((self._nq, self.nla_N), dtype=float)
        self.g_N(t, q)

        for k, i in enumerate(self.outer_masses):
            normal = self.closest_normals[k]
            WN[2*i, k] = normal[0]
            WN[2*i+1, k] = normal[1]
        return WN

    
    ## Define gap derivative for solver
    def g_N_dot(self, t, q, u):                                               
        return self.W_N(t, q).T @ u
    

    ## Define tangential contact
    ## Define tangential jacobian matrix for solver
    def W_F(self, t, q):
        WF = np.zeros((self._nq, self.nla_N), dtype=float)
        self.g_N(t, q)

        for k, i in enumerate(self.outer_masses):
            normal = self.closest_normals[k]
            tangent = np.array([normal[1], -normal[0]])
            tangent /= np.linalg.norm(tangent)
            WF[2*i, k] = tangent[0]
            WF[2*i+1,k] = tangent[1]
        return WF
    

    ## Define tangential relative velocity
    def gamma_F(self, t, q, u):
        return self.W_F(t,q).T @ u
    
    ## add full g_N for animation
    def g_full(self, t, q):
        gaps_full = np.full(self.nr_of_masses, np.inf)
        gaps_contact = self.g_N(t,q)
        for k, i in enumerate(self.outer_masses):
            gaps_full[i] = gaps_contact[k]
        return gaps_full
    


class mass_mass_contact:
    def __init__(self, MultiPixelbot, name):
        self.name = name

        # define necessary variables from pixelbot class
        self.MultiPixelbot = MultiPixelbot
        self.nr_of_masses = MultiPixelbot.total_masses
        self._nq = MultiPixelbot.nq
        self._nu = MultiPixelbot.nu

        # outer masses
        self.outer_masses = MultiPixelbot.outer_mass()
        self.same_pixel = MultiPixelbot.same_pixel
        self.neighbor_pixel = MultiPixelbot.neighbor_pixel

        self.pixel_size = self.MultiPixelbot.pixel_size 

        mass_trios = set()

        for pixel_masses in self.MultiPixelbot.bot_global_mass:            
            m0, m1, m2, m3 = pixel_masses
            pixel_edges = [(m0, m1), (m1, m2), (m2, m3), (m3, m0)]

            for (j1, j2) in pixel_edges:
                if (j1 not in self.outer_masses) or (j2 not in self.outer_masses):
                    continue
                
                for i in self.outer_masses:
                    if self.same_pixel(i, j1) or self.same_pixel(i, j2):
                        continue
                    if self.neighbor_pixel(i, j1) or self.neighbor_pixel(i, j2):
                        continue

                    trio = (i, j1, j2)
                    mass_trios.add(trio)
                
        self.mass_trios = list(mass_trios)

        self.segments = []
        for (i, j1, j2) in self.mass_trios:
            pj1 = self.MultiPixelbot.q0[2*j1:2*j1+2]
            pj2 = self.MultiPixelbot.q0[2*j2:2*j2+2]
            v = pj2 - pj1
            self.segments.append((pj1.copy(), v.copy()))
        

        self.nr_contact = len(self.mass_trios)

        # define contact variables for solver
        self.nla_N = self.nr_contact
        self.nla_F = self.nr_contact
        self.mu = 0.75 # 0 < mu < 1, 0 no friction
        self.e_N = np.full(self.nr_contact, 0.5) # 0 < eN < 1, 1 perfectly elastic (no energy loss)
        self.e_F = np.full(self.nr_contact, 0.0) # 0 < eF < 1, 1 perfectly elastic (no energy loss)

        # coulomb friction laws
        self.friction_laws = [
            ([i], [i], Sphere(self.mu)) for i in range(self.nr_contact)
            ]

        self.contact_radius = 0.05 * self.pixel_size

        self.closest_normals = [np.array([0.0, 1.0]) for _ in range(self.nr_contact)]
        self.closest_s = [0.0 for _ in range(self.nla_N)]


    def assembler_callback(self):
        self.qDOF = self.MultiPixelbot.qDOF
        self.uDOF = self.MultiPixelbot.uDOF


    def g_N(self, t, q):
        gaps = np.full(self.nla_N, np.inf, dtype=float)

        for k, (i, j1, j2) in enumerate(self.mass_trios):
            p_mass = np.asarray(q[2*i:2*i+2], dtype=float)
            best_dist = np.inf
            best_gap = np.inf
            best_normal = np.array([0.0, 1.0], dtype=float)

            p0, v = self.segments[k]
            seg_len = np.linalg.norm(v)
            v_unit = v/seg_len
            normal = np.array([-v_unit[1], v_unit[0]], dtype=float)
            normal /= np.linalg.norm(normal)

            # compute fraction along the segment where closest point lies
            proj = np.dot((p_mass - p0), v)
            v_len2 = np.dot(v,v)
            s = proj / v_len2
            if s < 0-self.contact_radius or s > 1+self.contact_radius:
                gaps[k] = np.inf
                self.closest_normals[k] = np.array([0.0, 1.0])
                self.closest_s[k] = 0.5
                continue

            p_closest = p0 + s * v

            d_vec = p_mass - p_closest
            dist = np.linalg.norm(d_vec)
    
            gap_val = np.dot(d_vec, normal) - self.contact_radius

            if dist < best_dist:
                best_gap = gap_val
                best_normal = normal
                best_dist = dist
            
            gaps[k] = best_gap
            self.closest_normals[k] = best_normal
            self.closest_s[k] = s
    
        return gaps
    

    def W_N(self, t, q):
        WN = np.zeros((self._nq, self.nla_N))
        self.g_N(t, q)

        for k, (i, j1, j2) in enumerate(self.mass_trios):
            normal = self.closest_normals[k]
            s = self.closest_s[k]

            WN[2*i, k] = normal[0]
            WN[2*i+1, k] = normal[1]

            WN[2*j1, k] = -(1.0-s)*normal[0]
            WN[2*j1+1, k] = -(1.0-s)*normal[1]

            WN[2*j2, k] = -s*normal[0]
            WN[2*j2+1, k] = -s*normal[1]
        return WN
    
    def g_N_dot(self, t, q, u):
        return self.W_N(t, q).T @ u


    ## Define tangential contact
    ## Define tangential jacobian matrix for solver
    def W_F(self, t, q):
        WF = np.zeros((self._nq, self.nla_N), dtype=float)
        self.g_N(t, q)

        for k, (i, j1, j2) in enumerate(self.mass_trios):
            normal = self.closest_normals[k]
            s = self.closest_s[k]

            tangent = np.array([normal[1], -normal[0]])
            tangent /= np.linalg.norm(tangent)
            WF[2*i, k] = tangent[0]
            WF[2*i+1, k] = tangent[1]

            WF[2*j1, k] = -(1.0-s)*tangent[0]
            WF[2*j1+1, k] = -(1.0-s)*tangent[1]

            WF[2*j2, k] = -s*tangent[0]
            WF[2*j2+1, k] = -s*tangent[1]
        return WF
    

    ## Define tangential relative velocity
    def gamma_F(self, t, q, u):
        return self.W_F(t,q).T @ u
    
    ## add full g_N for animation
    def g_full(self, t, q):
        gaps_full = np.full(self.nr_of_masses, np.inf)
        gaps_contact = self.g_N(t,q)
        for k, (i, j1, j2) in enumerate(self.mass_trios):
            gaps_full[i] = min(gaps_full[i], gaps_contact[k])
            gaps_full[j1] = min(gaps_full[j1], gaps_contact[k])
            gaps_full[j2] = min(gaps_full[j2], gaps_contact[k])
        return gaps_full
    
