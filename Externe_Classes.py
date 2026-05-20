import numpy as np

import matplotlib.tri as mtri
import trimesh
import bfieldtools
from bfieldtools.utils import combine_meshes

import scipy.constants
mu_0=scipy.constants.mu_0

class Coil_Layup():
    def __init__(self, coil_diameter, x_dist, y_dist, z_dist, n_windings, current, coil_plane_dist_to_origin_x, coil_plane_dist_to_origin_y, coil_plane_dist_to_origin_z, usable_length_x, usable_length_y, usable_length_z):
        self.coil_diameter = coil_diameter
        self.x_dist = x_dist
        self.y_dist = y_dist
        self.z_dist = z_dist
        self.n_windings = n_windings
        self.current = current
        self.coil_plane_dist_to_origin_x = coil_plane_dist_to_origin_x
        self.coil_plane_dist_to_origin_y = coil_plane_dist_to_origin_y
        self.coil_plane_dist_to_origin_z = coil_plane_dist_to_origin_z
        self.usable_length_x = usable_length_x
        self.usable_length_y = usable_length_y
        self.usable_length_z = usable_length_z

        # Compute the number of coils that can be placed in each direction
        self.number_of_coils_x = int(np.floor(max(self.usable_length_x - self.x_dist, 0) / (self.coil_diameter + self.x_dist)))
        self.number_of_coils_y = int(np.floor(max(self.usable_length_y - self.y_dist, 0) / (self.coil_diameter + self.y_dist)))
        self.number_of_coils_z = int(np.floor(max(self.usable_length_z - self.z_dist, 0) / (self.coil_diameter + self.z_dist)))

        # Compute the coordinates of the coil centers
        x_points = []
        if self.number_of_coils_x % 2 == 1:
            x_points.append(0)
            for i in range((self.number_of_coils_x - 1) // 2):
                x_points.append(i * (self.coil_diameter + self.x_dist))
                x_points.append(-i * (self.coil_diameter + self.x_dist))
        else:
            for i in range(self.number_of_coils_x // 2):
                x_points.append((i + 0.5) * (self.coil_diameter + self.x_dist))
                x_points.append(-(i + 0.5) * (self.coil_diameter + self.x_dist))

        y_points = []
        if self.number_of_coils_y % 2 == 1:
            y_points.append(0)
            for i in range((self.number_of_coils_y - 1) // 2):
                y_points.append(i * (self.coil_diameter + self.y_dist))
                y_points.append(-i * (self.coil_diameter + self.y_dist))
        else:
            for i in range(self.number_of_coils_y // 2):
                y_points.append((i + 0.5) * (self.coil_diameter + self.y_dist))
                y_points.append(-(i + 0.5) * (self.coil_diameter + self.y_dist))

        z_points = []
        if self.number_of_coils_z % 2 == 1:
            z_points.append(0)
            for i in range((self.number_of_coils_z - 1) // 2):
                z_points.append(i * (self.coil_diameter + self.z_dist))
                z_points.append(-i * (self.coil_diameter + self.z_dist))
        else:
            for i in range(self.number_of_coils_z // 2):
                z_points.append((i + 0.5) * (self.coil_diameter + self.z_dist))
                z_points.append(-(i + 0.5) * (self.coil_diameter + self.z_dist))

        # Convert to arrays
        x_points = np.array(x_points)
        y_points = np.array(y_points)
        z_points = np.array(z_points)

        # Define simple function to create circle around a center point
        def create_circle(center_p, center_q, diameter):
            theta = np.linspace(0, 2 * np.pi, 100)
            p = center_p + (diameter / 2) * np.cos(theta)
            q = center_q + (diameter / 2) * np.sin(theta)
            return p, q

        # XY plane: (n_coils_x * n_coils_y, 100, 3)
        n_coils_xy = len(x_points) * len(y_points)
        all_x_xy_pos = np.empty((n_coils_xy, 100))
        all_y_xy_pos = np.empty((n_coils_xy, 100))
        all_z_xy_pos = np.full((n_coils_xy, 100), self.coil_plane_dist_to_origin_z)
        all_z_xy_neg = np.full((n_coils_xy, 100), -self.coil_plane_dist_to_origin_z)

        self.grid_xy = []
        k = 0
        for i in range(len(x_points)):
            for j in range(len(y_points)):
                p, q = create_circle(x_points[i], y_points[j], self.coil_diameter)
                all_x_xy_pos[k] = p
                all_y_xy_pos[k] = q
                self.grid_xy.append((x_points[i], y_points[j], self.coil_plane_dist_to_origin_z))
                self.grid_xy.append((x_points[i], y_points[j], -self.coil_plane_dist_to_origin_z))
                k += 1

        # YZ plane: (n_coils_y * n_coils_z, 100, 3)
        n_coils_yz = len(y_points) * len(z_points)
        all_y_yz_pos = np.empty((n_coils_yz, 100))
        all_z_yz_pos = np.empty((n_coils_yz, 100))
        all_x_yz_pos = np.full((n_coils_yz, 100), self.coil_plane_dist_to_origin_x)
        all_x_yz_neg = np.full((n_coils_yz, 100), -self.coil_plane_dist_to_origin_x)

        self.grid_yz = []
        k = 0
        for i in range(len(y_points)):
            for j in range(len(z_points)):
                p, q = create_circle(y_points[i], z_points[j], self.coil_diameter)
                all_y_yz_pos[k] = p
                all_z_yz_pos[k] = q
                self.grid_yz.append((self.coil_plane_dist_to_origin_x, y_points[i], z_points[j]))
                self.grid_yz.append((-self.coil_plane_dist_to_origin_x, y_points[i], z_points[j]))
                k += 1

        # XZ plane: (n_coils_x * n_coils_z, 100, 3)
        n_coils_xz = len(x_points) * len(z_points)
        all_x_xz_pos = np.empty((n_coils_xz, 100))
        all_z_xz_pos = np.empty((n_coils_xz, 100))
        all_y_xz_pos = np.full((n_coils_xz, 100), self.coil_plane_dist_to_origin_y)
        all_y_xz_neg = np.full((n_coils_xz, 100), -self.coil_plane_dist_to_origin_y)

        self.grid_xz = []
        k = 0
        for i in range(len(x_points)):
            for j in range(len(z_points)):
                p, q = create_circle(x_points[i], z_points[j], self.coil_diameter)
                all_x_xz_pos[k] = p
                all_z_xz_pos[k] = q
                self.grid_xz.append((x_points[i], self.coil_plane_dist_to_origin_y, z_points[j]))
                self.grid_xz.append((x_points[i], -self.coil_plane_dist_to_origin_y, z_points[j]))
                k += 1

        self.grid = []
        if not self.grid_xy:
            print('Warning: No coils in xy-plane. Check that useable_length_ is not 0 for x or y!')
        else:
            self.grid.append(self.grid_xy)
        if not self.grid_yz:
            print('Warning: No coils in yz-plane. Check that useable_length_ is not 0 for y or z!')
        else:
            self.grid.append(self.grid_yz)
        if not self.grid_xz:
            print('Warning: No coils in xz-plane. Check that useable_length_ is not 0 for x or z!')
        else:
            self.grid.append(self.grid_xz)
        self.grid = np.array(self.grid)


        self.plane_xy_pos = np.stack([all_x_xy_pos, all_y_xy_pos, all_z_xy_pos], axis=-1)  # (n_coils_xy, 100, 3)
        self.plane_xy_neg = np.stack([all_x_xy_pos, all_y_xy_pos, all_z_xy_neg], axis=-1)  # (n_coils_xy, 100, 3)
        self.plane_yz_pos = np.stack([all_x_yz_pos, all_y_yz_pos, all_z_yz_pos], axis=-1)  # (n_coils_yz, 100, 3)
        self.plane_yz_neg = np.stack([all_x_yz_neg, all_y_yz_pos, all_z_yz_pos], axis=-1)  # (n_coils_yz, 100, 3)
        self.plane_xz_pos = np.stack([all_x_xz_pos, all_y_xz_pos, all_z_xz_pos], axis=-1)  # (n_coils_xz, 100, 3)
        self.plane_xz_neg = np.stack([all_x_xz_pos, all_y_xz_neg, all_z_xz_pos], axis=-1)  # (n_coils_xz, 100, 3)
        self.all_coils = np.concatenate([
            self.plane_xy_pos, self.plane_xy_neg, 
            self.plane_yz_pos, self.plane_yz_neg, 
            self.plane_xz_pos, self.plane_xz_neg], axis=0)  # (n_coils_total, 100, 3)
        
        # Create array which contains the number of coil windings for each coil (n_coils_total, n_windings)
        # Handle n_windings: scalar → broadcast OR array
        if np.isscalar(n_windings) or (hasattr(n_windings, 'shape') and n_windings.shape == ()):
            # Scalar: replicate für alle Coils
            self.n_windings = np.full(self.all_coils.shape[0], n_windings, dtype=int)
        elif isinstance(n_windings, (np.ndarray, list)):
            # Array: check length
            if len(n_windings) == self.all_coils.shape[0]:
                self.n_windings = np.array(n_windings, dtype=int)
            else:
                raise ValueError(f"n_windings length {len(n_windings)} != n_coils {self.all_coils.shape[0]}")
        else:
            raise ValueError("n_windings must be scalar, list or np.array")
        
        if np.isscalar(current) or (hasattr(current, 'shape') and current.shape == ()):
            # Scalar: replicate für alle Coils
            self.current = np.full(self.all_coils.shape[0], current, dtype=float)
        elif isinstance(current, (np.ndarray, list)):
            # Array: check length
            if len(current) == self.all_coils.shape[0]:
                self.current = np.array(current, dtype=float)
            else:
                raise ValueError(f"current length {len(current)} != n_coils {self.all_coils.shape[0]}")
        else:
            raise ValueError("current must be scalar, list or np.array")
        
        self.I_windings_total = self.current * self.n_windings  # (n_coils_total,)


    def coil_field_by_Biot_Savart(self, target_points):
            """
            target_points: (n_targets, 3)
            coils: (n_coils, n_points_per_coil, 3)
                Each coil must be closed or will be closed by connecting last to first.
            I_windings_total: (n_coils,) array of total currents (current * windings) for each coil
            returns: (n_targets, 3)
            """

            # Segment vectors along each coil
            dR = np.roll(self.all_coils, -1, axis=1) - self.all_coils                       # (n_coils, n_seg, 3); difference between consecutive points along each coil, with wrap-around for closed loop
            ds = np.linalg.norm(dR, axis=2, keepdims=True)                                  # (n_coils, n_seg, 1); length of each circle segment computed from geometry

            # Current element dℓ = ds * tangent_hat, here tangent_hat = dR / |dR|
            # so dℓ = dR. Multiply by current and winding count.
            dl = self.I_windings_total[:, None, None] * dR                                  # (n_coils, n_seg, 3); current * windings * tangential sector vector (not normalized, since ds is included)

            B = np.zeros((target_points.shape[0], 3), dtype=float)

            for i, target in enumerate(target_points):
                r = target[None, None, :] - self.all_coils                                  # (n_coils, n_seg, 3); $\vec{r}-\vec{r}'$
                r_norm = np.linalg.norm(r, axis=2)                                          # (n_coils, n_seg);    $|\vec{r}-\vec{r}'|$

                # Avoid division by zero if a target lies exactly on a segment point
                if np.any(r_norm <= 0):
                    print(f'WARNING: r_norm = {r_norm} which leads to a division by zero in the Biot-Savart calculation (Point {i})')

                cross = np.cross(dl, r)                                                     # (n_coils, n_seg, 3); $\mathrm{d}\vec{\ell} \cross (\vec{r}-\vec{r}')$
                contrib = np.zeros_like(cross)
                contrib = cross / (r_norm**3)[:, :, None]                                   # Note that r_norm has shape (n_coils, n_seg) and cross has shape (n_coils, n_seg, 3), therefore the axis for broadcasting are needed)

                B[i] = mu_0 / (4 * np.pi) * np.sum(contrib, axis=(0, 1))

            return B
    
    def create_mesh(self, n, door_removal, door_width, door_height, door_floor_offset, door_offset_x):
        
        # Initialise variables from method input
        self.door_removal = door_removal
        self.door_width = door_width
        self.door_height = door_height
        self.door_floor_offset = door_floor_offset
        self.door_offset_x = door_offset_x
        nx, ny, nz = n, n, n

        # Create eqidistant spacing in each direction as basis for the grid
        # size=coil_plane / 2                                                    # Size of the grid in each direction (Note: coil_loc = 0.9)
        x = np.linspace(-self.coil_plane_dist_to_origin_x, self.coil_plane_dist_to_origin_x, nx)
        y = np.linspace(-self.coil_plane_dist_to_origin_y, self.coil_plane_dist_to_origin_y, ny)
        z = np.linspace(-self.coil_plane_dist_to_origin_z, self.coil_plane_dist_to_origin_z, nz)

        #xy-plane = bottom, top -> stores points (vertices) of 2D square grid
        xx, yy = np.meshgrid(x, y)                                                  # xx is an array of ny copies of x (yy analogue)
        xy_plane = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])     # .ravel flattens the array to 1D and .column_stack puts arrays onto each other, s.t. the points (vertices) of the 2D grid are stored with a 0 attended to each point.

        #xz-plane = (front not anymore),back
        xx, zz = np.meshgrid(x, z)
        xz_plane = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()])

        #yz-plane = left, right
        yy, zz = np.meshgrid(y, z)
        yz_plane = np.column_stack([np.zeros(yy.size), yy.ravel(), zz.ravel()])

        # Create triangles for the square mesh, completely filling the meshing plane
        triangles_xy = []

        for i in range(ny - 1):
            for j in range(nx - 1):
                p0 = i * nx + j # point on grid (nx by ny)
                p1 = p0 + 1     # next point to the right of p0
                p2 = p0 + nx    # next point above p0
                p3 = p2 + 1     # next point to the right of p2

                # append triangle 1
                triangles_xy.append([p0, p1, p2])
                # append triangle 2
                triangles_xy.append([p1, p3, p2])

        triangles_xy = np.array(triangles_xy)

        # Construct a trimesh object from the meshes
        triangles_xz = triangles_xy.copy()
        triangles_yz = triangles_xy.copy()

        # Construct a trimesh object from the meshes
        tri_xy = trimesh.Trimesh(vertices = xy_plane, faces = triangles_xy, process = False)
        tri_xz = trimesh.Trimesh(vertices = xz_plane, faces = triangles_xz, process = False)
        tri_yz = trimesh.Trimesh(vertices = yz_plane, faces = triangles_yz, process = False)

        tri_xy = tri_xy.subdivide().subdivide()
        tri_xz = tri_xz.subdivide().subdivide()
        tri_yz = tri_yz.subdivide().subdivide()

        # XY faces (top/bottom)
        self.mesh_top = tri_xy.copy()
        self.mesh_top.apply_translation([0, 0, +self.coil_plane_dist_to_origin_z])

        self.mesh_bottom = tri_xy.copy()
        self.mesh_bottom.apply_translation([0, 0, -self.coil_plane_dist_to_origin_z])

        # XZ faces (front/back)
        self.mesh_front = tri_xz.copy()
        self.mesh_front.apply_translation([0, +self.coil_plane_dist_to_origin_y, 0])

        self.mesh_back = tri_xz.copy()
        self.mesh_back.apply_translation([0, -self.coil_plane_dist_to_origin_y, 0])

        # YZ faces (left/right)
        self.mesh_right = tri_yz.copy()
        self.mesh_right.apply_translation([+self.coil_plane_dist_to_origin_x, 0, 0])

        self.mesh_left = tri_yz.copy()
        self.mesh_left.apply_translation([-self.coil_plane_dist_to_origin_x, 0, 0])

        # Remove the door!!!
        # Safety feature
        if self.coil_plane_dist_to_origin_x == 0 or self.coil_plane_dist_to_origin_z == 0:
            self.door_removal = False                                   # If the wall containing the door does not exist, the dorr can not exist

        if self.door_removal == True:

            # Include door in xz-plane (front wall) -> remove by using .difference() which is a boolean operation substracting the door mesh from the rest
            door_xmin = self.usable_length_x/2 - self.door_offset_x - self.door_width
            door_xmax = self.usable_length_x/2 - self.door_offset_x
            door_zmin = -self.usable_length_z/2 + self.door_floor_offset
            door_zmax = -self.usable_length_z/2 + self.door_floor_offset + self.door_height

            door_faces = []
            front_wall_faces = []

            # print(f"Before loop: door_faces len={len(door_faces)}, front_wall_faces len={len(front_wall_faces)}")

            # Note that the door is located in the "back-plane"
            for fi, face in enumerate(self.mesh_back.faces):
                pts = self.mesh_back.vertices[face]
                xpts = pts[:, 0]                                # Get the x-coordinates of the triangle vertices
                zpts = pts[:, 2]                                # Get the z-coordinates of the triangle vertices
                
                # All 3 vertices inside door rectangle?
                if (all(door_xmin <= px <= door_xmax for px in xpts) and    # Check door
                    all(door_zmin <= pz <= door_zmax for pz in zpts)):
                    door_faces.append(face)
                elif (all(door_xmin >= px or px >= door_xmax for px in xpts) or    # Check front wall
                    all(door_zmin >= pz or pz >= door_zmax for pz in zpts)):
                    front_wall_faces.append(face)

            # Remaining faces form wall with hole
            door_faces = np.array(door_faces)
            front_wall_faces = np.array(front_wall_faces)

            # print(f"After loop: door_faces len={len(door_faces)}, front_wall_faces len={len(front_wall_faces)}")
            # print(f"Total faces: {len(self.coil_plus_xz.faces)}, Classified: {len(door_faces) + len(front_wall_faces)}")

            # Create separate meshes for wall and door with a small gap between them
            # This allows applying different boundary conditions later

            # Wall mesh
            wall_vertex_indices = np.unique(front_wall_faces.ravel())
            wall_vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(wall_vertex_indices)}
            wall_vertices = self.mesh_back.vertices[wall_vertex_indices]
            wall_faces_remapped = np.array([[wall_vertex_map[v] for v in face] for face in front_wall_faces])
            wall_mesh = trimesh.Trimesh(vertices = wall_vertices, faces = wall_faces_remapped, process=False)

            # Door mesh
            door_vertex_indices = np.unique(door_faces.ravel())
            door_vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(door_vertex_indices)}
            door_vertices = self.mesh_back.vertices[door_vertex_indices].copy()
            door_faces_remapped = np.array([[door_vertex_map[v] for v in face] for face in door_faces])
            door_mesh = trimesh.Trimesh(vertices = door_vertices, faces = door_faces_remapped, process=False)

            # Combined mesh with gap
            self.mesh_back = trimesh.util.concatenate([wall_mesh, door_mesh])

            # Note: Wall_mesh and door_mesh remain available for individual boundary condition application

        else:
            pass
        
        # Combine the meshes. Switch to special cases, if there are usable_lengths of 0 somewhere!
        
        self.total_planes=combine_meshes((
            self.mesh_top,self.mesh_bottom,
            self.mesh_front, self.mesh_back,
            self.mesh_right,self.mesh_left
            ))
        

        # The folling lines can be used, if mesh boundaries at the corners are connected. Note that the door boundary is not effected!
        # self.total_planes.merge_vertices(digits_vertex = 6)

        # self.total_planes.update_faces(self.total_planes.unique_faces())
        # self.total_planes.remove_unreferenced_vertices()
        # self.total_planes.fix_normals()

        # self.total_planes.merge_vertices(digits_vertex=6)

        self.coil_plane_boundaries =  trimesh.grouping.group_rows(
            self.total_planes.edges_sorted,
            require_count=1
            )
        
    def stream_function_coils (self, current, n_windings):

        self.current = current
        self.n_windings = n_windings

        face_outside_loop = [[] for _ in range(2 * self.number_of_coils_y * self.number_of_coils_z + 2 * self.number_of_coils_x * self.number_of_coils_z + 2 * self.number_of_coils_x * self.number_of_coils_y)]
        face_inside_loop  = [[] for _ in range(2 * self.number_of_coils_y * self.number_of_coils_z + 2 * self.number_of_coils_x * self.number_of_coils_z + 2 * self.number_of_coils_x * self.number_of_coils_y)]
        face_with_current = [[] for _ in range(2 * self.number_of_coils_y * self.number_of_coils_z + 2 * self.number_of_coils_x * self.number_of_coils_z + 2 * self.number_of_coils_x * self.number_of_coils_y)]

        for f_idx, faces in enumerate(self.total_planes.faces):
            r = self.total_planes.vertices[faces]                                           # shape (3, 3) -> Coordinates of the vertices belonging to the face

            # x = const plane
            if np.allclose(r[:, 0], r[0, 0]):                                               # All vertices are on the YZ-plane (left/right)
                for coil in range(2 * self.number_of_coils_y * self.number_of_coils_z):     # Iterarte through all coils (2* since ther are two yz-planes)
                    center_y, center_z = self.grid_yz[coil][1], self.grid_yz[coil][2]       # Assign center points of the coil
                    d_to_coil_center = (r[:, 1] - center_y)**2 + (r[:, 2] - center_z)**2    # Distance to the coil center

                    if np.all(d_to_coil_center < (self.coil_diameter / 2)**2):              # All vertices of this face are inside the coil
                        face_inside_loop[coil].append(f_idx)
                    elif np.all(d_to_coil_center > (self.coil_diameter / 2)**2):            # All vertices of this face are outside the circle
                        face_outside_loop[coil].append(f_idx)
                    elif np.any(d_to_coil_center < (self.coil_diameter / 2)**2 ) and np.any(d_to_coil_center > (self.coil_diameter / 2)**2): # The face is intersected by the coil, since some vertices are inside it, and others are outside
                        face_with_current[coil].append(f_idx)

            # y = const plane
            elif np.allclose(r[:, 1], r[0, 1]):                                             # All vertices are on the XZ-plane (front/back)
                for coil in range(2 * self.number_of_coils_x * self.number_of_coils_z):     # Iterarte through all coils (2* since ther are two yz-planes)
                    center_x, center_z = self.grid_xz[coil][0], self.grid_xz[coil][2]       # Assign center points of the coil
                    d_to_coil_center = (r[:, 0] - center_x)**2 + (r[:, 2] - center_z)**2    # Distance to the coil center

                    if np.all(d_to_coil_center < (self.coil_diameter / 2)**2):              # All vertices of this face are inside the coil
                        face_inside_loop[coil].append(f_idx)
                    elif np.all(d_to_coil_center > (self.coil_diameter / 2)**2):            # All vertices of this face are outside the circle
                        face_outside_loop[coil].append(f_idx)
                    elif np.any(d_to_coil_center < (self.coil_diameter / 2)**2 ) and np.any(d_to_coil_center > (self.coil_diameter / 2)**2): # The face is intersected by the coil, since some vertices are inside it, and others are outside
                        face_with_current[coil].append(f_idx)

            # z = const plane
            elif np.allclose(r[:, 2], r[0, 2]):                                             # All vertices are on the XY-plane (top/bottom)
                for coil in range(2 * self.number_of_coils_x * self.number_of_coils_y):     # Iterarte through all coils (2* since ther are two yz-planes)
                    center_x, center_y = self.grid_xy[coil][0], self.grid_xy[coil][1]       # Assign center points of the coil
                    d_to_coil_center = (r[:, 0] - center_x)**2 + (r[:, 1] - center_y)**2    # Distance to the coil center

                    if np.all(d_to_coil_center < (self.coil_diameter / 2)**2):              # All vertices of this face are inside the coil
                        face_inside_loop[coil].append(f_idx)
                    elif np.all(d_to_coil_center > (self.coil_diameter / 2)**2):            # All vertices of this face are outside the circle
                        face_outside_loop[coil].append(f_idx)
                    elif np.any(d_to_coil_center < (self.coil_diameter / 2)**2 ) and np.any(d_to_coil_center > (self.coil_diameter / 2)**2): # The face is intersected by the coil, since some vertices are inside it, and others are outside
                        face_with_current[coil].append(f_idx)

        stream_function_on_faces = np.zeros(len(self.total_planes.faces))
        dx = np.abs(self.total_planes.vertices[0][0] - self.total_planes.vertices[1][0])    # dx = dy = dz since the grid is build identically on each side !!!

        for f_idx in range(len(self.total_planes.faces)):                                   # Iterate through all faces
            if any(f_idx in lst for lst in face_inside_loop):                               # for all faces inside the coil do:
                stream_function_on_faces[f_idx] = self.current * self.n_windings * dx                 # Assign the analytical result of the streamfunction to these faces
            elif any(f_idx in lst for lst in face_with_current):                            # Assign every face which gets intersected by the coil half the value of the inside streamfunction
                stream_function_on_faces[f_idx] = 0.5 * self.current * self.n_windings * dx
            else:
                stream_function_on_faces[f_idx] = 0.0                                       # Assign all other faces the value 0 (this is the analytical solution!)

        stream_function_on_vertices = np.zeros(len(self.total_planes.vertices))             # We need to express the stream_function for all vertices, not faces

        for v_idx in range(len(self.total_planes.vertices)):
            indices = self.total_planes.vertex_faces[v_idx]
            indices = np.array([i for i in indices if i != -1])                             # Find the indices of the faces which are adjacent to one vertex
            stream_function_on_vertices[v_idx] = np.mean(stream_function_on_faces[indices]) # Assign each vertex a streamfunction value accourding to the average of the streamfunction value of all adjacent faces

        return stream_function_on_vertices


class Mu_material():
    def __init__(self, n, dim, thickness):
        # Initialise variables from constructor input
        self.dim = dim
        self.thickness = thickness
        self.n_discretization = n

        # Discretise the space available for the mesh
        nx, ny, nz = self.n_discretization, self.n_discretization, self.n_discretization
        size = self.dim/2                                                    # Size of the grid in each direction (Note: offset = 1.15)
        x = np.linspace(-size, size, nx)
        y = np.linspace(-size, size, ny)
        z = np.linspace(-size, size, nz)

        # Create meshes in each plane seperatly (These are just square meshes -> Arrays holding the vertices. Nothing more)
        # xy-plane = bottom, top
        xx, yy = np.meshgrid(x, y)
        xy = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

        # xz-plane=front,back
        xx, zz = np.meshgrid(x, z)
        xz = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()])

        # yz-plane=left, right
        yy, zz = np.meshgrid(y, z)
        yz = np.column_stack([np.zeros(yy.size), yy.ravel(), zz.ravel()])

        # Create triangles for the square mesh, completely filling the meshing plane
        triangles_xy=[]

        for i in range(ny - 1):
            for j in range(nx - 1):
                p0 = i * nx + j
                p1 = p0 + 1
                p2 = p0 + nx
                p3 = p2 + 1

                # append triangle 1
                triangles_xy.append([p0, p1, p2])
                # append triangle 2
                triangles_xy.append([p1, p3, p2])

        triangles_xy = np.array(triangles_xy)

        # Copy what we just did for the other two directions as well
        triangles_xz = triangles_xy.copy()
        triangles_yz = triangles_xy.copy()

        # Construct a trimesh object from the meshes
        tri_xy = trimesh.Trimesh(vertices = xy, faces = triangles_xy, process = False)
        tri_xz = trimesh.Trimesh(vertices = xz, faces = triangles_xz, process = False)
        tri_yz = trimesh.Trimesh(vertices = yz, faces = triangles_yz, process = False)

        # Place the meshes to the side, such that a cube (equal side lengths) is created
        offset = self.dim / 2                   # Location of the shield from the center (normal to surface) = 1/2 shield size (MSR is closed) -> compare to coil distance
        
        # XY faces (top/bottom)
        mesh_top = tri_xy.copy()
        mesh_top.apply_translation([0, 0, +offset])

        mesh_bottom = tri_xy.copy()
        mesh_bottom.apply_translation([0, 0, -offset])

        # XZ faces (front/back)
        mesh_front = tri_xz.copy()
        mesh_front.apply_translation([0, +offset, 0])

        mesh_back = tri_xz.copy()
        mesh_back.apply_translation([0, -offset, 0])

        # YZ faces (left/right)
        mesh_right = tri_yz.copy()
        mesh_right.apply_translation([+offset, 0, 0])

        mesh_left = tri_yz.copy()
        mesh_left.apply_translation([-offset, 0, 0])

        self.total_shield = trimesh.util.concatenate([
            mesh_top,
            mesh_bottom,
            mesh_front,
            mesh_back,
            mesh_right,
            mesh_left
            ])

        # CRUCIAL: Repair seams/duplicates
        self.total_shield.merge_vertices(digits_vertex = 6)  # Merge close Vertices (Edges of the cube)
        
        # remove duplicate geometry
        self.total_shield.update_faces(self.total_shield.unique_faces())
        self.total_shield.remove_unreferenced_vertices()
        self.total_shield.fix_normals()

        # Refine mesh
        self.total_shield = self.total_shield.subdivide().subdivide()

        # Merge the vertices at the edges of the cube, such that current can flow around the corner
        self.total_shield.merge_vertices(digits_vertex=6)

        # Debugging
        print("Watertight:", self.total_shield.is_watertight)
        print("Euler number:", self.total_shield.euler_number)

        # Create a trimesh boundary object
        self.boundaries = trimesh.grouping.group_rows(
            self.total_shield.edges_sorted,
            require_count=1
            )
        print("Boundary edges should have length = 0! They have length =", len(self.boundaries))

        # Create a trimesh shield_conductor object
        self.shield_conductor = bfieldtools.mesh_conductor.MeshConductor(
            mesh_obj = self.total_shield,
            basis_name = "inner"
            )
        
        # Jetzt: Keine Boundary-Edges mehr (closed manifold)
        self.inner_idx = bfieldtools.utils.find_mesh_boundaries(self.total_shield)

        #points_inside = total_shield.vertices - eps * total_shield.vertex_normals
        self.points_inside = self.total_shield.vertices - self.thickness * self.total_shield.vertex_normals
