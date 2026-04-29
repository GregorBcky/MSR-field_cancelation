import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import trimesh
import bfieldtools
from bfieldtools.utils import combine_meshes

class Coil_Layup():
    def __init__(self, coil_diameter, x_dist, y_dist, z_dist, n_windings, coil_plane_dist_to_origin_x, coil_plane_dist_to_origin_y, coil_plane_dist_to_origin_z, usable_length_x, usable_length_y, usable_length_z):
        self.coil_diameter = coil_diameter
        self.x_dist = x_dist
        self.y_dist = y_dist
        self.z_dist = z_dist
        self.n_windings = n_windings
        self.coil_plane_dist_to_origin_x = coil_plane_dist_to_origin_x
        self.coil_plane_dist_to_origin_y = coil_plane_dist_to_origin_y
        self.coil_plane_dist_to_origin_z = coil_plane_dist_to_origin_z
        self.usable_length_x = usable_length_x
        self.usable_length_y = usable_length_y
        self.usable_length_z = usable_length_z

        # Compute the number of coils that can be placed in each direction
        self.number_of_coils_x = int(np.floor((self.usable_length_x - self.x_dist) / (self.coil_diameter + self.x_dist)))
        self.number_of_coils_y = int(np.floor((self.usable_length_y - self.y_dist) / (self.coil_diameter + self.y_dist)))
        self.number_of_coils_z = int(np.floor((self.usable_length_z - self.z_dist) / (self.coil_diameter + self.z_dist)))

        # Compute the coordinates of the coil centers (deine Logik unverändert)
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

        self.grid = np.concatenate([np.array(self.grid_xy), np.array(self.grid_yz), np.array(self.grid_xz)])

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
            self.n_windings_per_coil = np.full(self.all_coils.shape[0], n_windings, dtype=int)
        elif isinstance(n_windings, (np.ndarray, list)):
            # Array: check length
            if len(n_windings) == self.all_coils.shape[0]:
                self.n_windings_per_coil = np.array(n_windings, dtype=int)
            else:
                raise ValueError(f"n_windings length {len(n_windings)} != n_coils {self.all_coils.shape[0]}")
        else:
            raise ValueError("n_windings must be scalar, list or np.array")

class Mu_material():
    def __init__(self, shield_dim, shield_thickness):
        self.shield_dim = shield_dim
        self.shield_thickness = shield_thickness
        self.n_discretization = 10

        nx_shield, ny_shield, nz_shield = self.n_discretization, self.n_discretization, self.n_discretization
        size_shield = self.shield_dim/2                                                    # Size of the grid in each direction (Note: shield_loc = 1.15)
        x_shield=np.linspace(-size_shield,size_shield,nx_shield)
        y_shield=np.linspace(-size_shield,size_shield,ny_shield)
        z_shield=np.linspace(-size_shield,size_shield,nz_shield)

        #xy-plane = bottom, top
        xx_shield,yy_shield=np.meshgrid(x_shield,y_shield)
        xy_shield=np.column_stack([xx_shield.ravel(),yy_shield.ravel(),np.zeros(xx_shield.size)])

        #xz-plane=front,back
        xx_shield,zz_shield=np.meshgrid(x_shield,z_shield)
        xz_shield=np.column_stack([xx_shield.ravel(),np.zeros(xx_shield.size),zz_shield.ravel()])

        #yz-plane=left, right
        yy_shield,zz_shield=np.meshgrid(y_shield,z_shield)
        yz_shield=np.column_stack([np.zeros(yy_shield.size),yy_shield.ravel(),zz_shield.ravel()])

        triangles_xy_shield=[]


        for i in range(ny_shield - 1):
            for j in range(nx_shield - 1):
                p0 = i * nx_shield + j
                p1 = p0 + 1
                p2 = p0 + nx_shield
                p3 = p2 + 1

                # triangle 1
                triangles_xy_shield.append([p0, p1, p2])
                # triangle 2
                triangles_xy_shield.append([p1, p3, p2])

        triangles_xy_shield = np.array(triangles_xy_shield)
        triangles_xz_shield=triangles_xy_shield.copy()
        triangles_yz_shield=triangles_xy_shield.copy()
        tri_xy_shield=trimesh.Trimesh(xy_shield,triangles_xy_shield)
        tri_xz_shield=trimesh.Trimesh(xz_shield,triangles_xz_shield)
        tri_yz_shield=trimesh.Trimesh(yz_shield,triangles_yz_shield)
        tri_xy_shield=tri_xy_shield.subdivide().subdivide()#.subdivide()
        tri_xz_shield=tri_xz_shield.subdivide().subdivide()#.subdivide()
        tri_yz_shield=tri_yz_shield.subdivide().subdivide()#.subdivide()

        x_shield = tri_xz_shield.vertices[:, 0]
        z_shield = tri_xz_shield.vertices[:, 2]
        faces_shield = tri_xz_shield.faces
        # Create triangulation object
        triang_shield = mtri.Triangulation(x_shield, z_shield, faces_shield)

        shield_loc = self.shield_dim / 2                                          # Location of the shield from the center (normal to surface) = 1/2 shield size (MSR is closed) -> compare to coil distance
        plus_off_xy_shield=np.array([0,0,shield_loc])
        minus_off_xy_shield=np.array([0,0,-shield_loc])
        plus_off_xz_shield=np.array([0,shield_loc,0])
        minus_off_xz_shield=np.array([0,-shield_loc,0])
        plus_off_yz_shield=np.array([shield_loc,0,0])
        minus_off_yz_shield=np.array([-shield_loc,0,0])

        shield_plus_xy=trimesh.Trimesh(
            tri_xy_shield.vertices+plus_off_xy_shield,tri_xy_shield.faces,process=False
        )
        shield_minus_xy=trimesh.Trimesh(
            tri_xy_shield.vertices+minus_off_xy_shield,tri_xy_shield.faces,process=False
        )
        shield_plus_xz=trimesh.Trimesh(
            tri_xz_shield.vertices+plus_off_xz_shield,tri_xz_shield.faces,process=False
        )
        shield_minus_xz=trimesh.Trimesh(
            tri_xz_shield.vertices+minus_off_xz_shield,tri_xz_shield.faces,process=False
        )
        shield_plus_yz=trimesh.Trimesh(
            tri_yz_shield.vertices+plus_off_yz_shield,tri_yz_shield.faces,process=False
        )
        shield_minus_yz=trimesh.Trimesh(
            tri_yz_shield.vertices+minus_off_yz_shield,tri_yz_shield.faces,process=False
        )

        total_shield = combine_meshes((
            shield_plus_xy, shield_minus_xy,
            shield_plus_xz, shield_minus_xz, 
            shield_plus_yz, shield_minus_yz
        ))

        # CRUCIAL: Repair seams/duplicates
        total_shield.merge_vertices()  # Merge nahe Vertices (Kanten)
        total_shield.fix_normals()
        # total_shield.fill_holes()  # Falls Löcher

        # Jetzt: Keine Boundary-Edges mehr (closed manifold)
        inner_idx_shield = bfieldtools.utils.find_mesh_boundaries(total_shield)
        print("Inner vertices nach Repair:", len(inner_idx_shield[0]))  # Sollte fast alle sein


        # From here on, plotting starts!!!
        fig, ax = plt.subplots(figsize=(6,6))
        ax.triplot(triang_shield, color='k', linewidth=0.8)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        inner_idx_shield=bfieldtools.utils.find_mesh_boundaries(total_shield)
        inner_vertex_idx_shield=inner_idx_shield[0]
        for idx in inner_vertex_idx_shield:
            boundary=total_shield.vertices[idx]
            ax.scatter(boundary[:,0],boundary[:,1],boundary[:,2])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        verts_shield=total_shield.vertices
        ax.scatter(verts_shield[:,0],verts_shield[:,1],verts_shield[:,2],alpha=0.5)

        points_inside = total_shield.vertices - self.shield_thickness * total_shield.vertex_normals
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(verts_shield[:,0],verts_shield[:,1],verts_shield[:,2],alpha=0.3)
        ax.scatter(points_inside[:,0],points_inside[:,1],points_inside[:,2],alpha=0.3)
