import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy
import os

mu_0=scipy.constants.mu_0
pi=scipy.constants.pi

def load_data_from_folder(folder_path_no_current, folder_path_with_current, L_x, L_y, L_z, step_size, shift_x, shift_y, shift_z):
    '''L_x, L_y, L_z: dimensions of the measurement volume in [m]
       step_size: step size of the measurement grid in [m]
       shift_x, shift_y, shift_z: shifts of the coordinate system in [m]'''

    target_point_coord = []
    target_point_coordinate_test = []
    B_target_point_no_current = []
    B_target_point_with_current = []

    for filename in os.listdir(folder_path_no_current):                  # iterates through all documents in the points folder
        if filename.endswith(".npz"):                                           # only load the .npz files
            file_path = os.path.join(folder_path_no_current, filename)
            data = np.load(file_path)                                           # load the data of the current point

            x = data['x_mm'] * 1e-3 - shift_x                                   # Umrechnung in [m] und Versetzen des Nullpunkts
            y = data['y_mm'] * 1e-3 - shift_y
            z = data['z_mm'] * 1e-3 - shift_z

            mean_Bx = data['mean_Bx_pT'] * 1e-12                                # Umrechnen in [T]
            mean_By = data['mean_By_pT'] * 1e-12
            mean_Bz = data['mean_Bz_pT'] * 1e-12

            
            each_target_point = np.array([x, y, z]).T                           # (Npoints, 3)
            each_B_target = np.array([mean_Bx, mean_By, mean_Bz]).T             # (Npoints, 3)

            target_point_coord.append(each_target_point)
            B_target_point_no_current.append(each_B_target)

    # Stack all files into final (total_Npoints, 3) arrays
    target_point_coord = np.vstack(target_point_coord)                          # (total_Npoints, 3)
    B_target_point_no_current = np.vstack(B_target_point_no_current)              # (total_Npoints, 3)

    i=0
    for filename in os.listdir(folder_path_with_current):                # iterates through all documents in the points folder
        if filename.endswith(".npz"):                                           # only load the .npz files
            file_path = os.path.join(folder_path_with_current, filename)
            data = np.load(file_path)                                           # load the data of the current point

            x = data['x_mm'] * 1e-3 - shift_x                                   # Umrechnung in [m] und Versetzen des Nullpunkts
            y = data['y_mm'] * 1e-3 - shift_y
            z = data['z_mm'] * 1e-3 - shift_z
            
            each_target_point = np.array([x, y, z]).T                           # (Npoints, 3)
            target_point_coordinate_test.append(each_target_point)

            if 'mean_Bx_pT' in data and 'mean_By_pT' in data and 'mean_Bz_pT' in data:
        
                mean_Bx = data['mean_Bx_pT'] * 1e-12                            # Umrechnen in [T]
                mean_By = data['mean_By_pT'] * 1e-12
                mean_Bz = data['mean_Bz_pT'] * 1e-12
            
            else:
                mean_Bx = np.nan                                                # These values will get interpolated later on!
                mean_By = np.nan
                mean_Bz = np.nan

                print(f'Warning: There are empty measurements which do not contain any B-field information except the measurement position at ({each_target_point}).')

            each_B_target = np.array([mean_Bx, mean_By, mean_Bz]).T             # (Npoints, 3)
            B_target_point_with_current.append(each_B_target)

    

    # Stack all files into final (total_Npoints, 3) arrays
    target_point_coordinate_test = np.vstack(target_point_coordinate_test)       # (total_Npoints, 3)
    B_target_point_with_current = np.vstack(B_target_point_with_current)         # (total_Npoints, 3)

    # Interpolate missing B-field values in the with-current measurements
    missing_rows = np.any(np.isnan(B_target_point_with_current), axis=1)
    if np.any(missing_rows):
        valid_rows = missing_rows == False
        valid_points = target_point_coordinate_test[valid_rows]
        missing_points = target_point_coordinate_test[missing_rows]
        valid_B = B_target_point_with_current[valid_rows]

        for comp in range(3):
            interpolated = scipy.interpolate.griddata(
                valid_points,
                valid_B[:, comp],
                missing_points,
                method = 'linear',
                fill_value = np.nan
            )

            if np.any(np.isnan(interpolated)):
                nearest = scipy.interpolate.griddata(
                    valid_points,
                    valid_B[:, comp],
                    missing_points[np.isnan(interpolated)],
                    method='nearest'
                )
                interpolated[np.isnan(interpolated)] = nearest

            B_target_point_with_current[missing_rows, comp] = interpolated

        print(f'Interpolated {missing_rows.sum()} missing B-field row(s) in with-current target data.')

    if target_point_coord.shape != target_point_coordinate_test.shape:           # Note: This only checks if the array shapes are equal. Not its entries!!!
        raise ValueError(f'The coordinates of the target points with and without current do not match. (without: {target_point_coord.shape}; with: {target_point_coordinate_test.shape}) Please check the data files.')
        # Note: The mapper does not necessarily take the same serpentine in both maps, which is why they can differ and we can only compare shapes
    # Debugging
    print(f'Target point coordinates and B-field at target points both need to have dimension (n_points, 3).\n  Shape of target coordinates: {target_point_coord.shape}\n  Shape of B-field with current at target points: {B_target_point_with_current.shape}\n  Shape of B-field without current at target points: {B_target_point_no_current.shape}')

    return target_point_coord, B_target_point_no_current, B_target_point_with_current

def plot_Experiment_results_coils(target_point_coord, B_target_point_no_current, B_target_point_with_current):

    x = target_point_coord[:, 0]
    y = target_point_coord[:, 1]
    z = target_point_coord[:, 2]

    Bx_no_I = B_target_point_no_current[:, 0]
    By_no_I = B_target_point_no_current[:, 1]
    Bz_no_I = B_target_point_no_current[:, 2]
    Bnorm_no_I = np.linalg.norm(B_target_point_no_current, axis=1)

    Bx_with_I = B_target_point_with_current[:, 0]
    By_with_I = B_target_point_with_current[:, 1]
    Bz_with_I = B_target_point_with_current[:, 2]
    Bnorm_with_I = np.linalg.norm(B_target_point_with_current, axis=1)

    fig = plt.figure(figsize=(18, 28))
    axs = [
        fig.add_subplot(4, 2, 1, projection='3d'),
        fig.add_subplot(4, 2, 3, projection='3d'),
        fig.add_subplot(4, 2, 5, projection='3d'),
        fig.add_subplot(4, 2, 7, projection='3d'),
        fig.add_subplot(4, 2, 2, projection='3d'),
        fig.add_subplot(4, 2, 4, projection='3d'),
        fig.add_subplot(4, 2, 6, projection='3d'),
        fig.add_subplot(4, 2, 8, projection='3d')
    ]

    component_data = [Bx_no_I, By_no_I, Bz_no_I, Bnorm_no_I, Bx_with_I, By_with_I, Bz_with_I, Bnorm_with_I]
    v_min = min(d.min() for d in component_data)
    v_max = max(d.max() for d in component_data)
    titles = [
        r'$B_x$ without current [T]',
        r'$B_y$ without current [T]',
        r'$B_z$ without current [T]',
        r'$|B|$ without current [T]',
        r'$B_x$ with current [T]',
        r'$B_y$ with current [T]',
        r'$B_z$ with current [T]',
        r'$|B|$ with current [T]'
    ]
    plots = [
        (Bx_no_I, r'$B_x$ [T]', v_min, v_max),
        (By_no_I, r'$B_y$ [T]', v_min, v_max),
        (Bz_no_I, r'$B_z$ [T]', v_min, v_max),
        (Bnorm_no_I, r'$|B|$ [T]', v_min, v_max),
        (Bx_with_I, r'$B_x$ [T]', v_min, v_max),
        (By_with_I, r'$B_y$ [T]', v_min, v_max),
        (Bz_with_I, r'$B_z$ [T]', v_min, v_max),
        (Bnorm_with_I,r'$|B|$ [T]', v_min, v_max)
    ]
    for ax, (data, label, vmin, vmax), title in zip(axs, plots, titles):
        sc = ax.scatter(x, y, z, c=data, s=100, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        ax.set_zlabel(r'$z$ [m]')
        plt.colorbar(sc, ax=ax, label=label)

    # fig.suptitle(f'First Measurement of coil influence on B-field at target points', fontsize = 20)
    plt.tight_layout()
    plt.show()


def plot_stream_function(vertices, stream_function):

    v_min = min(stream_function)
    v_max = max(stream_function)
    c_map = 'RdBu_r'                # Rot-Blau Colormap (symmetrisch)
    s=40                            # Punktgröße
    alpha=0.8                       # Transparenz der Punkte

    fig_3d = plt.figure(figsize=(12, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')  # Fix: fig_3d statt fig

    scatter_3d = ax_3d.scatter(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        c=stream_function, cmap='RdBu_r', s=40, alpha=0.8, edgecolors = 'none'
    )

    cbar_3d = plt.colorbar(scatter_3d, shrink=0.6, pad=0.1)
    cbar_3d.set_label(r'Stromfunktion $\psi$', fontsize=14, rotation=270, labelpad=20)

    ax_3d.set_xlabel('X [m]', fontsize=12)
    ax_3d.set_ylabel('Y [m]', fontsize=12)
    ax_3d.set_zlabel('Z [m]', fontsize=12)
    ax_3d.set_title('Stromfunktion auf Coil-Oberfläche', fontsize=16, pad=20)

    ax_3d.set_xlim(vertices[:,0].min()*1.05, vertices[:,0].max()*1.05)
    ax_3d.set_ylim(vertices[:,1].min()*1.05, vertices[:,1].max()*1.05)
    ax_3d.set_zlim(vertices[:,2].min()*1.05, vertices[:,2].max()*1.05)

    plt.tight_layout()
    plt.show()
    

    N = int(vertices.shape[0]/6)

    fig_faces = plt.figure(figsize = (24, 8))

    ax1 = fig_faces.add_subplot(2, 3, 1)
    scatter1 = ax1.scatter(
        vertices[:N, 0],
        vertices[:N, 1],
        c = stream_function[:N],     # Farbe nach Streamfunktion
        cmap = c_map,
        s = s,                
        vmin = v_min,
        vmax = v_max,
        alpha = alpha,
        edgecolors = 'none'
    )
    cbar1 = plt.colorbar(scatter1, shrink=0.6, pad=0.1)
    cbar1.set_label(r'Stromfunktion $\psi$', fontsize=14, rotation=270, labelpad=20)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title('Top')

    ax2 = fig_faces.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(
        vertices[N:2*N, 0],
        vertices[N:2*N, 1],
        c = stream_function[N:2*N],     # Farbe nach Streamfunktion
        cmap = c_map,
        s = s,                
        vmin = v_min,
        vmax = v_max,
        alpha = alpha,
        edgecolors = 'none'
    )
    cbar2 = plt.colorbar(scatter2, shrink=0.6, pad=0.1)
    cbar2.set_label(r'Stromfunktion $\psi$', fontsize=14, rotation=270, labelpad=20)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title('Bottom')

    ax3 = fig_faces.add_subplot(2, 3, 3)
    scatter3 = ax3.scatter(
        vertices[2*N:3*N, 0],
        vertices[2*N:3*N, 2],
        c = stream_function[2*N:3*N],     # Farbe nach Streamfunktion
        cmap = c_map,
        s = s,                
        vmin = v_min,
        vmax = v_max,
        alpha = alpha,
        edgecolors = 'none'
    )
    cbar3 = plt.colorbar(scatter3, shrink=0.6, pad=0.1)
    cbar3.set_label(r'Stromfunktion $\psi$', fontsize=14, rotation=270, labelpad=20)
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_title('Back')

    ax4 = fig_faces.add_subplot(2, 3, 4)
    scatter4 = ax4.scatter(
        vertices[3*N:4*N, 0],
        vertices[3*N:4*N, 2],
        c = stream_function[3*N:4*N],     # Farbe nach Streamfunktion
        cmap = c_map,
        s = s,                
        vmin = v_min,
        vmax = v_max,
        alpha = alpha,
        edgecolors = 'none'
    )
    cbar4 = plt.colorbar(scatter4, shrink=0.6, pad=0.1)
    cbar4.set_label(r'Stromfunktion $\psi$', fontsize=14, rotation=270, labelpad=20)
    ax4.set_aspect('equal', adjustable='box')
    ax4.set_title('Front')


    ax5 = fig_faces.add_subplot(2, 3, 5)
    scatter5 = ax5.scatter(
        vertices[4*N:5*N, 1],
        vertices[4*N:5*N, 2],
        c = stream_function[4*N:5*N],     # Farbe nach Streamfunktion
        cmap = c_map,
        s = s,                
        vmin = v_min,
        vmax = v_max,
        alpha = alpha,
        edgecolors = 'none'
    )
    cbar5 = plt.colorbar(scatter5, shrink=0.6, pad=0.1)
    cbar5.set_label(r'Stromfunktion $\psi$', fontsize=14, rotation=270, labelpad=20)
    ax5.set_aspect('equal', adjustable='box')
    ax5.set_title('right')

    ax6 = fig_faces.add_subplot(2, 3, 6)
    scatter6 = ax6.scatter(
        vertices[5*N:6*N, 1],
        vertices[5*N:6*N, 2],
        c = stream_function[5*N:6*N],     # Farbe nach Streamfunktion
        cmap = c_map,
        s = s,                
        vmin = v_min,
        vmax = v_max,
        alpha = alpha,
        edgecolors = 'none'
    )
    cbar6 = plt.colorbar(scatter6, shrink=0.6, pad=0.1)
    cbar6.set_label(r'Stromfunktion $\psi$', fontsize=14, rotation=270, labelpad=20)
    ax6.set_aspect('equal', adjustable='box')
    ax6.set_title('left')

    plt.tight_layout()
    plt.show()

def calculation_target_points(n, coil_plane_dist_to_origin_x, coil_plane_dist_to_origin_y, coil_plane_dist_to_origin_z, safety_distance):
    target_point_coord_calc = []
    point_coord_calc_x = np.linspace(-(coil_plane_dist_to_origin_x - safety_distance), +coil_plane_dist_to_origin_x - safety_distance, n)
    point_coord_calc_y = np.linspace(-(coil_plane_dist_to_origin_y - safety_distance), +coil_plane_dist_to_origin_y - safety_distance, n)
    point_coord_calc_z = np.linspace(-(coil_plane_dist_to_origin_z - safety_distance), +coil_plane_dist_to_origin_z - safety_distance, n)

    for idx_x in range(n):
        for idx_y in range(n):
            for idx_z in range(n):
                target_point_coord_calc.append([point_coord_calc_x[idx_x], point_coord_calc_y[idx_y], point_coord_calc_z[idx_z]])

    target_point_coord_calc = np.array(target_point_coord_calc)
    return target_point_coord_calc

def interpolate_B_on_coarse_grid(num_calc_target_points_fine, target_point_coord_calc_coarse, coil_plane_dist_to_origin_x, coil_plane_dist_to_origin_y, coil_plane_dist_to_origin_z, safety_distance, B_coil_predicted_fine):
    '''Note: This function is currently not used in the code. Still it is kept and not accessed just in case ;)'''

    x_fine = np.linspace(-(coil_plane_dist_to_origin_x - safety_distance), +(coil_plane_dist_to_origin_x - safety_distance), num_calc_target_points_fine)
    y_fine = np.linspace(-(coil_plane_dist_to_origin_y - safety_distance), +(coil_plane_dist_to_origin_y - safety_distance), num_calc_target_points_fine)
    z_fine = np.linspace(-(coil_plane_dist_to_origin_z - safety_distance), +(coil_plane_dist_to_origin_z - safety_distance), num_calc_target_points_fine)

    n = int((B_coil_predicted_fine.shape[0]+1)**(1/3))
    B_coil_predicted_fine_reshaped = B_coil_predicted_fine.reshape(n, n, n, 3)

    Bx_interp = scipy.interpolate.RegularGridInterpolator((x_fine, y_fine, z_fine), B_coil_predicted_fine_reshaped[..., 0], bounds_error=False, fill_value=None)
    By_interp = scipy.interpolate.RegularGridInterpolator((x_fine, y_fine, z_fine), B_coil_predicted_fine_reshaped[..., 1], bounds_error=False, fill_value=None)
    Bz_interp = scipy.interpolate.RegularGridInterpolator((x_fine, y_fine, z_fine), B_coil_predicted_fine_reshaped[..., 2], bounds_error=False, fill_value=None)

    Bx_new = Bx_interp(target_point_coord_calc_coarse)
    By_new = By_interp(target_point_coord_calc_coarse)
    Bz_new = Bz_interp(target_point_coord_calc_coarse)

    B_coil_predicted_coarse = np.column_stack((Bx_new, By_new, Bz_new))
    return B_coil_predicted_coarse

def points_close(p1, p2, tol):
    return np.linalg.norm(p1 - p2) < tol

def stitch_segments(segments, tol=1e-6):
    """
    segments: list of arrays, each array has shape (2, 3) mening two points in 3D space
    returns: list of ordered polylines
    """
    remaining = [seg.copy() for seg in segments if len(seg) > 0]        # Create a copy of the segments list to modify while iterating
    ordered_lines = []                                                  # This will hold the final ordered polylines

    while remaining:                                                    # While there are still segments to process
        line = remaining.pop(0).copy()                                  # Start a new line with the first segment and remove it from the remaining list

        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(remaining):
                seg = remaining[i]

                start_line = line[0]
                end_line = line[-1]
                start_seg = seg[0]
                end_seg = seg[-1]

                # Check if the start or end of the current line is close to the start or end of the segment (in any combination)
                if points_close(end_line, start_seg, tol):
                    line = np.vstack([line, seg[1:]])
                    remaining.pop(i)
                    changed = True
                    continue
                elif points_close(end_line, end_seg, tol):
                    line = np.vstack([line, seg[-2::-1]])
                    remaining.pop(i)
                    changed = True
                    continue
                elif points_close(start_line, end_seg, tol):
                    line = np.vstack([seg[:-1], line])
                    remaining.pop(i)
                    changed = True
                    continue
                elif points_close(start_line, start_seg, tol):
                    line = np.vstack([seg[::-1][:-1], line])
                    remaining.pop(i)
                    changed = True
                    continue

                i += 1

        ordered_lines.append(line)

    return ordered_lines


def find_all_contours(stream_func_coil, all_coords_verts, Steps, refinement_factor):
    # Reshape stream function for each plane (if needed for plotting or further analysis)
    vertices_per_plane = int(len(stream_func_coil) / 6)  # All 6 planes in the coil layout have the same numbers of vertices by construction
    stream_func_each_plane = stream_func_coil.reshape(6, vertices_per_plane)


    # Get coordinates on surface
    coords = all_coords_verts.reshape(6, vertices_per_plane, 3)
    x = coords[:, :, 0]
    y = coords[:, :, 1]
    z = coords[:, :, 2]


    # Create contour levels with desired spacing
    # The spacing value corresponds to the current in the wires, since the stream function is proportional to the current. Adjust this value to get more or fewer contour lines.
    contour_levels = np.linspace(stream_func_coil.min(), stream_func_coil.max(), Steps)
    print(f'There are {len(contour_levels)} global contour levels with a current of {(stream_func_coil.max()-stream_func_coil.min())/Steps} A in the stream function.')
    print('Note, that the more wires will lead to a closer approximation of the desired field, but also to a more complex coil layout and higher fabrication costs.')


    all_contours = []
    # NEW: Boolean array to track if contour is above zero (True) or below zero (False)
    # Shape will be (6 faces, number_of_contours_per_face) - matches all_contours structure
    contour_is_positive = []
    
    for face in range(6):
        print(f"\nProcessing plane {face + 1}/6...")
        
        # Get data for this specific plane
        x_plane = x[face]  # shape (vertices_per_plane,)
        y_plane = y[face]  # shape (vertices_per_plane,)
        z_plane = z[face]  # shape (vertices_per_plane,)
        stream_plane = stream_func_each_plane[face]  # shape (vertices_per_plane,)
        
        # Find which coordinate is constant (plane orientation)
        x_range = x_plane.max() - x_plane.min()
        y_range = y_plane.max() - y_plane.min()
        z_range = z_plane.max() - z_plane.min()
        
        ranges = [x_range, y_range, z_range]
        const_dim = np.argmin(ranges)           # 0=x, 1=y, 2=z is constant
        
        # Project to 2D based on plane orientation
        if const_dim == 0:                      # yz-plane (x is constant)
            u = y_plane
            v = z_plane
            plane_offset = x_plane[0]
            coord_map = [1, 2, 0]               # [v_idx, const_idx, u_idx]
        elif const_dim == 1:                    # xz-plane (y is constant)
            u = x_plane
            v = z_plane
            plane_offset = y_plane[0]
            coord_map = [0, 2, 1]               # [u_idx, const_idx, v_idx]
        elif const_dim == 2:                    # xy-plane (z is constant)
            u = x_plane
            v = y_plane
            plane_offset = z_plane[0]
            coord_map = [0, 1, 2]               # [u_idx, v_idx, const_idx]
            
        
        ui_2d = np.linspace(u.min(), u.max(), refinement_factor)
        vi_2d = np.linspace(v.min(), v.max(), refinement_factor)
        ui_mesh, vi_mesh = np.meshgrid(ui_2d, vi_2d, indexing='ij')


        # Interpolate stream function onto 2D grid (linearly)
        stream_2d = scipy.interpolate.griddata((u, v), stream_plane,
                         (ui_mesh, vi_mesh), method='linear')
        
        
        # Initialize contour list for this plane
        plane_contours = []
        
        # NEW: Initialize boolean list for this plane
        plane_contour_is_positive = []
        
        # For each contour level, extract the contour points
        for level in contour_levels:          # Iterate through all contour levels and extract the contour points for each level
            # Skip if level is outside the valid range for this plane
            plane_min = stream_2d.min()
            plane_max = stream_2d.max()
            
            if level < plane_min or level > plane_max:
                continue
            
            # Find contour point by marching through grid cells
            level_segments_2d = []                          # Holds a list of arrays, which contain the two crossing points of the contour of every cell in the 2D-plane
            for i in range(len(ui_2d) - 1):
                for j in range(len(vi_2d) - 1):
                    # Get 4 corners of this cell
                    s00 = stream_2d[i, j]                   # Value of interpolated stream function at the vertex {i, j}
                    s10 = stream_2d[i+1, j]                 # Value of interpolated stream function at the vertex {i+1, j}
                    s01 = stream_2d[i, j+1]                 # Value of interpolated stream function at the vertex {i, j+1}
                    s11 = stream_2d[i+1, j+1]               # Value of interpolated stream function at the vertex {i+1, j+1}


                    # Skip if any value is NaN
                    if any(np.isnan(v) for v in [s00, s10, s01, s11]):
                        continue
                    
                    # Find if level crosses the cell
                    diff = [s00 - level, s10 - level, s01 - level, s11 - level]
                    
                    # Check edges for crossings
                    intersections = []                      # This array holds the two points, where the contour enters and leaves the cell. (Array of 2 2D-points)
                    
                    # Bottom edge (0-1)
                    if diff[0] * diff[1] < 0:               # If the difference between the current level and the streamfunction changes sign -> there is a crossing
                        t = -diff[0] / (diff[1] - diff[0])  # linear interpolation to find the crossing point along the cell edge
                        u_int = ui_2d[i] + t * (ui_2d[i+1] - ui_2d[i])
                        v_int = vi_2d[j]
                        intersections.append((u_int, v_int))
                    
                    # Left edge (0-2)
                    if diff[0] * diff[2] < 0:
                        t = -diff[0] / (diff[2] - diff[0])
                        u_int = ui_2d[i]
                        v_int = vi_2d[j] + t * (vi_2d[j+1] - vi_2d[j])
                        intersections.append((u_int, v_int))
                    
                    # Right edge (1-3)
                    if diff[1] * diff[3] < 0:
                        t = -diff[1] / (diff[3] - diff[1])
                        u_int = ui_2d[i+1]
                        v_int = vi_2d[j] + t * (vi_2d[j+1] - vi_2d[j])
                        intersections.append((u_int, v_int))
                    
                    # Top edge (2-3)
                    if diff[2] * diff[3] < 0:
                        t = -diff[2] / (diff[3] - diff[2])
                        u_int = ui_2d[i] + t * (ui_2d[i+1] - ui_2d[i])
                        v_int = vi_2d[j+1]
                        intersections.append((u_int, v_int))
                    
                    # If we have 2 intersections (entering and exiting), we have a contour segment
                    if len(intersections) == 2:
                        level_segments_2d.append(np.array(intersections))
            
            
            level_segments_3d = []              # Initialise the 3D segment array. In this array the segment coordinates (2 2D-points) are transformed back to 3D coordinates by adding the offset
            for seg2d in level_segments_2d:
                seg3d = np.zeros((2, 3))
                seg3d[:, coord_map[0]] = seg2d[:, 0]
                seg3d[:, coord_map[1]] = seg2d[:, 1]
                seg3d[:, coord_map[2]] = plane_offset
                level_segments_3d.append(seg3d)         # This array holds the segments in the order in which they were found. (small (u, v) -> large (u, v)) not in order


            if len(level_segments_3d) > 0:              # If the contour exists, stitch it together, such that all the 2 3D-point segments are connected. This also means getting rid of the second copy of each point!
                stitched_level_contours = stitch_segments(level_segments_3d, tol=1e-8)
                plane_contours.extend(stitched_level_contours)
                
                is_positive = level > 0
                plane_contour_is_positive.extend([is_positive] * len(stitched_level_contours))
        
        # Add this plane's contours to the master list
        all_contours.append(plane_contours)             # This list contains the coordinates of all points, of all contours, of all faces.
        
        contour_is_positive.append(plane_contour_is_positive)
        
        print(f"    Plane {face + 1}: extracted {len(plane_contours)} contours")


    # Return both the contours and the boolean array
    return all_contours, contour_is_positive

def plot_contours(all_contours, stream_func_coil, total_coord, Steps):
    # plot the isolines of the streamfunction for each plane
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    face_titles = [
        "Face 1: Top (xy-plane)",
        "Face 2: Bottom (xy-plane)",
        "Face 3: Back (xz-plane)",
        "Face 4: Front (xz-plane)",
        "Face 5: Right (yz-plane)",
        "Face 6: Left (yz-plane)"
    ]

    vertices_per_plane = int(len(stream_func_coil) / 6)
    stream_func_each_plane = stream_func_coil.reshape(6, vertices_per_plane)
    coords = total_coord.reshape(6, vertices_per_plane, 3)
    x = coords[:, :, 0]
    y = coords[:, :, 1]
    z = coords[:, :, 2]

    for face_idx in range(6):
        ax = axes[face_idx]
        ax.set_title(face_titles[face_idx], fontsize=12)

        x_face = x[face_idx]
        y_face = y[face_idx]
        z_face = z[face_idx]
        scalar_face = stream_func_each_plane[face_idx]

        x_range = x_face.max() - x_face.min()
        y_range = y_face.max() - y_face.min()
        z_range = z_face.max() - z_face.min()

        const_dim = np.argmin([x_range, y_range, z_range])

        if const_dim == 0:  # yz-plane
            u_face = y_face
            v_face = z_face
            ax.set_xlabel('y [m]')
            ax.set_ylabel('z [m]')
        elif const_dim == 1:  # xz-plane
            u_face = x_face
            v_face = z_face
            ax.set_xlabel('x [m]')
            ax.set_ylabel('z [m]')
        else:  # xy-plane
            u_face = x_face
            v_face = y_face
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')

        triang = mtri.Triangulation(u_face, v_face)

        bg = ax.tricontourf(triang, scalar_face, levels=30, cmap='viridis', alpha=0.75)

        for contour_idx in range(len(all_contours[face_idx])):
            contour = all_contours[face_idx][contour_idx]

            if const_dim == 0:
                ax.plot(contour[:, 1], contour[:, 2], 'b-', linewidth=1.5)
            elif const_dim == 1:
                ax.plot(contour[:, 0], contour[:, 2], 'b-', linewidth=1.5)
            elif const_dim == 2:
                ax.plot(contour[:, 0], contour[:, 1], 'b-', linewidth=1.5)

        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    print(f'The current which needs to flow in every wire is {((stream_func_coil.max()-stream_func_coil.min())/Steps) * 10**3:.3f} mA.')


def reparameterize_by_arc_length(points, n_resample):
    """
    Reparameterize a contour (n_points, 3) by arc length for fft preperation.

    Parameters
    points : ndarray
        Shape (n_points, 3), ordered along the contour.
    n_resample : int
        Number of uniformly spaced points along arc length.

    Returns
    s_uniform : ndarray
        Uniform arc-length coordinate (n_resample,)
        (from 0 to total_length)
    points_uniform : ndarray
        Resampled contour points (n_resample, 3)
    """
    points = np.asarray(points)                                     # reformulate to np array (from list of lists)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")

    # Cumulative arc length
    diffs = np.diff(points, axis=0)                                 # Compute vektor between two points
    dists = np.linalg.norm(diffs, axis=1)                           # Calculate length of mentioned vector
    s = np.concatenate(([0.0], np.cumsum(dists)))                   # np.array holding [[0], [length]]
    total_length = s[-1]

    if total_length == 0:
        # Degenerate contour: all points at same location
        return s, points.copy()

    s_uniform = np.linspace(0, total_length, n_resample)            # Create np.array to hold interpolated coordinates

    # Interpolate each coordinate
    x_interp = scipy.interpolate.interp1d(s, points[:, 0], kind='linear')
    y_interp = scipy.interpolate.interp1d(s, points[:, 1], kind='linear')
    z_interp = scipy.interpolate.interp1d(s, points[:, 2], kind='linear')

    points_uniform = np.column_stack([
        x_interp(s_uniform),
        y_interp(s_uniform),
        z_interp(s_uniform)
    ])

    return s_uniform, points_uniform


def smooth_1d_periodic(signal, n_keep):
    """
    Smooth a periodic (prepared) 1D signal by keeping only the lowest n_keep Fourier modes
    on each side of the spectrum.
    """
    signal = np.asarray(signal)                 # np.array holding the coordinates in one direction
    N = len(signal)

    coeffs = scipy.fft.fft(signal)                        # now contains the fft of the coordinate in one direction

    filtered = np.zeros_like(coeffs)            # shall hold the truncated fft

    n_keep = max(1, min(n_keep, N // 2))        # At least one frequency must be kept! For very short contours, hold less frequencies (N//2)
    filtered[:n_keep + 1] = coeffs[:n_keep + 1] # copies the DC and lowest positive frequencies
    filtered[-n_keep:] = coeffs[-n_keep:]       # copies the negative-frequency side of the spectrum
                                                # Note: Even though for a real signal, the sdes are simply mirrored, we copy both sides of the fft for robustness.
    smoothed = scipy.fft.ifft(filtered).real              # performs inverse fft to retrieve function in "time-space". Note that the function is purely real!
    
    return smoothed


def smooth_contour_points(points, n_keep, n_resample):
    """
    Smooth one contour of shape (n_points, 3), with non-uniform spacing:
      1. Reparameterize by arc length
      2. FFT smoothing on uniform grid
      3. Optionally enforce closure

    Parameters
    points : ndarray
        Shape (n_points, 3)
    n_keep : int
        Number of Fourier modes to keep.
    n_resample : int
        Number of points after arc-length resampling.
    close_contour : bool
        If True, treat contour as periodic and enforce start==end.

    Returns
    smoothed_points : ndarray
        Shape (n_resample, 3)
    """
    # 1. Arc-length reparameterization (interpolate points of contour such that they have equal lengths)
    s_uniform, points_uniform = reparameterize_by_arc_length(points, n_resample)

    # 2. FFT smoothing for each coordinate
    smoothed = np.empty_like(points_uniform, dtype=float)
    for i in range(3):
        smoothed[:, i] = smooth_1d_periodic(points_uniform[:, i], n_keep)

    # 3. Enforce closure if needed
    smoothed[-1] = smoothed[0]

    return smoothed


def smooth_all_contours_from_list(all_contours, n_keep, n_resample):
    """
    Smooth contours stored as a list-of-lists structure:

        all_contours[face_idx][contour_idx] -> (n_points, 3) array

    Returns a new list of lists with the same structure, but with smoothed
    contours of shape (n_resample, 3).

    Parameters
    all_contours : list
        List of length 6, each element is a list of contours for that face.
    n_keep : int
        Number of Fourier modes to keep.
    n_resample : int
        Number of points after arc-length resampling.
    close_contour : bool
        If True, treat contours as periodic.

    Returns
    smoothed_contours : list
        Same structure as all_contours, but with smoothed arrays.
    """
    if not isinstance(all_contours, list) or len(all_contours) != 6:
        raise ValueError("all_contours must be a list of length 6 (faces)")

    smoothed_contours = []

    for face_idx in range(6):
        face_contours = all_contours[face_idx]
        if not isinstance(face_contours, list):
            raise ValueError(f"all_contours[{face_idx}] must be a list of contours")

        smoothed_face = []
        for contour_idx, points in enumerate(face_contours):
            points = np.asarray(points)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"Contour at face {face_idx}, contour {contour_idx} must be shape (n_points, 3), got {points.shape}")

            smoothed = smooth_contour_points(points, n_keep, n_resample)
            smoothed_face.append(smoothed)

        smoothed_contours.append(smoothed_face)

    return smoothed_contours

def biot_savart_from_contours(contours, target_point_coord_exp, current_per_wire):
    """
    Compute magnetic field B at target points from contour wires using
    discretized Biot-Savart law.

    Parameters
    contours : list
        Nested list structure:
            contours[face_idx][contour_idx] -> array of shape (n_points, 3)
        Each contour is interpreted as a polyline wire.
    target_point_coord_exp : ndarray
        Shape (N_target, 3), observation points.
    current_per_wire : float
        Current flowing through each contour wire [A].

    Returns
    B : ndarray
        Shape (N_target, 3), magnetic field at each target point [T].
    """
    target_point_coord_exp = np.asarray(target_point_coord_exp, dtype=float)
    if target_point_coord_exp.ndim != 2 or target_point_coord_exp.shape[1] != 3:
        raise ValueError("target_point_coord_exp must have shape (N_target, 3)")

    B_total = np.zeros_like(target_point_coord_exp, dtype=float)

    prefactor = mu_0 * current_per_wire / (4.0 * np.pi)

    for face in contours:
        for contour in face:
            pts = np.asarray(contour, dtype=float)              # Make list to array
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError("Each contour must have shape (n_points, 3)")
            if len(pts) < 2:
                continue

            # If contour is closed, ensure last point connects back to first.
            # Only add closing segment if not already closed.
            if not np.allclose(pts[0], pts[-1]):
                pts = np.vstack([pts, pts[0]])

            r1 = pts[:-1]           # segment start points, shape (Nseg, 3)
            r2 = pts[1:]            # segment end points, shape (Nseg, 3)
            dl = r2 - r1            # segment vectors, shape (Nseg, 3)
            mid = 0.5 * (r1 + r2)   # segment centers, shape (Nseg, 3)

            # Vector from segment center to target points:
            # R has shape (N_target, Nseg, 3)
            R = target_point_coord_exp[:, None, :] - mid[None, :, :]    # R is the vector from the middle of the line segment to the target point
            R_norm = np.linalg.norm(R, axis=2)                          # Norm over axis where coordinates are

            # Cross product dl x R for each target and segment
            dl_b = dl[None, :, :]       # shape (1, Nseg, 3)
            cross = np.cross(dl_b, R)   # shape (N_target, Nseg, 3)
            R_norm_cubed = R_norm**3    # Cubed norm, Note, that R_norm_cubed can not be zero, since the target points should not be on the coil planes
            if np.any(R_norm_cubed) == 0:
                raise ValueError("Target point for B-field lies on coil plane! -> Check setup!!!")

            dB = prefactor * cross / R_norm_cubed[:, :, None]   # Biot Savart
            B_total += np.sum(dB, axis=1)                       # Add up all contributions from different line segments to the B-field of one point

    return B_total

def plot_singular_values(C, normalize=True, log_scale=True):
    C = np.asarray(C)

    # Full SVD; singular values are sorted descending
    s = np.linalg.svd(C, compute_uv=False)

    if normalize:
        s_plot = s / s[0]
        ylabel = r"$\sigma_i / \sigma_{\max}$"
    else:
        s_plot = s
        ylabel = r"$\sigma_i$"

    idx = np.arange(len(s_plot))

    plt.figure(figsize=(8, 5))
    plt.plot(idx, s_plot, lw=2)
    plt.xlabel("Singular value index")
    plt.ylabel(ylabel)
    if log_scale:
        plt.yscale("log")
    plt.grid(True, which="major", ls="-", alpha=1)
    plt.grid(True, which="minor", ls="--", alpha=0.5)
    plt.title("Singular values of total coupling matrix")
    plt.tight_layout()
    plt.show()

    return

def interpolate_cartesian_grid(target_points, B_fields, n_points_longest):
    """
    Interpolate both coordinates and magnetic fields on a homogeneously spaced
    Cartesian grid with n_points_longest along the longest dimension.
    """
    
    # Step 1: Extract the grid structure from target_points
    x_unique = np.unique(target_points[:, 0])
    y_unique = np.unique(target_points[:, 1])
    z_unique = np.unique(target_points[:, 2])
    
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    
    print(f"Original grid: {nx} × {ny} × {nz} = {nx*ny*nz} points")
    
    # Step 2: Determine grid dimensions
    x_range = x_unique.max() - x_unique.min()
    y_range = y_unique.max() - y_unique.min()
    z_range = z_unique.max() - z_unique.min()
    
    ranges = [x_range, y_range, z_range]
    longest_idx = np.argmax(ranges)
    longest_range = ranges[longest_idx]
    
    print(f"Grid ranges: X={x_range:.3f}, Y={y_range:.3f}, Z={z_range:.3f}")
    
    # Step 3: Calculate number of points for each dimension
    spacing = longest_range / (n_points_longest - 1)
    
    nx_new = max(2, int(round(x_range / spacing + 1)))
    ny_new = max(2, int(round(y_range / spacing + 1)))
    nz_new = max(2, int(round(z_range / spacing + 1)))
    
    print(f"New grid: {nx_new} × {ny_new} × {nz_new} = {nx_new*ny_new*nz_new} points")
    
    # Step 4: Create new uniformly spaced grid coordinates
    x_new = np.linspace(x_unique.min(), x_unique.max(), nx_new)
    y_new = np.linspace(y_unique.min(), y_unique.max(), ny_new)
    z_new = np.linspace(z_unique.min(), z_unique.max(), nz_new)
    
    # Step 5: Create 3D grid for original coordinates (for griddata)
    X_orig, Y_orig, Z_orig = np.meshgrid(x_unique, y_unique, z_unique, indexing='ij')
    
    # Step 6: Stack original coordinates into (n_points, 3) format for griddata
    points_orig = np.column_stack([X_orig.ravel(), Y_orig.ravel(), Z_orig.ravel()])
    
    # Step 7: Reshape B_fields into 3D array
    B_x_3d = np.zeros((nx, ny, nz))
    B_y_3d = np.zeros((nx, ny, nz))
    B_z_3d = np.zeros((nx, ny, nz))
    
    for i in range(len(target_points)):
        xi, yi, zi = target_points[i]
        ix = np.where(x_unique == xi)[0][0]
        iy = np.where(y_unique == yi)[0][0]
        iz = np.where(z_unique == zi)[0][0]
        B_x_3d[ix, iy, iz] = B_fields[i, 0]
        B_y_3d[ix, iy, iz] = B_fields[i, 1]
        B_z_3d[ix, iy, iz] = B_fields[i, 2]
    
    # Step 8: Create new 3D grid for interpolation output
    X_new, Y_new, Z_new = np.meshgrid(x_new, y_new, z_new, indexing='ij')
    points_new = np.column_stack([X_new.ravel(), Y_new.ravel(), Z_new.ravel()])
    
    # Step 9: Interpolate each B-field component (FIXED: pass points correctly)
    B_x_interp = scipy.interpolate.griddata(points_orig, B_x_3d.ravel(), points_new, method='linear')
    B_y_interp = scipy.interpolate.griddata(points_orig, B_y_3d.ravel(), points_new, method='linear')
    B_z_interp = scipy.interpolate.griddata(points_orig, B_z_3d.ravel(), points_new, method='linear')
    
    # Step 10: Flatten and create output arrays
    target_points_interp = points_new.copy()
    
    B_fields_interp = np.column_stack([
        B_x_interp.ravel(),
        B_y_interp.ravel(),
        B_z_interp.ravel()
    ])
    
    # Step 11: Remove NaN values (from extrapolation)
    valid_mask = np.all(np.isfinite(B_fields_interp), axis=1)
    target_points_interp = target_points_interp[valid_mask]
    B_fields_interp = B_fields_interp[valid_mask]
    
    print(f"Final grid: {len(target_points_interp)} points (after removing NaN)")
    
    return target_points_interp, B_fields_interp

def point_in_contour_2d(point, contour_points):
    """
    Check if a 2D point is inside a polygon contour using ray-casting algorithm.
    Also returns whether point is exactly on the contour edge.
    """
    x, y = point
    n = len(contour_points)
    inside = False
    
    # Convert to numpy array for easier indexing
    contour_points = np.array(contour_points)
    
    # Check if point is on contour edge
    on_contour = False
    for i in range(n):
        j = (i + 1) % n
        x1, y1 = contour_points[i]
        x2, y2 = contour_points[j]
        
        cross = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
        if abs(cross) < 1e-10:
            if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
                on_contour = True
                break
    
    if on_contour:
        return False, True
    
    # Ray-casting algorithm
    for i in range(n):
        j = (i + 1) % n
        x1, y1 = contour_points[i]
        x2, y2 = contour_points[j]
        
        if ((y1 > y) != (y2 > y)):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            
            if x < x_intersect:
                inside = not inside
    
    return inside, False


def create_coil_stream_function(wire_positions, contour_is_positive, current, target_vertices):
    """
    DEBUG VERSION - with prints to trace the execution
    """
    n_vertices = len(target_vertices)
    n_faces = 6
    
    # Initialize stream function array
    stream_values = np.zeros(n_vertices)
    
    print(f"\n{'='*60}")
    print(f"Starting create_coil_stream_function")
    print(f"target_vertices shape: {target_vertices.shape}")
    print(f"wire_positions has {len(wire_positions)} faces")
    print(f"contour_is_positive has {len(contour_is_positive)} faces")
    print(f"{'='*60}\n")
    
    # Track which vertices get non-zero values
    vertices_with_values = np.zeros(n_vertices, dtype=int)
    
    # Process each face independently
    for face_idx in range(n_faces):
        print(f"\n{'='*60}")
        print(f"PROCESSING FACE {face_idx + 1}")
        print(f"{'='*60}")
        
        face_contours = wire_positions[face_idx]        # List of all Contours on a face (number_of_contours, number_of_points_in_one_contour, 3)
        face_positive = contour_is_positive[face_idx]   # List of current sign of each Contour with length (number_of_contours)
        n_contours = len(face_contours)
        
        print(f"  Number of contours on this face: {n_contours}")
        
        if n_contours == 0:
            print(f"  >>> NO CONTOURS on this face, skipping")
            continue
        
        print(f"  Contour booleans (current sign): {face_positive}")
        
        # Determine which axis is normal to this face
        all_points_face = []
        for contour in face_contours:
            for pt in contour:
                all_points_face.append(pt)
        
        all_points_face = np.array(all_points_face)
        
        variance = np.var(all_points_face, axis=0)      # If all points are on the same plane, one dimension will always be the same -> very small variance
        normal_axis = np.argmin(variance)               # choose index with smallest variance
        
        const_coord = all_points_face[0, normal_axis]   # Determine constant offset of the plane by
        
        print(f"  Normal axis: {normal_axis} (x=0, y=1, z=2)")
        print(f"  Constant coordinate value: {const_coord:.4f} m")
        print(f"  Variance along axes: x={variance[0]:.1e}, y={variance[1]:.1e}, z={variance[2]:.1e}")
        
        # Extract 2D coordinates for this face
        contour_2d = []
        for contour in face_contours:
            contour_2d_points = []
            for pt in contour:
                if normal_axis == 0:
                    contour_2d_points.append([pt[1], pt[2]])
                elif normal_axis == 1:
                    contour_2d_points.append([pt[0], pt[2]])
                elif normal_axis == 2:
                    contour_2d_points.append([pt[0], pt[1]])
            contour_2d.append(contour_2d_points)
        
        print(f"  2D contours extracted: {len(contour_2d)} which all have length {len(contour_2d[0])}")
        
        # Extract 2D coordinates for all target vertices accourding to the plane (all points are extracted!)
        if normal_axis == 0:
            vertex_2d = target_vertices[:, [1, 2]]
        elif normal_axis == 1:
            vertex_2d = target_vertices[:, [0, 2]]
        elif normal_axis == 2:
            vertex_2d = target_vertices[:, [0, 1]]
        
        # Check which vertices are on this face (within tolerance)
        face_vertices_mask = np.abs(target_vertices[:, normal_axis] - const_coord) < 1e-6
        face_vertices_count = np.sum(face_vertices_mask)
        
        print(f"\n  Vertices on this face: {face_vertices_count} out of {n_vertices}")
        
        # For each vertex on this face, check which contours enclose it
        face_nonzero_count = 0
        
        for vert_idx in range(n_vertices):
            if not face_vertices_mask[vert_idx]:
                continue  # Skip vertices not on this face
            
            vertex_point = vertex_2d[vert_idx]
            
            positive_count = 0
            negative_count = 0
            on_contour = False
            
            for contour_idx in range(n_contours):
                contour_points = contour_2d[contour_idx]
                is_positive = face_positive[contour_idx]
                
                inside, on_edge = point_in_contour_2d(vertex_point, contour_points)
                
                if on_edge:
                    on_contour = True
                    break
                elif inside:
                    if is_positive:
                        positive_count += 1
                    else:
                        negative_count += 1
            
            # Assign analytical value
            if on_contour:
                if positive_count == 0 and negative_count == 0:
                    new_value = 0
                else:
                    if positive_count > 0:
                        inner_value = current * positive_count
                        outer_value = current * (positive_count - 1)
                    else:
                        inner_value = -current * negative_count
                        outer_value = -current * (negative_count - 1)
                    new_value = (inner_value + outer_value) / 2
                    
            elif not on_contour:
                if positive_count > 0 and negative_count == 0:
                    new_value = current * positive_count
                elif negative_count > 0 and positive_count == 0:
                    new_value = -current * negative_count
                elif positive_count > 0 and negative_count > 0:
                    new_value = current * (positive_count - negative_count)
                else:
                    new_value = 0
            
            stream_values[vert_idx] = new_value
            
            if new_value != 0:
                face_nonzero_count += 1
                vertices_with_values[vert_idx] += 1
        
        print(f"\n  Face {face_idx}: {face_nonzero_count} vertices got non-zero values")
        print(f"{'='*60}\n")    
    
    return stream_values, vertices_with_values