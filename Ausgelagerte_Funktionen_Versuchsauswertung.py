import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.interpolate

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

    fig_3d = plt.figure(figsize=(12, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')  # Fix: fig_3d statt fig

    scatter_3d = ax_3d.scatter(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        c=stream_function, cmap='RdBu_r', s=40, alpha=0.7
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
        vmax = v_max
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
        vmax = v_max
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
        vmax = v_max
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
        vmax = v_max
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
        vmax = v_max
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
        vmax = v_max
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