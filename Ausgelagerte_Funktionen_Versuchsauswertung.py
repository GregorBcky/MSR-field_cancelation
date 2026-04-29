import numpy as np
import os

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
    
            mean_Bx = data['mean_Bx_pT'] * 1e-12                                # Umrechnen in [T]
            mean_By = data['mean_By_pT'] * 1e-12
            mean_Bz = data['mean_Bz_pT'] * 1e-12

            each_target_point = np.array([x, y, z]).T                           # (Npoints, 3)
            each_B_target = np.array([mean_Bx, mean_By, mean_Bz]).T             # (Npoints, 3)
            
            target_point_coordinate_test.append(each_target_point)
            B_target_point_with_current.append(each_B_target)

    # Stack all files into final (total_Npoints, 3) arrays
    target_point_coordinate_test = np.vstack(target_point_coordinate_test)       # (total_Npoints, 3)
    B_target_point_with_current = np.vstack(B_target_point_with_current)             # (total_Npoints, 3)

    if not np.array_equal(target_point_coord, target_point_coordinate_test):
        raise ValueError('The coordinates of the target points with and without current do not match. Please check the data files.')

    # Debugging
    print(f'Target point coordinates and B-field at target points both need to have dimension (n_points, 3).\n  Shape of target coordinates: {target_point_coord.shape}\n  Shape of B-field with current at target points: {B_target_point_with_current.shape}\n  Shape of B-field without current at target points: {B_target_point_no_current.shape}')

    return target_point_coord, B_target_point_no_current, B_target_point_with_current