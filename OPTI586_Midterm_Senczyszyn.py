#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 18:35:26 2026

@author: lukebear

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% 1 - Import Data

# Load the CSV file into a DataFrame
df = pd.read_csv('rayData_midterm.csv')

# Display basic information about the dataset
print(df.info())

# Display the first few rows
print(df.head())


#Mult surf normals by -1 to match textbook notation

# Invert the normals (Surf_L, Surf_M, Surf_N)
df['Surf_L'] = -df['Surf_L']
df['Surf_M'] = -df['Surf_M']
df['Surf_N'] = -df['Surf_N']

# # Save the modified data to a new CSV
# inverted_filename = 'rayData_inverted_normals.csv'
# df.to_csv(inverted_filename, index=False)

# Display the head to verify
print(df.head())
# print(f"\nSaved inverted normals to {inverted_filename}")

#%% 2 - Calculate PRT Matrices

def get_prt_matrix(k_in, k_out, eta, n1, n2, is_reflection=False, n_metal=None):
    """
    Calculates the 3x3 PRT matrix for a single surface interaction.
    """
    # Normalize vectors just in case
    k_in = k_in / np.linalg.norm(k_in)
    k_out = k_out / np.linalg.norm(k_out)
    eta = eta / np.linalg.norm(eta)
    
    # Calculate s-vector (orthonormal to k_in and eta)
    s = np.cross(k_in, eta)
    s_norm = np.linalg.norm(s)
    
    if s_norm < 1e-10:
        # Normal incidence case: pick an arbitrary s perpendicular to k_in
        if abs(k_in[0]) < 0.9:
            s = np.array([1, 0, 0])
        else:
            s = np.array([0, 1, 0])
        s = np.cross(k_in, s)
        s = s / np.linalg.norm(s)
    else:
        s = s / s_norm
        
    # Define p_in and p_out
    p_in = np.cross(k_in, s)
    p_out = np.cross(k_out, s)
    
    # Cosines for Fresnel
    cos_theta1 = np.abs(np.dot(k_in, eta))
    
    if is_reflection:
        # Reflection at a metal surface
        # n1 is incident medium, n_metal is the mirror material
        n2_eff = n_metal
        sin_theta1 = np.sqrt(1 - cos_theta1**2 + 0j)
        sin_theta2 = (n1 / n2_eff) * sin_theta1
        cos_theta2 = np.sqrt(1 - sin_theta2**2 + 0j)
        
        rs = (n1 * cos_theta1 - n2_eff * cos_theta2) / (n1 * cos_theta1 + n2_eff * cos_theta2)
        rp = (n2_eff * cos_theta1 - n1 * cos_theta2) / (n2_eff * cos_theta1 + n1 * cos_theta2)
    else:
        # Refraction
        # n1 is incident, n2 is exiting
        sin_theta1 = np.sqrt(1 - cos_theta1**2 + 0j)
        sin_theta2 = (n1 / n2) * sin_theta1
        cos_theta2 = np.sqrt(1 - sin_theta2**2 + 0j)
        
        rs = (n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2)
        rp = (n2 * cos_theta1 - n1 * cos_theta2) / (n2 * cos_theta1 + n1 * cos_theta2)

    # Construct PRT Matrix: P = rs*(s*sT) + rp*(p_out*p_inT) + (k_out*k_inT)
    P = rs * np.outer(s, s) + rp * np.outer(p_out, p_in) + np.outer(k_out, k_in)
    return P

# Aluminum index at 550nm
n_Al = 1.0152 + 6.6273j

# Initialize lists to store matrices for each ray
P2_list, P3_list, P5_list, P8_list = [], [], [], []
PA_list, PB_list, PC_list = [], [], []

ray_ids = sorted(df['Ray_ID'].unique()) # sort to ensure correct rays 

for rid in ray_ids:
    ray_data = df[df['Ray_ID'] == rid].sort_values('Surf_num')
    
    # Helper to get k and eta
    def get_k(surf_num):
        row = ray_data[ray_data['Surf_num'] == surf_num]
        return np.array([row['Ray_L'].values[0], row['Ray_M'].values[0], row['Ray_N'].values[0]])
    
    def get_params(surf_num):
        row = ray_data[ray_data['Surf_num'] == surf_num]
        eta = np.array([row['Surf_L'].values[0], row['Surf_M'].values[0], row['Surf_N'].values[0]])
        n_in = row['Index_in'].values[0]
        n_out = row['Index_out'].values[0]
        return eta, n_in, n_out

    # P2: Refraction at surf 2 (k_in is from surf 1)
    k1 = get_k(1)
    k2 = get_k(2)
    eta2, n_in2, n_out2 = get_params(2)
    p2 = get_prt_matrix(k1, k2, eta2, n_in2, n_out2)
    P2_list.append(p2)
    
    # P3: Refraction at surf 3 (k_in is from surf 2)
    k3 = get_k(3)
    eta3, n_in3, n_out3 = get_params(3)
    p3 = get_prt_matrix(k2, k3, eta3, n_in3, n_out3)
    P3_list.append(p3)
    
    # P5: Reflection at surf 5 (k_in is from surf 4)
    k4 = get_k(4)
    k5 = get_k(5)
    eta5, n_in5, n_out5 = get_params(5)
    # Ref index of Al for reflection
    p5 = get_prt_matrix(k4, k5, eta5, n_in5, n_out5, is_reflection=True, n_metal=n_Al)
    P5_list.append(p5)
    
    # P8: Reflection at surf 8 (k_in is from surf 7)
    k7 = get_k(7)
    k8 = get_k(8)
    eta8, n_in8, n_out8 = get_params(8)
    p8 = get_prt_matrix(k7, k8, eta8, n_in8, n_out8, is_reflection=True, n_metal=n_Al)
    P8_list.append(p8)
    
    # Combine matrices
    pa = p3 @ p2
    pb = p5 @ pa
    pc = p8 @ pb
    
    PA_list.append(pa)
    PB_list.append(pb)
    PC_list.append(pc)

# Reshape into 9x9 grids for plotting (total 81 rays)
def prepare_grid(matrix_list):
    # matrix_list is length 81, each element is 3x3
    # Return a (3, 3, 9, 9) array
    grid = np.zeros((3, 3, 9, 9), dtype=complex)
    for idx, mat in enumerate(matrix_list):
        r = idx // 9
        c = idx % 9
        for i in range(3):
            for j in range(3):
                grid[i, j, r, c] = mat[i, j]
    return grid

PA_grid = prepare_grid(PA_list)
PB_grid = prepare_grid(PB_list)
PC_grid = prepare_grid(PC_list)

def plot_prt_pupil(grid, title_suffix, filename):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            # Plot Magnitude log10
            mag = np.abs(grid[i, j])
            # Use small epsilon for log10 to avoid -inf
            im = ax.imshow(np.log10(mag + 1e-15), cmap='viridis', extent=[-1, 1, -1, 1])
            ax.set_title(f'P[{i+1},{j+1}] Mag')
            fig.colorbar(im, ax=ax)
    fig.suptitle(f'PRT Pupil Magnitude (log10) - {title_suffix}')
    plt.savefig(filename + "_mag.png")
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            # Plot Phase -180 to 180
            phase = np.angle(grid[i, j], deg=True)
            im = ax.imshow(phase, cmap='hsv', vmin=-180, vmax=180, extent=[-1, 1, -1, 1])
            ax.set_title(f'P[{i+1},{j+1}] Phase')
            fig.colorbar(im, ax=ax)
    fig.suptitle(f'PRT Pupil Phase (deg) - {title_suffix}')
    plt.savefig(filename + "_phase.png")

plot_prt_pupil(PA_grid, "Lens (PA)", "PA")
plot_prt_pupil(PB_grid, "Lens + M1 (PB)", "PB")
plot_prt_pupil(PC_grid, "Lens + M1 + M2 (PC)", "PC")

# Perform the suggested test on one ray (Ray 0)
# P_C * k1 should be k8
k1_0 = np.array([df[(df['Ray_ID']==0) & (df['Surf_num']==1)]['Ray_L'].values[0],
                df[(df['Ray_ID']==0) & (df['Surf_num']==1)]['Ray_M'].values[0],
                df[(df['Ray_ID']==0) & (df['Surf_num']==1)]['Ray_N'].values[0]])
k8_0 = np.array([df[(df['Ray_ID']==0) & (df['Surf_num']==8)]['Ray_L'].values[0],
                df[(df['Ray_ID']==0) & (df['Surf_num']==8)]['Ray_M'].values[0],
                df[(df['Ray_ID']==0) & (df['Surf_num']==8)]['Ray_N'].values[0]])

pc_0 = PC_list[0]
k8_calc = pc_0 @ k1_0
print(f"Test Ray 0: Expected k8 = {k8_0}")
print(f"Test Ray 0: Calculated k8 = {k8_calc.real}")
print(f"Difference: {np.linalg.norm(k8_0 - k8_calc.real)}")