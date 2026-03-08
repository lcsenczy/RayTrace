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

#%% 2 - Calculate PRT Matrices for 2 and combine for 3



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

#%% 3 - Plot at PA, PB, and PC

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

# def plot_prt_pupil(grid, title_suffix, filename):
#     fig, axes = plt.subplots(3, 3, figsize=(12, 10))
#     plt.subplots_adjust(hspace=0.4, wspace=0.4)
#     for i in range(3):
#         for j in range(3):
#             ax = axes[i, j]
#             # Plot Magnitude log10
#             mag = np.abs(grid[i, j])
#             # Use small epsilon for log10 to avoid -inf
#             im = ax.imshow(np.log10(mag + 1e-15), cmap='viridis', extent=[-1, 1, -1, 1])
#             ax.set_title(f'P[{i+1},{j+1}] Mag')
#             fig.colorbar(im, ax=ax)
#     fig.suptitle(f'PRT Pupil Magnitude (log10) - {title_suffix}')
#     plt.savefig(filename + "_mag.png")
    
#     fig, axes = plt.subplots(3, 3, figsize=(12, 10))
#     plt.subplots_adjust(hspace=0.4, wspace=0.4)
#     for i in range(3):
#         for j in range(3):
#             ax = axes[i, j]
#             # Plot Phase -180 to 180
#             phase = np.angle(grid[i, j], deg=True)
#             im = ax.imshow(phase, cmap='hsv', vmin=-180, vmax=180, extent=[-1, 1, -1, 1])
#             ax.set_title(f'P[{i+1},{j+1}] Phase')
#             fig.colorbar(im, ax=ax)
#     fig.suptitle(f'PRT Pupil Phase (deg) - {title_suffix}')
#     plt.savefig(filename + "_phase.png")

# plot_prt_pupil(PA_grid, "Lens (PA)", "PA")
# plot_prt_pupil(PB_grid, "Lens + M1 (PB)", "PB")
# plot_prt_pupil(PC_grid, "Lens + M1 + M2 (PC)", "PC")

# # Perform the suggested test on one ray (Ray 0)
# # P_C * k1 should be k8
# k1_0 = np.array([df[(df['Ray_ID']==0) & (df['Surf_num']==1)]['Ray_L'].values[0],
#                 df[(df['Ray_ID']==0) & (df['Surf_num']==1)]['Ray_M'].values[0],
#                 df[(df['Ray_ID']==0) & (df['Surf_num']==1)]['Ray_N'].values[0]])
# k8_0 = np.array([df[(df['Ray_ID']==0) & (df['Surf_num']==8)]['Ray_L'].values[0],
#                 df[(df['Ray_ID']==0) & (df['Surf_num']==8)]['Ray_M'].values[0],
#                 df[(df['Ray_ID']==0) & (df['Surf_num']==8)]['Ray_N'].values[0]])

# pc_0 = PC_list[0]
# k8_calc = pc_0 @ k1_0
# print(f"Test Ray 0: Expected k8 = {k8_0}")
# print(f"Test Ray 0: Calculated k8 = {k8_calc.real}")
# print(f"Difference: {np.linalg.norm(k8_0 - k8_calc.real)}")


def plot_prt_pupil(grid, title_suffix, filename):
    # Create a 3x6 grid: 3 rows, 6 columns (left 3 for mag, right 3 for phase)
    fig, axes = plt.subplots(3, 6, figsize=(24, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    
    for i in range(3):
        for j in range(3):
            # ---------------------------
            # 1. Plot Magnitude (Columns 0, 1, 2)
            # ---------------------------
            ax_mag = axes[i, j]
            mag = np.abs(grid[i, j])
            # Use small epsilon for log10 to avoid -inf
            im_mag = ax_mag.imshow(np.log10(mag + 1e-15), cmap='viridis', extent=[-1, 1, -1, 1])
            ax_mag.set_title(f'P[{i+1},{j+1}] Mag')
            fig.colorbar(im_mag, ax=ax_mag)
            
            # ---------------------------
            # 2. Plot Phase (Columns 3, 4, 5)
            # ---------------------------
            ax_phase = axes[i, j + 3]
            phase = np.angle(grid[i, j], deg=True)
            im_phase = ax_phase.imshow(phase, cmap='hsv', vmin=-180, vmax=180, extent=[-1, 1, -1, 1])
            ax_phase.set_title(f'P[{i+1},{j+1}] Phase')
            fig.colorbar(im_phase, ax=ax_phase)
            
    fig.suptitle(f'PRT Pupil Magnitude (log10) and Phase (deg) - {title_suffix}', fontsize=16)
    # plt.savefig(filename + "_combined.png", bbox_inches='tight')
    # plt.close(fig) # Close the figure to free up memory

# Example Usage:
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


#%% 4 Calc DP for the input rays

# 1. Identify Chief Ray (rho_X=0, rho_Y=0)
chief_ray_row = df[(df['rho_X'] == 0) & (df['rho_Y'] == 0) & (df['Surf_num'] == 1)]
if chief_ray_row.empty:
    # Fallback to Ray_ID 40
    chief_ray_row = df[(df['Ray_ID'] == 40) & (df['Surf_num'] == 1)]

A = np.array([chief_ray_row['Ray_L'].values[0], 
              chief_ray_row['Ray_M'].values[0], 
              chief_ray_row['Ray_N'].values[0]])
A = A / np.linalg.norm(A)

print(f"Chief ray direction (Antipole A): {A}")

# 2. Define reference basis at A
x0 = np.array([1, 0, 0], dtype=float)
# y0 should be perpendicular to A and x0
y0 = np.cross(A, x0)
y0 = y0 / np.linalg.norm(y0)

# Check orthogonality
print(f"x0 dot A: {np.dot(x0, A)}")
print(f"y0 dot A: {np.dot(y0, A)}")
print(f"x0 dot y0: {np.dot(x0, y0)}")

# 3. Calculate Double Pole Basis for all 81 rays
ray_ids = sorted(df['Ray_ID'].unique())
basis_matrices = []

for rid in ray_ids:
    row = df[(df['Ray_ID'] == rid) & (df['Surf_num'] == 1)]
    k = np.array([row['Ray_L'].values[0], row['Ray_M'].values[0], row['Ray_N'].values[0]])
    k = k / np.linalg.norm(k)
    
    # Double pole basis vectors
    denom = 1 + np.dot(k, A)
    xk = x0 - (np.dot(k, x0) * (k + A)) / denom
    yk = y0 - (np.dot(k, y0) * (k + A)) / denom
    
    # Basis matrix (columns are x, y, k)
    D = np.column_stack((xk, yk, k))
    basis_matrices.append(D)

# 4. Reshape for plotting
grid_D = np.zeros((3, 3, 9, 9))
for idx, D in enumerate(basis_matrices):
    r, c = idx // 9, idx % 9
    for i in range(3):
        for j in range(3):
            grid_D[i, j, r, c] = D[i, j]

# Visualize the 9 elements
fig, axes = plt.subplots(3, 3, figsize=(10, 8))
for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        im = ax.imshow(grid_D[i, j], extent=[-1, 1, -1, 1], cmap='coolwarm')
        ax.set_title(f'D[{i+1},{j+1}]')
        fig.colorbar(im, ax=ax)

fig.suptitle("Double Pole Basis Matrix Elements (Input Rays)")
plt.tight_layout()
# plt.savefig("double_pole_basis.png")

# Verify orthnormality of one example (Ray 0)
D0 = basis_matrices[0]
print(f"\nRay 0 Basis Matrix:\n{D0}")
print(f"D0^T * D0 (Should be identity):\n{D0.T @ D0}")

#%% 5 - Double pole for surfs 3, 5, 8

def get_double_pole_basis(df_surface, x0_vec):
    # Find chief ray (A) for this surface
    chief_ray = df_surface[(df_surface['rho_X'] == 0) & (df_surface['rho_Y'] == 0)]
    if chief_ray.empty:
        chief_ray = df_surface[df_surface['Ray_ID'] == 40]
        
    A = np.array([chief_ray['Ray_L'].values[0], chief_ray['Ray_M'].values[0], chief_ray['Ray_N'].values[0]])
    A = A / np.linalg.norm(A)
    
    x0 = np.array(x0_vec, dtype=float)
    x0 = x0 / np.linalg.norm(x0)
    
    # Orthogonalize if needed, though usually x0 is chosen perp to A, or we just cross it
    # But double pole parallel transport expects A dot x0 = 0? 
    # Actually, standard double pole definition:
    # If x0 is not perfectly perpendicular to A, we project it? 
    # Usually y0 = A x x0. Then we redefine x0 = y0 x A to ensure perfect orthogonality.
    y0 = np.cross(A, x0)
    y0 = y0 / np.linalg.norm(y0)
    x0 = np.cross(y0, A)
    x0 = x0 / np.linalg.norm(x0)
    
    ray_ids = sorted(df_surface['Ray_ID'].unique())
    basis_matrices = []
    
    for rid in ray_ids:
        row = df_surface[df_surface['Ray_ID'] == rid]
        k = np.array([row['Ray_L'].values[0], row['Ray_M'].values[0], row['Ray_N'].values[0]])
        k = k / np.linalg.norm(k)
        
        denom = 1 + np.dot(k, A)
        if denom < 1e-12:
            xk = x0
            yk = y0
        else:
            xk = x0 - (np.dot(k, x0) * (k + A)) / denom
            yk = y0 - (np.dot(k, y0) * (k + A)) / denom
            
        D = np.column_stack((xk, yk, k))
        basis_matrices.append(D)
        
    return basis_matrices, A, x0, y0

def plot_basis(matrices, title, filename):
    grid = np.zeros((3, 3, 9, 9))
    for idx, D in enumerate(matrices):
        r, c = idx // 9, idx % 9
        for i in range(3):
            for j in range(3):
                grid[i, j, r, c] = D[i, j]
                
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            im = ax.imshow(grid[i, j], extent=[-1, 1, -1, 1], cmap='coolwarm')
            ax.set_title(f'D[{i+1},{j+1}]')
            fig.colorbar(im, ax=ax)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)

# 1. Rays leaving Surface 3 ( incident on 4 )
# Surf_num == 4
df_s4 = df[df['Surf_num'] == 4]
D_out3, A3, x0_3, y0_3 = get_double_pole_basis(df_s4, [1, 0, 0])
plot_basis(D_out3, "Double Pole Basis - Output of Surf 3", "D_out3.png")

# 2. Rays leaving Surface 5 ( incident on 6 )
# Surf_num == 6
df_s6 = df[df['Surf_num'] == 6]
D_out5, A5, x0_5, y0_5 = get_double_pole_basis(df_s6, [1, 0, 0])
plot_basis(D_out5, "Double Pole Basis - Output of Surf 5", "D_out5.png")

# 3. Rays leaving Surface 8 ( incident on 9 )
# Surf_num == 9
df_s9 = df[df['Surf_num'] == 9]
D_out8, A8, x0_8, y0_8 = get_double_pole_basis(df_s9, [0, 1, 0])
plot_basis(D_out8, "Double Pole Basis - Output of Surf 8", "D_out8.png")

print(f"Surf 3 (incident on 4) Antipole A: {A3}")
print(f"Surf 5 (incident on 6) Antipole A: {A5}")
print(f"Surf 8 (incident on 9) Antipole A: {A8}")

#%% 6- PRT to jones

# Helper to get PRT matrix
def get_prt_matrix(k_in, k_out, eta, n1, n2, is_reflection=False, n_metal=None):
    k_in = k_in / np.linalg.norm(k_in)
    k_out = k_out / np.linalg.norm(k_out)
    eta = eta / np.linalg.norm(eta)
    
    s = np.cross(k_in, eta)
    s_norm = np.linalg.norm(s)
    
    if s_norm < 1e-12:
        if abs(k_in[0]) < 0.9: s = np.array([1, 0, 0])
        else: s = np.array([0, 1, 0])
        s = np.cross(k_in, s)
        s = s / np.linalg.norm(s)
    else:
        s = s / s_norm
        
    p_in = np.cross(k_in, s)
    p_out = np.cross(k_out, s)
    
    cos_theta1 = np.abs(np.dot(k_in, eta))
    sin_theta1 = np.sqrt(max(0, 1 - cos_theta1**2) + 0j)
    
    if is_reflection:
        sin_theta2 = (n1 / n_metal) * sin_theta1
        cos_theta2 = np.sqrt(1 - sin_theta2**2 + 0j)
        rs = (n1 * cos_theta1 - n_metal * cos_theta2) / (n1 * cos_theta1 + n_metal * cos_theta2)
        rp = (n_metal * cos_theta1 - n1 * cos_theta2) / (n_metal * cos_theta1 + n1 * cos_theta2)
    else:
        sin_theta2 = (n1 / n2) * sin_theta1
        cos_theta2 = np.sqrt(1 - sin_theta2**2 + 0j)
        rs = (n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2)
        rp = (n2 * cos_theta1 - n1 * cos_theta2) / (n2 * cos_theta1 + n1 * cos_theta2)

    return rs * np.outer(s, s) + rp * np.outer(p_out, p_in) + np.outer(k_out, k_in)

# Double pole basis helper
def get_basis(k_vec, A, x0):
    k = k_vec / np.linalg.norm(k_vec)
    A = A / np.linalg.norm(A)
    x0 = x0 / np.linalg.norm(x0)
    
    y0 = np.cross(A, x0)
    y0 = y0 / np.linalg.norm(y0)
    x0 = np.cross(y0, A)
    x0 = x0 / np.linalg.norm(x0)
    
    denom = 1 + np.dot(k, A)
    if denom < 1e-12:
        xk, yk = x0, y0
    else:
        xk = x0 - (np.dot(k, x0) * (k + A)) / denom
        yk = y0 - (np.dot(k, y0) * (k + A)) / denom
        
    return np.column_stack((xk, yk, k))

n_Al = 1.0152 + 6.6273j
ray_ids = sorted(df['Ray_ID'].unique())

JA_list, JB_list, JC_list = [], [], []

# Antipoles and references
A_in = np.array([0, 0, 1]); x0_in = np.array([1, 0, 0])
A_outA = np.array([0, 0, 1]); x0_outA = np.array([1, 0, 0])
A_outB = np.array([0, 1, 0]); x0_outB = np.array([1, 0, 0])
A_outC = np.array([1, 0, 0]); x0_outC = np.array([0, 1, 0])

for rid in ray_ids:
    ray_data = df[df['Ray_ID'] == rid].set_index('Surf_num')
    
    def row_k(s): return np.array([ray_data.loc[s, 'Ray_L'], ray_data.loc[s, 'Ray_M'], ray_data.loc[s, 'Ray_N']])
    def row_eta(s): return np.array([ray_data.loc[s, 'Surf_L'], ray_data.loc[s, 'Surf_M'], ray_data.loc[s, 'Surf_N']])
    
    # Bases
    Din = get_basis(row_k(1), A_in, x0_in)
    DoutA = get_basis(row_k(4), A_outA, x0_outA)
    DoutB = get_basis(row_k(6), A_outB, x0_outB)
    DoutC = get_basis(row_k(9), A_outC, x0_outC)
    
    # PRT matrices
    p2 = get_prt_matrix(row_k(2), row_k(3), row_eta(2), ray_data.loc[2, 'Index_in'], ray_data.loc[2, 'Index_out'])
    p3 = get_prt_matrix(row_k(3), row_k(4), row_eta(3), ray_data.loc[3, 'Index_in'], ray_data.loc[3, 'Index_out'])
    p5 = get_prt_matrix(row_k(5), row_k(6), row_eta(5), ray_data.loc[5, 'Index_in'], ray_data.loc[5, 'Index_out'], True, n_Al)
    p8 = get_prt_matrix(row_k(8), row_k(9), row_eta(8), ray_data.loc[8, 'Index_in'], ray_data.loc[8, 'Index_out'], True, n_Al)
    
    pa = p3 @ p2
    pb = p5 @ pa
    pc = p8 @ pb
    
    # Jones matrices (2x2 top-left of Dout^T * P * Din)
    ja = (DoutA.T @ pa @ Din)[:2, :2]
    jb = (DoutB.T @ pb @ Din)[:2, :2]
    jc = (DoutC.T @ pc @ Din)[:2, :2]
    
    JA_list.append(ja)
    JB_list.append(jb)
    JC_list.append(jc)

def prep_jones(mlist):
    grid = np.zeros((2, 2, 9, 9), dtype=complex)
    for idx, m in enumerate(mlist):
        r, c = idx // 9, idx % 9
        grid[:,:,r,c] = m
    return grid

JA_g = prep_jones(JA_list)
JB_g = prep_jones(JB_list)
JC_g = prep_jones(JC_list)

# def plot_jones(grid, title, fname_prefix):
#     fig, axes = plt.subplots(2, 2, figsize=(8, 6))
#     for i in range(2):
#         for j in range(2):
#             ax = axes[i,j]
#             im = ax.imshow(np.abs(grid[i,j]), extent=[-1,1,-1,1], cmap='viridis')
#             ax.set_title(f'J[{i+1},{j+1}] Mag')
#             fig.colorbar(im, ax=ax)
#     fig.suptitle(f"Magnitude - {title}")
#     plt.tight_layout()
#     plt.savefig(f"{fname_prefix}_mag.png")
    
#     fig, axes = plt.subplots(2, 2, figsize=(8, 6))
#     for i in range(2):
#         for j in range(2):
#             ax = axes[i,j]
#             im = ax.imshow(np.angle(grid[i,j], deg=True), extent=[-1,1,-1,1], cmap='hsv', vmin=-180, vmax=180)
#             ax.set_title(f'J[{i+1},{j+1}] Phase')
#             fig.colorbar(im, ax=ax)
#     fig.suptitle(f"Phase (deg) - {title}")
#     plt.tight_layout()
#     plt.savefig(f"{fname_prefix}_phase.png")

# plot_jones(JA_g, "JA (Lens)", "JA")
# plot_jones(JB_g, "JB (Lens+M1)", "JB")
# plot_jones(JC_g, "JC (Lens+M1+M2)", "JC")

# print("Jones pupil generation complete.")

# def plot_jones(grid, title, fname_prefix):
#     # Create a 2x4 grid: 2 rows, 4 columns (left 2 for mag, right 2 for phase)
#     fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    
#     for i in range(2):
#         for j in range(2):
#             # ---------------------------
#             # 1. Plot Magnitude (Columns 0, 1)
#             # ---------------------------
#             ax_mag = axes[i, j]
#             im_mag = ax_mag.imshow(np.abs(grid[i, j]), extent=[-1, 1, -1, 1], cmap='viridis')
#             ax_mag.set_title(f'J[{i+1},{j+1}] Mag')
#             fig.colorbar(im_mag, ax=ax_mag)
            
#             # ---------------------------
#             # 2. Plot Phase (Columns 2, 3)
#             # ---------------------------
#             ax_phase = axes[i, j + 2]
#             im_phase = ax_phase.imshow(np.angle(grid[i, j], deg=True), extent=[-1, 1, -1, 1], cmap='hsv', vmin=-180, vmax=180)
#             ax_phase.set_title(f'J[{i+1},{j+1}] Phase')
#             fig.colorbar(im_phase, ax=ax_phase)
            
#     fig.suptitle(f"Jones Pupil Magnitude and Phase - {title}", fontsize=16)
#     plt.tight_layout()
#     # plt.savefig(f"{fname_prefix}_combined.png", bbox_inches='tight')
#     # plt.close(fig)  # Close the figure to free up memory

# # Example Usage:
# plot_jones(JA_g, "JA (Lens)", "JA")
# plot_jones(JB_g, "JB (Lens+M1)", "JB")
# plot_jones(JC_g, "JC (Lens+M1+M2)", "JC")

# print("Jones pupil generation complete.")

def plot_jones(grid, title, fname_prefix):
    # Create a 2x4 grid: 2 rows, 4 columns (left 2 for mag, right 2 for phase)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    
    for i in range(2):
        for j in range(2):
            # ---------------------------
            # 1. Plot Magnitude (Columns 0, 1) in log10
            # ---------------------------
            ax_mag = axes[i, j]
            # Calculate magnitude and apply log10 (with a small epsilon to prevent -inf)
            mag = np.abs(grid[i, j])
            im_mag = ax_mag.imshow(np.log10(mag + 1e-15), extent=[-1, 1, -1, 1], cmap='viridis')
            ax_mag.set_title(f'J[{i+1},{j+1}] Mag (log10)')
            fig.colorbar(im_mag, ax=ax_mag)
            
            # ---------------------------
            # 2. Plot Phase (Columns 2, 3)
            # ---------------------------
            ax_phase = axes[i, j + 2]
            im_phase = ax_phase.imshow(np.angle(grid[i, j], deg=True), extent=[-1, 1, -1, 1], cmap='hsv', vmin=-180, vmax=180)
            ax_phase.set_title(f'J[{i+1},{j+1}] Phase')
            fig.colorbar(im_phase, ax=ax_phase)
            
    fig.suptitle(f"Jones Pupil Magnitude (log10) and Phase - {title}", fontsize=16)
    plt.tight_layout()
    # plt.savefig(f"{fname_prefix}_combined.png", bbox_inches='tight')
    # plt.close(fig)  # Close the figure to free up memory

# Example Usage:
plot_jones(JA_g, "JA (Lens)", "JA")
plot_jones(JB_g, "JB (Lens+M1)", "JB")
plot_jones(JC_g, "JC (Lens+M1+M2)", "JC")

print("Jones pupil generation complete.")

#%% 7 - Diattenuation

def get_prt_matrix(k_in, k_out, eta, n1, n2, is_reflection=False, n_metal=None):
    k_in = k_in / np.linalg.norm(k_in)
    k_out = k_out / np.linalg.norm(k_out)
    eta = eta / np.linalg.norm(eta)
    s = np.cross(k_in, eta)
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-12:
        if abs(k_in[0]) < 0.9: s = np.array([1, 0, 0])
        else: s = np.array([0, 1, 0])
        s = np.cross(k_in, s)
        s = s / np.linalg.norm(s)
    else:
        s = s / s_norm
        
    p_in = np.cross(k_in, s)
    p_out = np.cross(k_out, s)
    cos_theta1 = np.abs(np.dot(k_in, eta))
    sin_theta1 = np.sqrt(max(0, 1 - cos_theta1**2) + 0j)
    
    if is_reflection:
        sin_theta2 = (n1 / n_metal) * sin_theta1
        cos_theta2 = np.sqrt(1 - sin_theta2**2 + 0j)
        rs = (n1 * cos_theta1 - n_metal * cos_theta2) / (n1 * cos_theta1 + n_metal * cos_theta2)
        rp = (n_metal * cos_theta1 - n1 * cos_theta2) / (n_metal * cos_theta1 + n1 * cos_theta2)
    else:
        sin_theta2 = (n1 / n2) * sin_theta1
        cos_theta2 = np.sqrt(1 - sin_theta2**2 + 0j)
        rs = (n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2)
        rp = (n2 * cos_theta1 - n1 * cos_theta2) / (n2 * cos_theta1 + n1 * cos_theta2)

    return rs * np.outer(s, s) + rp * np.outer(p_out, p_in) + np.outer(k_out, k_in)

def get_basis(k_vec, A, x0):
    k = k_vec / np.linalg.norm(k_vec)
    A = A / np.linalg.norm(A)
    x0 = x0 / np.linalg.norm(x0)
    y0 = np.cross(A, x0)
    y0 = y0 / np.linalg.norm(y0)
    x0 = np.cross(y0, A)
    x0 = x0 / np.linalg.norm(x0)
    denom = 1 + np.dot(k, A)
    if denom < 1e-12:
        xk, yk = x0, y0
    else:
        xk = x0 - (np.dot(k, x0) * (k + A)) / denom
        yk = y0 - (np.dot(k, y0) * (k + A)) / denom
    return np.column_stack((xk, yk, k))

n_Al = 1.0152 + 6.6273j
ray_ids = sorted(df['Ray_ID'].unique())

A_in = np.array([0, 0, 1]); x0_in = np.array([1, 0, 0])
A_outA = np.array([0, 0, 1]); x0_outA = np.array([1, 0, 0])
A_outB = np.array([0, 1, 0]); x0_outB = np.array([1, 0, 0])
A_outC = np.array([1, 0, 0]); x0_outC = np.array([0, 1, 0])

diat_A = np.zeros((9, 9))
diat_B = np.zeros((9, 9))
diat_C = np.zeros((9, 9))

def compute_diattenuation(J):
    # D = (Tmax - Tmin) / (Tmax + Tmin)
    # Tmax, Tmin are squared singular values of J
    U, S, Vh = np.linalg.svd(J)
    Tmax = np.abs(S[0])**2
    Tmin = np.abs(S[1])**2
    if Tmax + Tmin == 0:
        return 0
    return (Tmax - Tmin) / (Tmax + Tmin)

for idx, rid in enumerate(ray_ids):
    ray_data = df[df['Ray_ID'] == rid].set_index('Surf_num')
    def row_k(s): return np.array([ray_data.loc[s, 'Ray_L'], ray_data.loc[s, 'Ray_M'], ray_data.loc[s, 'Ray_N']])
    def row_eta(s): return np.array([ray_data.loc[s, 'Surf_L'], ray_data.loc[s, 'Surf_M'], ray_data.loc[s, 'Surf_N']])
    
    Din = get_basis(row_k(1), A_in, x0_in)
    DoutA = get_basis(row_k(4), A_outA, x0_outA)
    DoutB = get_basis(row_k(6), A_outB, x0_outB)
    DoutC = get_basis(row_k(9), A_outC, x0_outC)
    
    p2 = get_prt_matrix(row_k(2), row_k(3), row_eta(2), ray_data.loc[2, 'Index_in'], ray_data.loc[2, 'Index_out'])
    p3 = get_prt_matrix(row_k(3), row_k(4), row_eta(3), ray_data.loc[3, 'Index_in'], ray_data.loc[3, 'Index_out'])
    p5 = get_prt_matrix(row_k(5), row_k(6), row_eta(5), ray_data.loc[5, 'Index_in'], ray_data.loc[5, 'Index_out'], True, n_Al)
    p8 = get_prt_matrix(row_k(8), row_k(9), row_eta(8), ray_data.loc[8, 'Index_in'], ray_data.loc[8, 'Index_out'], True, n_Al)
    
    pa = p3 @ p2
    pb = p5 @ pa
    pc = p8 @ pb
    
    ja = (DoutA.T @ pa @ Din)[:2, :2]
    jb = (DoutB.T @ pb @ Din)[:2, :2]
    jc = (DoutC.T @ pc @ Din)[:2, :2]
    
    r, c = idx // 9, idx % 9
    diat_A[r, c] = compute_diattenuation(ja)
    diat_B[r, c] = compute_diattenuation(jb)
    diat_C[r, c] = compute_diattenuation(jc)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

imA = axes[0].imshow(diat_A, cmap='plasma', extent=[-1, 1, -1, 1])
axes[0].set_title('Diattenuation - JA (Lens)')
fig.colorbar(imA, ax=axes[0])

imB = axes[1].imshow(diat_B, cmap='plasma', extent=[-1, 1, -1, 1])
axes[1].set_title('Diattenuation - JB (Lens + M1)')
fig.colorbar(imB, ax=axes[1])

imC = axes[2].imshow(diat_C, cmap='plasma', extent=[-1, 1, -1, 1])
axes[2].set_title('Diattenuation - JC (Lens + M1 + M2)')
fig.colorbar(imC, ax=axes[2])

plt.tight_layout()

print("Diattenuation calculation complete. Saved to diattenuation_pupils.png.")