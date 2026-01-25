import numpy as np

# Example stiffness and damping ratio parameters
k_values = np.array([1000, 1500, 2000])  # Stiffness coefficients
xi_values = np.array([0.7, 0.6, 0.5])   # Damping ratios

# Create diagonal matrices for stiffness and damping ratios
k_d = np.diag(k_values)
xi_d = np.diag(xi_values)

# Calculating the damping coefficients
# Step 1: Calculate the square root of the diagonal elements of k_d
sqrt_k_d = np.sqrt(k_d)

# Step 2: Calculate damping coefficients, d_d = 2 * xi * sqrt(k)
d_values = 2 * np.diag(xi_d) * sqrt_k_d

# Create the diagonal matrix for damping coefficients
d_d = np.diag(d_values)

print("k_d matrix:\n", k_d)
print("xi_d matrix:\n", xi_d)
print("d_d matrix:\n", d_d)
