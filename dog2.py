import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Parameters
# ============================================================

# Number of Gaussian centers in each direction
n_basis_x = 10
n_basis_y = 10

# Number of spatial plotting / integration points
# These can be much larger than the number of Gaussians
n_grid_x = 100
n_grid_y = 100

# Spatial domain
x_min = 0.0
x_max = 9.0

y_min = 0.0
y_max = 9.0

# Gaussian width
w = 1.0


# ============================================================
# Gaussian centers
# ============================================================

center_x_1d = np.linspace(x_min, x_max, n_basis_x)
center_y_1d = np.linspace(y_min, y_max, n_basis_y)

CENTER_X, CENTER_Y = np.meshgrid(
    center_x_1d,
    center_y_1d,
    indexing="xy",
)

# Flatten the Gaussian-center coordinates
center_x = CENTER_X.ravel()
center_y = CENTER_Y.ravel()

n_basis = len(center_x)

print("Number of Gaussian basis functions:", n_basis)


# ============================================================
# X and Y operators in Gaussian-center space
# ============================================================
#
# These are exactly analogous to the original 1D
#
#     X = diag(range(l))
#
# construction.
#
# X and Y themselves are diagonal and commute.
# ============================================================

X = np.diag(center_x)
Y = np.diag(center_y)


# ============================================================
# Gaussian matrix evaluated AT THE GAUSSIAN CENTERS
# ============================================================
#
# This matrix is square:
#
#       n_basis x n_basis
#
# and plays the role of your original G matrix.
# ============================================================

dx_centers = center_x[:, None] - center_x[None, :]
dy_centers = center_y[:, None] - center_y[None, :]

G_centers = np.exp(
    -(dx_centers**2 + dy_centers**2) / w**2
)


# ============================================================
# Transform X and Y into the Gaussian coefficient basis
# ============================================================
#
# Instead of explicitly computing
#
#     inv(G_centers) @ X @ G_centers
#
# use solve(), which is more numerically stable.
#
# Ax = G^-1 X G
# Ay = G^-1 Y G
# ============================================================

A_x = np.linalg.solve(
    G_centers,
    X @ G_centers,
)

A_y = np.linalg.solve(
    G_centers,
    Y @ G_centers,
)


# ============================================================
# Check commutation
# ============================================================

commutator = A_x @ A_y - A_y @ A_x

print("\n" + "=" * 80)
print("COMMUTATOR CHECK")
print("=" * 80)

print(
    "||[A_x, A_y]|| =",
    np.linalg.norm(commutator),
)


# ============================================================
# Simultaneous diagonalization
# ============================================================
#
# Because Ax and Ay commute, they possess common eigenvectors.
#
# Individually:
#
#     Ax
#
# is degenerate because every Gaussian with the same x
# coordinate shares an eigenvalue.
#
# Likewise Ay is degenerate.
#
# A generic linear combination removes these degeneracies.
#
# Using an irrational coefficient reduces the likelihood of
# accidental degeneracy.
# ============================================================

alpha = np.sqrt(2.0)

A_combined = A_x + alpha * A_y

combined_eigenvalues, eigenvectors = np.linalg.eig(A_combined)


# ============================================================
# Recover X and Y eigenvalues for each common eigenvector
# ============================================================

eigenvalues_x = np.zeros(n_basis)
eigenvalues_y = np.zeros(n_basis)

for i in range(n_basis):

    v = eigenvectors[:, i]

    # Rayleigh quotients
    eigenvalues_x[i] = np.real_if_close(
        np.vdot(v, A_x @ v) / np.vdot(v, v)
    ).real

    eigenvalues_y[i] = np.real_if_close(
        np.vdot(v, A_y @ v) / np.vdot(v, v)
    ).real


# ============================================================
# Sort common eigenvectors by Y, then X
# ============================================================
#
# This gives them a predictable spatial ordering.
# ============================================================

sort_idx = np.lexsort(
    (
        eigenvalues_x,
        eigenvalues_y,
    )
)

eigenvalues_x = eigenvalues_x[sort_idx]
eigenvalues_y = eigenvalues_y[sort_idx]

combined_eigenvalues = combined_eigenvalues[sort_idx]

eigenvectors = eigenvectors[:, sort_idx]


# ============================================================
# Fine spatial grid
# ============================================================
#
# This grid is independent of the number of Gaussian centers.
# Increase n_grid_x and n_grid_y to improve spatial resolution.
# ============================================================

x_grid = np.linspace(
    x_min,
    x_max,
    n_grid_x,
)

y_grid = np.linspace(
    y_min,
    y_max,
    n_grid_y,
)

GRID_X, GRID_Y = np.meshgrid(
    x_grid,
    y_grid,
    indexing="xy",
)

grid_x = GRID_X.ravel()
grid_y = GRID_Y.ravel()

n_grid = len(grid_x)

dx = x_grid[1] - x_grid[0]
dy = y_grid[1] - y_grid[0]

dA = dx * dy


# ============================================================
# Evaluate all Gaussian basis functions on the fine grid
# ============================================================
#
# Shape:
#
#       G_grid.shape =
#           (n_grid_x * n_grid_y, n_basis)
#
# Each COLUMN is one 2D Gaussian basis function.
# ============================================================

grid_dx = grid_x[:, None] - center_x[None, :]
grid_dy = grid_y[:, None] - center_y[None, :]

G_grid = np.exp(
    -(grid_dx**2 + grid_dy**2) / w**2
)


# ============================================================
# Normalize eigenvector signs / phases
# ============================================================
#
# We first contract each eigenvector into real space.
#
# If its largest-magnitude value is negative, flip the
# eigenvector.
# ============================================================

for i in range(n_basis):

    v = eigenvectors[:, i]

    f = G_grid @ v

    max_idx = np.argmax(np.abs(f))

    # For this real problem the eigenvectors should be
    # effectively real.
    if np.real(f[max_idx]) < 0:
        eigenvectors[:, i] *= -1


# ============================================================
# Contract ALL simultaneous eigenvectors over Gaussians
# ============================================================

contracted_eigenvectors = G_grid @ eigenvectors


# ============================================================
# Gram matrices
# ============================================================
#
# The fine spatial grid approximates the continuous inner
# products.
# ============================================================

gram_gaussians = (
    G_grid.conj().T
    @ G_grid
    * dA
)

gram_contracted = (
    contracted_eigenvectors.conj().T
    @ contracted_eigenvectors
    * dA
)


# ============================================================
# Print information BEFORE plotting
# ============================================================

np.set_printoptions(
    precision=4,
    suppress=True,
    linewidth=200,
)

print("\n" + "=" * 80)
print("X EIGENVALUES")
print("=" * 80)
print(eigenvalues_x)

print("\n" + "=" * 80)
print("Y EIGENVALUES")
print("=" * 80)
print(eigenvalues_y)

print("\n" + "=" * 80)
print("GRAM MATRIX OF ORIGINAL 2D GAUSSIANS")
print("=" * 80)
print(gram_gaussians)

print("\n" + "=" * 80)
print("GRAM MATRIX OF CONTRACTED COMMON EIGENFUNCTIONS")
print("=" * 80)
print(gram_contracted)


# ============================================================
# Check simultaneous diagonalization
# ============================================================

V_inv = np.linalg.inv(eigenvectors)

D_x = V_inv @ A_x @ eigenvectors
D_y = V_inv @ A_y @ eigenvectors

offdiag_x = D_x - np.diag(np.diag(D_x))
offdiag_y = D_y - np.diag(np.diag(D_y))

print("\n" + "=" * 80)
print("SIMULTANEOUS DIAGONALIZATION CHECK")
print("=" * 80)

print(
    "Off-diagonal norm of transformed X:",
    np.linalg.norm(offdiag_x),
)

print(
    "Off-diagonal norm of transformed Y:",
    np.linalg.norm(offdiag_y),
)


# ============================================================
# Plot sum of all Gaussian basis functions
# ============================================================

gaussian_sum = np.sum(
    G_grid,
    axis=1,
).reshape(
    n_grid_y,
    n_grid_x,
)

fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(
    111,
    projection="3d",
)

ax.plot_surface(
    GRID_X,
    GRID_Y,
    gaussian_sum,
    cmap="viridis",
)

ax.set_title(
    "Sum of All 2D Gaussian Basis Functions"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Amplitude")

plt.tight_layout()
plt.show()


# ============================================================
# Find Gaussian nearest the center of the domain
# ============================================================

target_x = 0.5 * (x_min + x_max)
target_y = 0.5 * (y_min + y_max)

distance_to_center = (
    (center_x - target_x)**2
    + (center_y - target_y)**2
)

center_basis_idx = np.argmin(
    distance_to_center
)

center_basis_x = center_x[center_basis_idx]
center_basis_y = center_y[center_basis_idx]

print("\n" + "=" * 80)
print("CENTER BASIS FUNCTION")
print("=" * 80)

print(
    "Basis index:",
    center_basis_idx,
)

print(
    "Center:",
    (
        center_basis_x,
        center_basis_y,
    ),
)


# ============================================================
# Find common eigenvector associated with center position
# ============================================================

distance_eigenvalue_to_center = (
    (eigenvalues_x - center_basis_x)**2
    + (eigenvalues_y - center_basis_y)**2
)

center_eigen_idx = np.argmin(
    distance_eigenvalue_to_center
)

print(
    "Common eigenvector index:",
    center_eigen_idx,
)

print(
    "X eigenvalue:",
    eigenvalues_x[center_eigen_idx],
)

print(
    "Y eigenvalue:",
    eigenvalues_y[center_eigen_idx],
)


# ============================================================
# Original center Gaussian
# ============================================================

center_gaussian = G_grid[
    :,
    center_basis_idx
].reshape(
    n_grid_y,
    n_grid_x,
)


# ============================================================
# Contracted simultaneous eigenfunction at the center
# ============================================================

center_contracted = np.real_if_close(
    contracted_eigenvectors[
        :,
        center_eigen_idx
    ]
).reshape(
    n_grid_y,
    n_grid_x,
)


# ============================================================
# Plot original center Gaussian
# ============================================================

fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(
    111,
    projection="3d",
)

ax.plot_surface(
    GRID_X,
    GRID_Y,
    center_gaussian,
    cmap="Oranges",
)

ax.set_title(
    "Original Gaussian at Center"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Amplitude")

plt.tight_layout()
plt.show()


# ============================================================
# Plot contracted common eigenfunction
# ============================================================

fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(
    111,
    projection="3d",
)

ax.plot_surface(
    GRID_X,
    GRID_Y,
    center_contracted,
    cmap="viridis",
)

ax.set_title(
    "Simultaneously Diagonalized Center Function\n"
    f"x = {eigenvalues_x[center_eigen_idx]:.4f}, "
    f"y = {eigenvalues_y[center_eigen_idx]:.4f}"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Amplitude")

plt.tight_layout()
plt.show()