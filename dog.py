import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

# ============================================================
# Parameters
# ============================================================

n_basis = 116       # number of Gaussian basis functions
n_grid = 1000       # number of spatial grid points

x_min = 0.0
x_max = 115.0

w = 1.0

# ============================================================
# Spatial grid
# ============================================================

x = np.linspace(x_min, x_max, n_grid)
dx = x[1] - x[0]

# ============================================================
# Gaussian centers
# ============================================================

centers = np.linspace(x_min, x_max, n_basis)

# G has shape:
#
#     (number of spatial points, number of Gaussians)
#
#     n_grid x n_basis
#
G = np.exp(
    -((x[:, None] - centers[None, :]) ** 2) / w**2
)

# ============================================================
# Position operator on the fine spatial grid
# ============================================================

X = np.diag(x)

# ============================================================
# Project operator into the Gaussian basis
# ============================================================

# Include dx because these matrix products approximate integrals
S = G.T @ G * dx
H = G.T @ X @ G * dx

# ============================================================
# Solve generalized eigenvalue problem
#
#       H c = lambda S c
#
# eigh automatically returns eigenvalues in ascending order
# ============================================================

eigenvalues, eigenvectors = eigh(H, S)

# ============================================================
# Contract eigenvectors back onto spatial grid
# ============================================================

contracted_eigenvectors = G @ eigenvectors

# ============================================================
# Gram matrices
# ============================================================

gram_gaussians = G.T @ G * dx

gram_contracted = (
    contracted_eigenvectors.T
    @ contracted_eigenvectors
    * dx
)

np.set_printoptions(
    precision=4,
    suppress=True,
    linewidth=200,
)

print("=" * 80)
print("GRAM MATRIX OF ORIGINAL GAUSSIANS")
print("=" * 80)
print(gram_gaussians)

print("\n" + "=" * 80)
print("GRAM MATRIX OF CONTRACTED EIGENFUNCTIONS")
print("=" * 80)
print(gram_contracted)

# ============================================================
# Sum of Gaussians
# ============================================================

# gaussian_sum = np.sum(G, axis=1)

# plt.figure(figsize=(10, 5))
# plt.plot(
#     x,
#     gaussian_sum,
#     label="Sum of Gaussians",
#     color="darkred",
# )
# plt.title("Sum of Gaussians")
# plt.xlabel("x")
# plt.ylabel("Amplitude")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.legend()
# plt.tight_layout()
# plt.show()

# ============================================================
# Plot eigenfunctions
# ============================================================

for i in range(eigenvectors.shape[1]):

    v = eigenvectors[:, i].copy()

    # Contract onto fine spatial grid
    f_contracted = G @ v

    # Flip sign so largest-magnitude point is positive
    max_idx = np.argmax(np.abs(f_contracted))

    if f_contracted[max_idx] < 0:
        v *= -1
        f_contracted *= -1

    # Gaussian centered at corresponding basis center
    gaussian_i = G[:, i]

    plt.figure(figsize=(10, 5))

    plt.plot(
        x,
        f_contracted,
        label=rf"Contracted eigenfunction {i}",
        color="navy",
    )

    plt.plot(
        x,
        gaussian_i,
        label=rf"Gaussian {i}",
        color="darkorange",
        linestyle="--",
    )

    plt.title(
        f"Eigenfunction {i}, "
        f"Eigenvalue = {eigenvalues[i]:.6f}"
    )

    plt.xlabel("x")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()