import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from G10 import G10, cf1092, jval

# ============================================================
# Parameters
# ============================================================

n_basis = 123       # match the number of Gaussian parents in G10
n_grid = 1000       # number of spatial grid points

x_min = 0.0
x_max = 122.0

w = 0.5

# Optional comparison curve: stretch each diagonalized function about its
# eigenvalue without moving its center.
show_stretched_diagonalized = True
diagonalized_stretch_factor = 3.0

# ============================================================
# Spatial grid
# ============================================================

x = np.linspace(x_min, x_max, n_grid)
dx = x[1] - x[0]

# ============================================================
# Gaussian centers
# ============================================================

centers = np.linspace(x_min, x_max, n_basis)

# Centers and coefficients of the Gaussian components used by the dilated G10.
# Evaluating G10((x - lambda) / 3) changes the parent centers from j / 3 to
# lambda + j, so their spacing matches the diagonalization basis spacing.
g10_relative_centers = np.asarray(jval, dtype=float)
g10_coefficients = np.asarray(cf1092, dtype=float)
g10_coefficients_normalized = (
    g10_coefficients / np.max(np.abs(g10_coefficients))
)

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

    # A horizontal stretch by s about lambda is f(lambda + (x - lambda) / s).
    # Evaluate the Gaussian expansion directly at the transformed coordinates.
    if show_stretched_diagonalized:
        if diagonalized_stretch_factor <= 0.0:
            raise ValueError("diagonalized_stretch_factor must be positive")

        x_unstretched = eigenvalues[i] + (
            x - eigenvalues[i]
        ) / diagonalized_stretch_factor
        G_unstretched = np.exp(
            -(
                (x_unstretched[:, None] - centers[None, :]) ** 2
            ) / w**2
        )
        f_contracted_stretched = G_unstretched @ v

    # Gaussian centered at corresponding basis center
    gaussian_i = G[:, i]

    # Dilate G10 by a factor of three and translate it so its parent-center
    # spacing matches the diagonalization basis around the eigenvalue.
    g10_at_eigenvalue = np.array(
        G10((x - eigenvalues[i]) / 3.0),
        dtype=float,
        copy=True,
    )

    # Match the G10 peak height to the peak of the diagonalized function.
    g10_peak = np.max(g10_at_eigenvalue)
    diagonalized_peak = np.max(f_contracted)

    if not np.isclose(g10_peak, 0.0):
        g10_at_eigenvalue *= diagonalized_peak / g10_peak

    # A single original G10 parent Gaussian centered at the eigenvalue.
    # G10 uses exp[-0.5 * (3x - j)^2], so a translated parent has the
    # profile exp[-0.5 * 9 * (x - lambda)^2].  Give it the same peak as
    # the diagonalized function for a direct width comparison.
    g10_parent_gaussian = diagonalized_peak * np.exp(
        -0.5 * 9.0 * (x - eigenvalues[i]) ** 2
    )

    # Normalize the diagonalization coefficients so both center distributions
    # can be compared on the same vertical scale.
    v_normalized = v / np.max(np.abs(v))
    g10_centers = eigenvalues[i] + g10_relative_centers

    fig, (ax_functions, ax_centers) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax_functions.plot(
        x,
        f_contracted,
        label=rf"Contracted eigenfunction {i}",
        color="navy",
    )

    if show_stretched_diagonalized:
        ax_functions.plot(
            x,
            f_contracted_stretched,
            label=(
                rf"Diagonalized function stretched $\times"
                rf"{diagonalized_stretch_factor:g}$"
            ),
            color="deepskyblue",
            linestyle="--",
            linewidth=1.8,
        )

    ax_functions.plot(
        x,
        gaussian_i,
        label=rf"Gaussian {i}",
        color="darkorange",
        linestyle="--",
    )

    ax_functions.plot(
        x,
        g10_at_eigenvalue,
        label=rf"Dilated G10 centered at $\lambda_{i}$",
        color="crimson",
        linestyle=":",
        linewidth=2,
    )

    ax_functions.plot(
        x,
        g10_parent_gaussian,
        label=rf"G10 parent Gaussian at $\lambda_{i}$",
        color="forestgreen",
        linestyle="-.",
        linewidth=1.8,
    )

    ax_functions.set_title(
        f"Eigenfunction {i}, "
        f"Eigenvalue = {eigenvalues[i]:.6f}"
    )

    ax_functions.set_ylabel("Amplitude")
    ax_functions.grid(True, linestyle="--", alpha=0.6)
    ax_functions.legend()

    ax_centers.vlines(
        centers,
        0.0,
        v_normalized,
        color="navy",
        alpha=0.75,
        linewidth=1.2,
        label="Diagonalized Gaussian weights",
    )
    ax_centers.scatter(
        centers,
        v_normalized,
        color="navy",
        s=14,
    )

    ax_centers.vlines(
        g10_centers,
        0.0,
        g10_coefficients_normalized,
        color="crimson",
        alpha=0.65,
        linewidth=1.0,
        label="Dilated G10 Gaussian weights",
    )
    ax_centers.scatter(
        g10_centers,
        g10_coefficients_normalized,
        color="crimson",
        s=12,
    )
    ax_centers.axhline(0.0, color="black", linewidth=0.8)
    ax_centers.set_xlabel("Gaussian center")
    ax_centers.set_ylabel("Normalized\ncoefficient")
    ax_centers.grid(True, linestyle="--", alpha=0.6)
    ax_centers.legend()

    fig.tight_layout()
    plt.show()
    plt.close(fig)
