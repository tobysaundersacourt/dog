import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Parameters
# ============================================================

# Arbitrary list of Gaussian function centers:
#
#     [(x_0, y_0),
#      (x_1, y_1),
#      ...
#      (x_N, y_N)]
#
# These do NOT need to lie on a regular grid.

# Hexagonal grid of Gaussian centers

# Size of the physical region we actually care about
n_rows = 11
n_cols = 11

# Number of extra staggered hexagonal layers added around the physical region.
# These ghost functions participate in the basis transformation and
# diagonalization, but the plots/diagnostics focus on the physical interior.
n_ghost_layers = 2

spacing = 1.0

function_centers = []
center_rows = []
center_cols = []
physical_center_mask_list = []

# Build an extended staggered hexagonal grid.
for row in range(-n_ghost_layers, n_rows + n_ghost_layers):
    y = row * spacing * np.sqrt(3) / 2

    # Shift every other row by half a spacing
    x_offset = 0.5 * spacing if row % 2 == 1 else 0.0

    for col in range(-n_ghost_layers, n_cols + n_ghost_layers):
        x = col * spacing + x_offset

        function_centers.append((x, y))
        center_rows.append(row)
        center_cols.append(col)

        physical_center_mask_list.append(
            (0 <= row < n_rows)
            and (0 <= col < n_cols)
        )

function_centers = np.asarray(function_centers, dtype=float)
center_rows = np.asarray(center_rows, dtype=int)
center_cols = np.asarray(center_cols, dtype=int)
physical_center_mask = np.asarray(physical_center_mask_list, dtype=bool)

# Center the PHYSICAL hexagonal patch around (0, 0). The ghost grid is shifted
# by exactly the same amount so it remains a continuation of the lattice.
physical_mean = np.mean(
    function_centers[physical_center_mask],
    axis=0,
)

function_centers -= physical_mean

# Physical-region bounds. These are the bounds we will trust and display.
physical_centers = function_centers[physical_center_mask]

physical_x_min = np.min(physical_centers[:, 0])
physical_x_max = np.max(physical_centers[:, 0])
physical_y_min = np.min(physical_centers[:, 1])
physical_y_max = np.max(physical_centers[:, 1])

# Gaussian width
w = 1.0

# Fine spatial resolution used for evaluating / plotting functions
n_grid_x = 150
n_grid_y = 150

# Amount of extra plotting space around outermost Gaussian centers
plot_padding = 3.0 * w


# ============================================================
# Convert function centers to arrays
# ============================================================

function_centers = np.asarray(
    function_centers,
    dtype=float,
)

if function_centers.ndim != 2 or function_centers.shape[1] != 2:
    raise ValueError(
        "function_centers must be a list/array of (x, y) pairs."
    )

center_x = function_centers[:, 0]
center_y = function_centers[:, 1]

n_basis = len(function_centers)

print("=" * 80)
print("BASIS INFORMATION")
print("=" * 80)

print("Number of physical Gaussian centers:", np.sum(physical_center_mask))
print("Number of ghost Gaussian centers:", np.sum(~physical_center_mask))
print("Total number of Gaussian basis functions:", n_basis)

print("\nFunction centers:")
print(function_centers)


# ============================================================
# Original X and Y operators
# ============================================================
#
# Thinking of the Gaussian basis functions as a vector:
#
#     phi = [phi_0, phi_1, ..., phi_N]
#
# X and Y are initially diagonal:
#
#     X = diag(x_0, x_1, ..., x_N)
#
#     Y = diag(y_0, y_1, ..., y_N)
#
# ============================================================

X = np.diag(center_x)
Y = np.diag(center_y)

print("\n" + "=" * 80)
print("ORIGINAL X OPERATOR")
print("=" * 80)
print(X)

print("\n" + "=" * 80)
print("ORIGINAL Y OPERATOR")
print("=" * 80)
print(Y)


# ============================================================
# Gaussian transformation matrix evaluated at function centers
# ============================================================
#
# G_centers[i, j]
#
# is Gaussian j evaluated at center i.
#
# Thus:
#
#                 | phi_0(r_0)  phi_1(r_0) ... |
#                 | phi_0(r_1)  phi_1(r_1) ... |
#     G_centers = |      ...         ...       |
#
#
# For isotropic 2D Gaussians:
#
# phi_j(x, y) =
#
#     exp(
#         -[(x-x_j)^2 + (y-y_j)^2] / w^2
#     )
#
# ============================================================

delta_x = (
    center_x[:, None]
    - center_x[None, :]
)

delta_y = (
    center_y[:, None]
    - center_y[None, :]
)

G_centers = np.exp(
    -(
        delta_x**2
        + delta_y**2
    )
    / w**2
)


# ============================================================
# Check conditioning
# ============================================================
#
# If Gaussian centers are extremely close together or w is
# very large, G_centers can become poorly conditioned.
# ============================================================

condition_number = np.linalg.cond(G_centers)

print("\n" + "=" * 80)
print("GAUSSIAN TRANSFORMATION MATRIX")
print("=" * 80)

print(G_centers)

print("\nCondition number of G_centers:")
print(condition_number)


# ============================================================
# Transform X and Y into Gaussian coefficient space
# ============================================================
#
# A_x = G^-1 X G
#
# A_y = G^-1 Y G
#
# solve() is preferable to explicitly calculating inv(G).
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
# Check that transformed X and Y commute
# ============================================================
#
# Since
#
#     [X, Y] = 0
#
# and both undergo the same similarity transformation,
#
#     [A_x, A_y]
#         = G^-1 [X,Y] G
#         = 0.
#
# ============================================================

commutator = (
    A_x @ A_y
    - A_y @ A_x
)

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
# X or Y individually may have degeneracies.
#
# We therefore diagonalize a generic linear combination:
#
#     A_combined = A_x + alpha A_y
#
# Since A_x and A_y commute, an eigenvector of a
# nondegenerate A_combined is simultaneously an eigenvector
# of both.
#
# An irrational alpha helps avoid accidental degeneracies.
# ============================================================

alpha = np.sqrt(2.0)

A_combined = (
    A_x
    + alpha * A_y
)

combined_eigenvalues, eigenvectors = np.linalg.eig(
    A_combined
)


# ============================================================
# Obtain X and Y eigenvalues of each common eigenvector
# ============================================================

eigenvalues_x = np.zeros(n_basis)
eigenvalues_y = np.zeros(n_basis)

for i in range(n_basis):

    v = eigenvectors[:, i]

    eigenvalues_x[i] = np.real_if_close(
        np.vdot(
            v,
            A_x @ v,
        )
        / np.vdot(v, v)
    ).real

    eigenvalues_y[i] = np.real_if_close(
        np.vdot(
            v,
            A_y @ v,
        )
        / np.vdot(v, v)
    ).real


# ============================================================
# Sort common eigenvectors by position
# ============================================================
#
# Sort primarily by y, then by x.
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

# Ensure arbitrary complex phase normalization below is always legal.
eigenvectors = eigenvectors.astype(complex)


# ============================================================
# Fine spatial grid
# ============================================================
#
# This grid is independent of the number and arrangement of
# Gaussian centers.
# ============================================================

# Evaluate functions on a box large enough to include the ghost basis.
# This is important because basis integrals should include the ghost tails.
x_min = np.min(center_x) - plot_padding
x_max = np.max(center_x) + plot_padding

y_min = np.min(center_y) - plot_padding
y_max = np.max(center_y) + plot_padding

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

dx = x_grid[1] - x_grid[0]
dy = y_grid[1] - y_grid[0]

dA = dx * dy

# Spatial points inside the physical region. Diagnostics reported as
# "interior" use only these points, while the basis integrals use the full
# extended grid containing the ghost layer.
interior_spatial_mask = (
    (grid_x >= physical_x_min)
    & (grid_x <= physical_x_max)
    & (grid_y >= physical_y_min)
    & (grid_y <= physical_y_max)
)


# ============================================================
# Evaluate Gaussian basis on fine spatial grid
# ============================================================
#
# Shape:
#
#     G_grid.shape =
#
#         (n_grid_x * n_grid_y, n_basis)
#
# Each column corresponds to one Gaussian basis function.
# ============================================================

grid_delta_x = (
    grid_x[:, None]
    - center_x[None, :]
)

grid_delta_y = (
    grid_y[:, None]
    - center_y[None, :]
)

G_grid = np.exp(
    -(
        grid_delta_x**2
        + grid_delta_y**2
    )
    / w**2
)


# ============================================================
# Normalize sign / phase of simultaneous eigenvectors
# ============================================================
#
# The sign of an eigenvector is arbitrary.
#
# Contract each eigenvector over the Gaussians and require
# that its largest-magnitude spatial value be positive.
# ============================================================

for i in range(n_basis):

    v = eigenvectors[:, i]

    f_contracted = G_grid @ v

    max_idx = np.argmax(
        np.abs(f_contracted)
    )

    phase = np.angle(
        f_contracted[max_idx]
    )

    # Remove arbitrary complex phase
    eigenvectors[:, i] *= np.exp(
        -1j * phase
    )


# ============================================================
# Contract ALL eigenvectors over Gaussian basis
# ============================================================

contracted_eigenvectors = (
    G_grid @ eigenvectors
)

# ============================================================
# Physical diagonalized basis functions
# ============================================================
#
# Ghost centers participate in the transformation/diagonalization, but
# interpolation and partition-of-unity diagnostics use only the states
# associated with the original physical centers.
# ============================================================

physical_basis_indices = np.where(physical_center_mask)[0]

physical_eigen_indices = []

for basis_idx in physical_basis_indices:
    x0 = center_x[basis_idx]
    y0 = center_y[basis_idx]

    distance = (
        (eigenvalues_x - x0)**2
        + (eigenvalues_y - y0)**2
    )

    physical_eigen_indices.append(np.argmin(distance))

physical_eigen_indices = np.asarray(physical_eigen_indices, dtype=int)

physical_eigen_x = eigenvalues_x[physical_eigen_indices]
physical_eigen_y = eigenvalues_y[physical_eigen_indices]

physical_contracted_eigenvectors = (
    contracted_eigenvectors[:, physical_eigen_indices]
)


# ============================================================
# Gram matrices
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
# Print results before plotting
# ============================================================

np.set_printoptions(
    precision=4,
    suppress=True,
    linewidth=200,
)

print("\n" + "=" * 80)
print("SORTED X EIGENVALUES")
print("=" * 80)
print(eigenvalues_x)

print("\n" + "=" * 80)
print("SORTED Y EIGENVALUES")
print("=" * 80)
print(eigenvalues_y)

print("\n" + "=" * 80)
print("RECOVERED (X, Y) EIGENVALUE PAIRS")
print("=" * 80)

for i in range(n_basis):

    print(
        f"{i:4d}: "
        f"("
        f"{eigenvalues_x[i]: .8f}, "
        f"{eigenvalues_y[i]: .8f}"
        f")"
    )


print("\n" + "=" * 80)
print("GRAM MATRIX OF ORIGINAL GAUSSIANS")
print("=" * 80)
print(gram_gaussians)

print("\n" + "=" * 80)
print("GRAM MATRIX OF CONTRACTED EIGENFUNCTIONS")
print("=" * 80)
print(gram_contracted)


# ============================================================
# Check simultaneous diagonalization
# ============================================================
#
# Since this is a similarity transformation problem, use
#
#     V^-1 A V
#
# rather than V^† A V.
# ============================================================

V_inv = np.linalg.inv(eigenvectors)

D_x = (
    V_inv
    @ A_x
    @ eigenvectors
)

D_y = (
    V_inv
    @ A_y
    @ eigenvectors
)

offdiag_x = (
    D_x
    - np.diag(np.diag(D_x))
)

offdiag_y = (
    D_y
    - np.diag(np.diag(D_y))
)

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
)

gaussian_sum = gaussian_sum.reshape(
    n_grid_y,
    n_grid_x,
)

fig = plt.figure(
    figsize=(10, 7)
)

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

# Show Gaussian centers on xy plane
ax.scatter(
    center_x,
    center_y,
    np.zeros(n_basis),
    color="black",
    s=25,
)

ax.set_title(
    "Sum of All Gaussian Basis Functions"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Amplitude")

plt.tight_layout()
plt.show()


# ============================================================
# Select basis function closest to geometric center
# ============================================================

target_x = 0.0
target_y = 0.0

physical_indices = np.where(
    physical_center_mask
)[0]

distance_to_center = (
    (center_x[physical_indices] - target_x)**2
    + (center_y[physical_indices] - target_y)**2
)

center_basis_idx = physical_indices[
    np.argmin(distance_to_center)
]

center_basis_x = center_x[
    center_basis_idx
]

center_basis_y = center_y[
    center_basis_idx
]

print("\n" + "=" * 80)
print("SELECTED CENTER BASIS FUNCTION")
print("=" * 80)

print(
    "Basis index:",
    center_basis_idx,
)

print(
    "Function center:",
    (
        center_basis_x,
        center_basis_y,
    ),
)


# ============================================================
# Find common eigenvector associated with that position
# ============================================================

distance_to_eigenvalue = (
    (
        eigenvalues_x
        - center_basis_x
    )**2
    +
    (
        eigenvalues_y
        - center_basis_y
    )**2
)

center_eigen_idx = np.argmin(
    distance_to_eigenvalue
)

print(
    "Common eigenvector index:",
    center_eigen_idx,
)

print(
    "Recovered position eigenvalue:",
    (
        eigenvalues_x[center_eigen_idx],
        eigenvalues_y[center_eigen_idx],
    ),
)


# ============================================================
# Extract original Gaussian
# ============================================================

center_gaussian = G_grid[
    :,
    center_basis_idx
]

center_gaussian = center_gaussian.reshape(
    n_grid_y,
    n_grid_x,
)


# ============================================================
# Extract contracted simultaneous eigenfunction
# ============================================================

center_contracted = contracted_eigenvectors[
    :,
    center_eigen_idx
]

center_contracted = np.real_if_close(
    center_contracted
)

center_contracted = center_contracted.reshape(
    n_grid_y,
    n_grid_x,
)


# ============================================================
# Plot original selected Gaussian
# ============================================================

fig = plt.figure(
    figsize=(10, 7)
)

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
    "Original Selected Gaussian\n"
    f"Center = "
    f"({center_basis_x:.4f}, "
    f"{center_basis_y:.4f})"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Amplitude")

plt.tight_layout()
plt.show()


# ============================================================
# Plot corresponding simultaneously diagonalized function
# ============================================================

fig = plt.figure(
    figsize=(10, 7)
)

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
    "Simultaneously Diagonalized Function\n"
    f"X eigenvalue = "
    f"{eigenvalues_x[center_eigen_idx]:.4f}, "
    f"Y eigenvalue = "
    f"{eigenvalues_y[center_eigen_idx]:.4f}"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Amplitude")

plt.tight_layout()
plt.show()

# ============================================================
# Plot Gaussian function centers in the X-Y plane
# ============================================================

plt.figure(figsize=(8, 8))

# Plot physical Gaussian centers
plt.scatter(
    center_x[physical_center_mask],
    center_y[physical_center_mask],
    s=60,
    label="Physical Gaussian centers",
)

# Plot ghost/buffer centers separately
plt.scatter(
    center_x[~physical_center_mask],
    center_y[~physical_center_mask],
    s=35,
    marker="x",
    label="Ghost Gaussian centers",
)

# Highlight the currently selected physical basis function
plt.scatter(
    center_basis_x,
    center_basis_y,
    s=180,
    marker="*",
    label="Selected physical center",
)

plt.title("Gaussian Function Centers")
plt.xlabel("x")
plt.ylabel("y")

plt.grid(
    True,
    linestyle="--",
    alpha=0.6,
)

plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Plot sum of diagonalized basis functions along a zig-zag chain
# ============================================================

# Contract all diagonalized basis functions onto the spatial grid
contracted_eigenvectors = G_grid @ eigenvectors

# Choose a simple zig-zag chain from the two middle rows of the
# staggered hexagonal arrangement.
#
# With the construction above, basis indices are ordered row-by-row:
#
#     basis index = row * n_cols + col
#
# We alternate between the two middle rows while moving across columns.

lower_row = (n_rows - 1) // 2
upper_row = lower_row + 1

zigzag_basis_indices = []

for col in range(n_cols):

    lower_match = np.where(
        (center_rows == lower_row)
        & (center_cols == col)
        & physical_center_mask
    )[0]

    if len(lower_match) == 1:
        zigzag_basis_indices.append(lower_match[0])

    if upper_row < n_rows:
        upper_match = np.where(
            (center_rows == upper_row)
            & (center_cols == col)
            & physical_center_mask
        )[0]

        if len(upper_match) == 1:
            zigzag_basis_indices.append(upper_match[0])

zigzag_basis_indices = np.asarray(
    zigzag_basis_indices,
    dtype=int,
)

# Match each original Gaussian center in the zig-zag chain to the
# simultaneously diagonalized function with the nearest (X, Y)
# eigenvalue pair.
zigzag_eigen_indices = []

for basis_idx in zigzag_basis_indices:
    x0 = center_x[basis_idx]
    y0 = center_y[basis_idx]

    distance = (
        (eigenvalues_x - x0)**2
        + (eigenvalues_y - y0)**2
    )

    zigzag_eigen_indices.append(np.argmin(distance))

zigzag_eigen_indices = np.asarray(zigzag_eigen_indices, dtype=int)

# Sum ONLY the diagonalized functions belonging to the zig-zag chain
zigzag_sum = np.sum(
    contracted_eigenvectors[:, zigzag_eigen_indices],
    axis=1,
)

zigzag_sum = np.real_if_close(zigzag_sum)

zigzag_sum = zigzag_sum.reshape(
    n_grid_y,
    n_grid_x,
)

# Coordinates of the selected function centers
zigzag_x = center_x[zigzag_basis_indices]
zigzag_y = center_y[zigzag_basis_indices]

# Surface plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    GRID_X,
    GRID_Y,
    zigzag_sum,
    cmap="viridis",
    alpha=0.85,
)

# Plot the selected zig-zag function centers on the z = 0 plane
ax.plot(
    zigzag_x,
    zigzag_y,
    np.zeros_like(zigzag_x),
    "o-",
    linewidth=2,
    markersize=6,
    label="Zig-zag chain",
)

ax.set_title(
    "Sum of Diagonalized Basis Functions Along Zig-Zag Chain"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Sum")

ax.legend()

plt.tight_layout()
plt.show()

# ============================================================
# Interpolating fit using the diagonalized basis functions
# ============================================================

def interpolating_gaussian_fit(
    target_width_factor=5.0,
    target_center=(0.0, 0.0),
    plot_target=True,
):
    """
    Interpolate a broad Gaussian using the diagonalized basis functions:

        g_fit(x, y) = sum_i g(x_i, y_i) * phi_i(x, y) / w_i

    with

        w_i = integral phi_i(x, y) dx dy.

    Here (x_i, y_i) are the simultaneous X/Y eigenvalues associated
    with the diagonalized basis function phi_i.
    """

    target_width = target_width_factor * w
    x0, y0 = target_center

    # Exact target Gaussian on the fine spatial grid
    target_flat = np.exp(
        -(
            (grid_x - x0)**2
            + (grid_y - y0)**2
        )
        / target_width**2
    )

    target_function = target_flat.reshape(
        n_grid_y,
        n_grid_x,
    )

    # Evaluate target at the centers of the diagonalized functions
    target_at_centers = np.exp(
        -(
            (physical_eigen_x - x0)**2
            + (physical_eigen_y - y0)**2
        )
        / target_width**2
    )

    # Numerical integrals w_i = integral phi_i(x,y) dx dy
    basis_weights = (
        np.sum(
            physical_contracted_eigenvectors,
            axis=0,
        )
        * dA
    )

    basis_weights = np.real_if_close(
        basis_weights
    )

    # Guard against division by approximately zero
    weight_scale = np.max(
        np.abs(basis_weights)
    )

    weight_tolerance = (
        1e-12
        * max(weight_scale, 1.0)
    )

    if np.any(
        np.abs(basis_weights)
        < weight_tolerance
    ):
        bad_indices = np.where(
            np.abs(basis_weights)
            < weight_tolerance
        )[0]

        raise ValueError(
            "One or more diagonalized basis functions have nearly zero "
            "integral, so the interpolation formula divides by approximately "
            "zero. Problematic basis indices: "
            f"{bad_indices.tolist()}"
        )

    # Area represented by one center in a hexagonal lattice:
    #
    #     A_cell = (sqrt(3) / 2) * spacing^2
    #
    # Include this Riemann-sum area factor in the interpolation weights.
    cell_area = (
        np.sqrt(3.0)
        / 2.0
        * spacing**2
    )

    # c_i = A_cell * g(x_i, y_i) / w_i
    interpolation_coefficients = (
        cell_area
        * target_at_centers
        / basis_weights
    )

    # g_fit(x,y) = sum_i c_i phi_i(x,y)
    fitted_flat = (
        physical_contracted_eigenvectors
        @ interpolation_coefficients
    )

    fitted_flat = np.real_if_close(
        fitted_flat
    )

    fitted_function = fitted_flat.reshape(
        n_grid_y,
        n_grid_x,
    )

    # Diagnostics
    error = (
        fitted_function
        - target_function
    )

    error_flat = error.ravel()
    interior_error = error_flat[
        interior_spatial_mask
    ]

    rms_error = np.sqrt(
        np.mean(
            np.abs(interior_error)**2
        )
    )

    max_error = np.max(
        np.abs(interior_error)
    )

    print("\n" + "=" * 80)
    print("INTERPOLATING GAUSSIAN FIT")
    print("=" * 80)

    print("Target center:", target_center)
    print("Original Gaussian width:", w)
    print("Target Gaussian width:", target_width)
    print("Hexagonal cell area:", cell_area)
    print(
        "Physical functions used in interpolation:",
        physical_contracted_eigenvectors.shape[1],
    )
    print("Ghost-associated functions used in interpolation:", 0)

    print(
        "Minimum |basis integral|:",
        np.min(
            np.abs(basis_weights)
        ),
    )

    print(
        "Maximum |basis integral|:",
        np.max(
            np.abs(basis_weights)
        ),
    )

    print(
        "RMS interpolation error:",
        rms_error,
    )

    print(
        "Maximum interpolation error:",
        max_error,
    )

    # ------------------------------------------------------------
    # Plot the original target Gaussian FIRST
    # ------------------------------------------------------------

    if plot_target:

        fig = plt.figure(
            figsize=(10, 7)
        )

        ax = fig.add_subplot(
            111,
            projection="3d",
        )

        ax.plot_surface(
            GRID_X,
            GRID_Y,
            target_function,
            cmap="Oranges",
        )

        ax.set_title(
            "Original Target Gaussian\n"
            f"Width = {target_width_factor:.1f} x original width"
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("Amplitude")

        ax.set_xlim(physical_x_min, physical_x_max)
        ax.set_ylim(physical_y_min, physical_y_max)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Plot interpolating fitted function SECOND
    # ------------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot_surface(
        GRID_X,
        GRID_Y,
        fitted_function,
        cmap="viridis",
    )

    ax.scatter(
        physical_eigen_x,
        physical_eigen_y,
        np.zeros_like(physical_eigen_x),
        s=25,
        label="Diagonalized function centers",
    )

    ax.set_title(
        "Interpolating Fit to Centered Gaussian\n"
        f"Target width = {target_width_factor:.1f} x original width"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Amplitude")

    ax.set_xlim(physical_x_min, physical_x_max)
    ax.set_ylim(physical_y_min, physical_y_max)

    ax.legend()

    plt.tight_layout()
    plt.show()

    return (
        fitted_function,
        target_function,
        basis_weights,
        interpolation_coefficients,
    )


# ============================================================
# Run the interpolation fit
# ============================================================

fitted_function, target_function, basis_weights, interpolation_coefficients = (
    interpolating_gaussian_fit(
        target_width_factor=5.0,
        target_center=(0.0, 0.0),
        plot_target=True,
    )
)

# ============================================================
# Galerkin fit using scalar products with the diagonalized basis
# ============================================================

def galerkin_gaussian_fit(
    target_width_factor=5.0,
    target_center=(0.0, 0.0),
):
    """
    Fit the same target Gaussian using the coefficient rule

        c_i = <phi_i, g> / <phi_i, phi_i>

    and then contract those coefficients over the diagonalized basis:

        g_Galerkin(x,y) = sum_i c_i phi_i(x,y)

    This is the orthogonal-basis projection formula with explicit
    normalization by each basis-function norm. It is exact as a Galerkin
    projection when the phi_i are mutually orthogonal. If residual
    off-diagonal overlaps remain, the full nonorthogonal Galerkin system
    would instead solve S c = b.
    """

    target_width = target_width_factor * w
    x0, y0 = target_center

    # Exact target function on the fine spatial grid
    target_flat = np.exp(
        -(
            (grid_x - x0)**2
            + (grid_y - y0)**2
        )
        / target_width**2
    )

    target_function_galerkin = target_flat.reshape(
        n_grid_y,
        n_grid_x,
    )

    # ------------------------------------------------------------
    # Scalar products <phi_i, g>
    # ------------------------------------------------------------

    scalar_products = (
        contracted_eigenvectors.conj().T
        @ target_flat
        * dA
    )

    # ------------------------------------------------------------
    # Norm squared <phi_i, phi_i>
    # ------------------------------------------------------------

    basis_norm_squared = (
        np.sum(
            np.abs(contracted_eigenvectors)**2,
            axis=0,
        )
        * dA
    )

    norm_scale = np.max(
        np.abs(basis_norm_squared)
    )

    norm_tolerance = (
        1e-14
        * max(norm_scale, 1.0)
    )

    if np.any(
        np.abs(basis_norm_squared)
        < norm_tolerance
    ):
        bad_indices = np.where(
            np.abs(basis_norm_squared)
            < norm_tolerance
        )[0]

        raise ValueError(
            "One or more diagonalized basis functions have nearly zero "
            "norm. Problematic basis indices: "
            f"{bad_indices.tolist()}"
        )

    # ------------------------------------------------------------
    # Galerkin coefficients
    #
    #     c_i = <phi_i, g> / <phi_i, phi_i>
    # ------------------------------------------------------------

    galerkin_coefficients = (
        scalar_products
        / basis_norm_squared
    )

    # ------------------------------------------------------------
    # Contract coefficients over basis functions
    # ------------------------------------------------------------

    galerkin_flat = (
        contracted_eigenvectors
        @ galerkin_coefficients
    )

    galerkin_flat = np.real_if_close(
        galerkin_flat
    )

    galerkin_function = galerkin_flat.reshape(
        n_grid_y,
        n_grid_x,
    )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    galerkin_error = (
        galerkin_function
        - target_function_galerkin
    )

    galerkin_error_flat = galerkin_error.ravel()
    galerkin_interior_error = galerkin_error_flat[
        interior_spatial_mask
    ]

    galerkin_rms_error = np.sqrt(
        np.mean(
            np.abs(galerkin_interior_error)**2
        )
    )

    galerkin_max_error = np.max(
        np.abs(galerkin_interior_error)
    )

    print("\n" + "=" * 80)
    print("GALERKIN GAUSSIAN FIT")
    print("=" * 80)

    print(
        "Target center:",
        target_center,
    )

    print(
        "Target Gaussian width:",
        target_width,
    )

    print(
        "Minimum basis norm squared:",
        np.min(
            np.abs(basis_norm_squared)
        ),
    )

    print(
        "Maximum basis norm squared:",
        np.max(
            np.abs(basis_norm_squared)
        ),
    )

    print(
        "RMS Galerkin error:",
        galerkin_rms_error,
    )

    print(
        "Maximum Galerkin error:",
        galerkin_max_error,
    )

    # ------------------------------------------------------------
    # Plot Galerkin fitted function THIRD, independently autoscaled
    # ------------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot_surface(
        GRID_X,
        GRID_Y,
        galerkin_function,
        cmap="viridis",
    )

    ax.scatter(
        eigenvalues_x,
        eigenvalues_y,
        np.zeros_like(eigenvalues_x),
        s=25,
        label="Diagonalized function centers",
    )

    ax.set_title(
        "Galerkin Fit to Centered Gaussian\n"
        f"Target width = {target_width_factor:.1f} x original width"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Amplitude")

    ax.set_xlim(physical_x_min, physical_x_max)
    ax.set_ylim(physical_y_min, physical_y_max)

    ax.legend()

    plt.tight_layout()
    plt.show()

    return (
        galerkin_function,
        target_function_galerkin,
        basis_norm_squared,
        scalar_products,
        galerkin_coefficients,
    )


# ============================================================
# Run the Galerkin fit
# ============================================================

(
    galerkin_function,
    target_function_galerkin,
    galerkin_basis_norm_squared,
    galerkin_scalar_products,
    galerkin_coefficients,
) = galerkin_gaussian_fit(
    target_width_factor=5.0,
    target_center=(0.0, 0.0),
)

# ============================================================
# Partition-of-unity diagnostic
# ============================================================

def plot_partition_of_unity():
    """
    Test whether the area-corrected, integral-normalized diagonalized basis

        A_cell * sum_i phi_i(x,y) / w_i

    approximately reproduces the constant function 1.

    Here

        w_i = integral phi_i(x,y) dx dy

    and

        A_cell = (sqrt(3)/2) * spacing^2

    for the hexagonal lattice.
    """

    # Integral of each diagonalized basis function
    basis_integrals = (
        np.sum(
            physical_contracted_eigenvectors,
            axis=0,
        )
        * dA
    )

    basis_integrals = np.real_if_close(
        basis_integrals
    )

    # Hexagonal area represented by one function center
    cell_area = (
        np.sqrt(3.0)
        / 2.0
        * spacing**2
    )

    # Protect against division by numerically zero integrals
    integral_scale = np.max(
        np.abs(basis_integrals)
    )

    integral_tolerance = (
        1e-12
        * max(integral_scale, 1.0)
    )

    bad_indices = np.where(
        np.abs(basis_integrals)
        < integral_tolerance
    )[0]

    print("\n" + "=" * 80)
    print("PARTITION-OF-UNITY DIAGNOSTIC")
    print("=" * 80)

    print(
        "Hexagonal cell area:",
        cell_area,
    )

    print(
        "Physical functions used in partition sum:",
        physical_contracted_eigenvectors.shape[1],
    )
    print("Ghost-associated functions used in partition sum:", 0)

    print(
        "Basis integrals w_i:",
    )
    print(basis_integrals)

    print(
        "Minimum |w_i|:",
        np.min(
            np.abs(basis_integrals)
        ),
    )

    print(
        "Maximum |w_i|:",
        np.max(
            np.abs(basis_integrals)
        ),
    )

    if len(bad_indices) > 0:
        print(
            "WARNING: basis functions with nearly zero integral:",
            bad_indices.tolist(),
        )

    # Only divide where numerically safe
    safe_integrals = basis_integrals.copy()

    safe_integrals[
        np.abs(safe_integrals)
        < integral_tolerance
    ] = np.nan

    # ------------------------------------------------------------
    # Partition-of-unity field:
    #
    #     P(x,y) = A_cell * sum_i phi_i(x,y) / w_i
    #
    # ------------------------------------------------------------

    partition_flat = (
        physical_contracted_eigenvectors
        @ (
            cell_area
            / safe_integrals
        )
    )

    partition_flat = np.real_if_close(
        partition_flat
    )

    partition = partition_flat.reshape(
        n_grid_y,
        n_grid_x,
    )

    partition_flat_for_stats = partition.ravel()

    finite_partition = partition_flat_for_stats[
        interior_spatial_mask
        & np.isfinite(partition_flat_for_stats)
    ]

    if finite_partition.size > 0:
        print(
            "Partition minimum:",
            np.min(finite_partition),
        )

        print(
            "Partition maximum:",
            np.max(finite_partition),
        )

        print(
            "Partition mean:",
            np.mean(finite_partition),
        )

        print(
            "RMS deviation from 1:",
            np.sqrt(
                np.mean(
                    (finite_partition - 1.0)**2
                )
            ),
        )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot_surface(
        GRID_X,
        GRID_Y,
        partition,
        cmap="viridis",
    )

    ax.scatter(
        physical_eigen_x,
        physical_eigen_y,
        np.zeros_like(physical_eigen_x),
        s=25,
        label="Diagonalized function centers",
    )

    ax.set_title(
        "Partition-of-Unity Test\n"
        r"$A_{\rm cell}\sum_i \phi_i(x,y)/w_i$"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Value")

    ax.set_xlim(physical_x_min, physical_x_max)
    ax.set_ylim(physical_y_min, physical_y_max)

    ax.legend()

    plt.tight_layout()
    plt.show()

    return (
        partition,
        basis_integrals,
        cell_area,
    )


# ============================================================
# Run partition-of-unity diagnostic
# ============================================================

partition_of_unity, partition_basis_integrals, partition_cell_area = (
    plot_partition_of_unity()
)

