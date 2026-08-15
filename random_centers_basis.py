import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Parameters
# ============================================================

# Randomly distributed Gaussian function centers

num_centers = 25
random_seed = 7

# Spatial region in which the centers are placed
center_x_min = -2.5
center_x_max = 2.5
center_y_min = -2.5
center_y_max = 2.5

rng = np.random.default_rng(random_seed)

function_centers = np.column_stack((
    rng.uniform(center_x_min, center_x_max, num_centers),
    rng.uniform(center_y_min, center_y_max, num_centers),
))

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

print("Number of Gaussian basis functions:", n_basis)

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


# ============================================================
# Fine spatial grid
# ============================================================
#
# This grid is independent of the number and arrangement of
# Gaussian centers.
# ============================================================

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

target_x = (
    np.min(center_x)
    + np.max(center_x)
) / 2.0

target_y = (
    np.min(center_y)
    + np.max(center_y)
) / 2.0

distance_to_center = (
    (center_x - target_x)**2
    + (center_y - target_y)**2
)

center_basis_idx = np.argmin(
    distance_to_center
)

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

# Plot all Gaussian centers
plt.scatter(
    center_x,
    center_y,
    s=60,
    label="Gaussian centers",
)

# Highlight the currently selected basis function
plt.scatter(
    center_basis_x,
    center_basis_y,
    s=180,
    marker="*",
    label="Selected center",
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
# Plot sum of diagonalized basis functions along a random chain
# ============================================================

# Contract all diagonalized basis functions onto the spatial grid
contracted_eigenvectors = G_grid @ eigenvectors

# Build a short nearest-neighbor chain through the random centers.
# Start from the center closest to the geometric center, then repeatedly
# move to the nearest unused center.

chain_length = min(10, n_basis)

start_idx = np.argmin(
    (center_x - np.mean(center_x))**2
    + (center_y - np.mean(center_y))**2
)

random_chain_basis_indices = [start_idx]
unused = set(range(n_basis))
unused.remove(start_idx)

while len(random_chain_basis_indices) < chain_length and unused:
    current = random_chain_basis_indices[-1]

    next_idx = min(
        unused,
        key=lambda j: (
            (center_x[j] - center_x[current])**2
            + (center_y[j] - center_y[current])**2
        ),
    )

    random_chain_basis_indices.append(next_idx)
    unused.remove(next_idx)

random_chain_basis_indices = np.asarray(
    random_chain_basis_indices,
    dtype=int,
)

# Match each selected original Gaussian center to the simultaneously
# diagonalized function with the nearest (X, Y) eigenvalue pair.
random_chain_eigen_indices = []

for basis_idx in random_chain_basis_indices:
    x0 = center_x[basis_idx]
    y0 = center_y[basis_idx]

    distance = (
        (eigenvalues_x - x0)**2
        + (eigenvalues_y - y0)**2
    )

    random_chain_eigen_indices.append(np.argmin(distance))

random_chain_eigen_indices = np.asarray(
    random_chain_eigen_indices,
    dtype=int,
)

# Sum ONLY the diagonalized functions belonging to the selected chain
random_chain_sum = np.sum(
    contracted_eigenvectors[:, random_chain_eigen_indices],
    axis=1,
)

random_chain_sum = np.real_if_close(random_chain_sum)

random_chain_sum = random_chain_sum.reshape(
    n_grid_y,
    n_grid_x,
)

# Coordinates of the selected function centers
chain_x = center_x[random_chain_basis_indices]
chain_y = center_y[random_chain_basis_indices]

# Surface plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    GRID_X,
    GRID_Y,
    random_chain_sum,
    cmap="viridis",
    alpha=0.85,
)

# Plot selected random-center chain on the z = 0 plane
ax.plot(
    chain_x,
    chain_y,
    np.zeros_like(chain_x),
    "o-",
    linewidth=2,
    markersize=6,
    label="Nearest-neighbor chain",
)

ax.set_title(
    "Sum of Diagonalized Basis Functions Along Random-Center Chain"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Sum")

ax.legend()

plt.tight_layout()
plt.show()

