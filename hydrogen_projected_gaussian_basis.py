import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_genlaguerre


def hydrogen_radial_part(r, N, m_abs):
    """
    Unnormalized radial part of a 2D hydrogen bound state in atomic units.

    Principal-shell convention used here:
        N = 1, 2, 3, ...
        n_r + |m| = N - 1

    The 2D Coulomb energy is
        E_N = -1 / [2 (N - 1/2)^2].

    The overall normalization is intentionally omitted here because all
    hydrogen orbitals are orthonormalized numerically on the finite box.
    """
    n_r = N - 1 - m_abs
    if n_r < 0:
        raise ValueError("For shell N, |m| must satisfy |m| <= N - 1.")

    n_eff = N - 0.5
    rho = 2.0 * r / n_eff

    return (
        rho**m_abs
        * np.exp(-rho / 2.0)
        * eval_genlaguerre(n_r, 2 * m_abs, rho)
    )


def raw_2d_hydrogen_orbitals(x, y, n_principal):
    """
    Return all real 2D hydrogen orbitals through principal shell n_principal.

    For each shell N:
        m_abs = 0, ..., N-1
        n_r = N - 1 - m_abs

    m_abs = 0 gives one orbital.
    m_abs > 0 gives cosine and sine real combinations.

    Therefore shell N contains 2N-1 orbitals, and shells 1...n contain n^2
    orbitals in total.

    Parameters
    ----------
    x, y : 1D arrays of identical length
        Spatial coordinates relative to the hydrogen center.
    n_principal : int
        Highest principal shell N to include.

    Returns
    -------
    H : ndarray, shape (n_points, n_principal**2)
        Raw, not-yet-normalized orbitals.
    labels : list[str]
        Human-readable orbital labels.
    """
    if int(n_principal) != n_principal or n_principal < 1:
        raise ValueError("n_principal must be a positive integer.")

    n_principal = int(n_principal)

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    orbitals = []
    labels = []

    for N in range(1, n_principal + 1):
        for m_abs in range(0, N):
            n_r = N - 1 - m_abs
            radial = hydrogen_radial_part(r, N, m_abs)

            if m_abs == 0:
                orbitals.append(radial)
                labels.append(f"N={N}, nr={n_r}, m=0")
            else:
                # Real combinations of the +/-m complex pair.
                orbitals.append(np.sqrt(2.0) * radial * np.cos(m_abs * theta))
                labels.append(f"N={N}, nr={n_r}, |m|={m_abs}, cos")

                orbitals.append(np.sqrt(2.0) * radial * np.sin(m_abs * theta))
                labels.append(f"N={N}, nr={n_r}, |m|={m_abs}, sin")

    return np.column_stack(orbitals), labels


def make_radial_gaussian_centers(
    radial_spacing=1.25,
    n_rings=6,
    include_center=True,
):
    """
    Build approximately uniformly spaced Gaussian centers on concentric rings.

    The number of centers on each ring is chosen from circumference / spacing,
    so neighboring centers have roughly the same separation radially and
    azimuthally.
    """
    centers = []

    if include_center:
        centers.append((0.0, 0.0))

    for ring in range(1, n_rings + 1):
        radius = ring * radial_spacing
        n_theta = max(
            6,
            int(np.round(2.0 * np.pi * radius / radial_spacing)),
        )

        # Alternate a half-step angular offset between rings to avoid putting
        # every radial line on top of the previous one.
        offset = 0.5 * (ring % 2) * (2.0 * np.pi / n_theta)

        angles = (
            np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
            + offset
        )

        for theta in angles:
            centers.append(
                (
                    radius * np.cos(theta),
                    radius * np.sin(theta),
                )
            )

    return np.asarray(centers, dtype=float)


def simultaneous_position_diagonalization(A_x, A_y):
    """
    Simultaneously diagonalize commuting X/Y operators by diagonalizing a
    generic irrational linear combination.
    """
    alpha = np.sqrt(2.0)
    A_combined = A_x + alpha * A_y

    _, eigenvectors = np.linalg.eig(A_combined)

    n_basis = eigenvectors.shape[1]
    eigenvalues_x = np.zeros(n_basis)
    eigenvalues_y = np.zeros(n_basis)

    for i in range(n_basis):
        v = eigenvectors[:, i]
        denom = np.vdot(v, v)

        eigenvalues_x[i] = np.real_if_close(
            np.vdot(v, A_x @ v) / denom
        ).real

        eigenvalues_y[i] = np.real_if_close(
            np.vdot(v, A_y @ v) / denom
        ).real

    # Sort primarily by y, then x, matching the convention used previously.
    sort_idx = np.lexsort((eigenvalues_x, eigenvalues_y))

    return (
        eigenvalues_x[sort_idx],
        eigenvalues_y[sort_idx],
        eigenvectors[:, sort_idx],
    )


def run_hydrogen_projected_gaussians(
    n_principal,
    box_size=20.0,
    n_grid_x=180,
    n_grid_y=180,
    gaussian_width=1.0,
    radial_spacing=1.25,
    n_rings=6,
    plot_padding=1.0,
):
    """
    Construct a radial Gaussian basis, project it onto the subspace orthogonal
    to all 2D hydrogen orbitals through principal shell n_principal, and then
    perform the same X/Y simultaneous diagonalization used in the earlier
    Gaussian-basis code.

    Returns a dictionary containing the important matrices and arrays.
    """

    # ============================================================
    # Spatial box
    # ============================================================

    half_box = box_size / 2.0

    x_grid = np.linspace(-half_box, half_box, n_grid_x)
    y_grid = np.linspace(-half_box, half_box, n_grid_y)

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
    # Hydrogen orbitals centered in the middle of the box
    # ============================================================

    H_raw_grid, hydrogen_labels = raw_2d_hydrogen_orbitals(
        grid_x,
        grid_y,
        n_principal,
    )

    n_hydrogen = H_raw_grid.shape[1]

    # Weighted QR orthonormalization on the finite spatial box:
    #
    #     H^T H dA = I
    #
    # If sqrt(dA) H_raw = Q R, then
    #
    #     H_ortho = H_raw R^-1
    #
    weighted_H = np.sqrt(dA) * H_raw_grid
    Q, R = np.linalg.qr(weighted_H, mode="reduced")

    H_grid = Q / np.sqrt(dA)

    hydrogen_gram = H_grid.T @ H_grid * dA

    # ============================================================
    # Radial Gaussian centers
    # ============================================================

    function_centers = make_radial_gaussian_centers(
        radial_spacing=radial_spacing,
        n_rings=n_rings,
        include_center=True,
    )

    center_x = function_centers[:, 0]
    center_y = function_centers[:, 1]

    n_basis = len(function_centers)

    if np.max(np.sqrt(center_x**2 + center_y**2)) > half_box - plot_padding:
        raise ValueError(
            "The radial Gaussian grid reaches too close to or beyond the box "
            "boundary. Increase box_size or reduce n_rings/radial_spacing."
        )

    if n_basis <= n_hydrogen:
        raise ValueError(
            f"The Gaussian basis has {n_basis} functions but the hydrogen "
            f"subspace contains {n_hydrogen} orbitals. Increase n_rings or "
            "reduce n_principal."
        )

    # ============================================================
    # Raw Gaussians on fine spatial grid
    # ============================================================

    grid_delta_x = grid_x[:, None] - center_x[None, :]
    grid_delta_y = grid_y[:, None] - center_y[None, :]

    G_grid = np.exp(
        -(
            grid_delta_x**2
            + grid_delta_y**2
        )
        / gaussian_width**2
    )

    # ============================================================
    # Hydrogen orbitals evaluated at Gaussian centers
    # ============================================================
    #
    # Apply the SAME QR basis transformation used on the fine grid,
    # so these are the same orthonormalized hydrogen functions
    # evaluated at the centers.
    # ============================================================

    H_raw_centers, _ = raw_2d_hydrogen_orbitals(
        center_x,
        center_y,
        n_principal,
    )

    H_centers = np.linalg.solve(
        R.T,
        H_raw_centers.T,
    ).T

    # ============================================================
    # Project each Gaussian orthogonal to the hydrogen subspace
    # ============================================================
    #
    # H_grid is orthonormal under the finite-grid inner product.
    #
    # Projection coefficients:
    #
    #     C = H^T G dA
    #
    # Projected Gaussians:
    #
    #     G_perp = G - H C
    # ============================================================

    hydrogen_overlap_coeffs = (
        H_grid.T @ G_grid * dA
    )

    G_perp_grid = (
        G_grid
        - H_grid @ hydrogen_overlap_coeffs
    )

    G_perp_centers = (
        np.exp(
            -(
                (center_x[:, None] - center_x[None, :])**2
                + (center_y[:, None] - center_y[None, :])**2
            )
            / gaussian_width**2
        )
        - H_centers @ hydrogen_overlap_coeffs
    )

    # Verify hydrogen orthogonality.
    hydrogen_projected_overlap = (
        H_grid.T @ G_perp_grid * dA
    )

    # ============================================================
    # Original X and Y operators in center-label space
    # ============================================================

    X = np.diag(center_x)
    Y = np.diag(center_y)

    # ============================================================
    # Similarity transformation using projected Gaussian basis
    # ============================================================

    condition_number = np.linalg.cond(G_perp_centers)

    if not np.isfinite(condition_number) or condition_number > 1e12:
        raise np.linalg.LinAlgError(
            "The projected Gaussian center matrix is singular or very poorly "
            f"conditioned (condition number = {condition_number:.3e}). "
            "Try reducing gaussian_width, changing radial_spacing, increasing "
            "the box, or using more Gaussian rings."
        )

    A_x = np.linalg.solve(
        G_perp_centers,
        X @ G_perp_centers,
    )

    A_y = np.linalg.solve(
        G_perp_centers,
        Y @ G_perp_centers,
    )

    commutator = A_x @ A_y - A_y @ A_x

    # ============================================================
    # Simultaneous X/Y diagonalization
    # ============================================================

    eigenvalues_x, eigenvalues_y, eigenvectors = (
        simultaneous_position_diagonalization(
            A_x,
            A_y,
        )
    )

    # Use complex dtype so arbitrary phase normalization below is safe
    # even when np.linalg.eig happened to return a real-valued array.
    eigenvectors = eigenvectors.astype(complex)

    # ============================================================
    # Contract the diagonalized coefficient vectors over the
    # projected Gaussians
    # ============================================================

    # Remove arbitrary sign/phase based on the largest-magnitude
    # point of each contracted function.
    for i in range(n_basis):
        f = G_perp_grid @ eigenvectors[:, i]
        max_idx = np.argmax(np.abs(f))
        phase = np.angle(f[max_idx])

        eigenvectors[:, i] *= np.exp(-1j * phase)

    contracted_eigenvectors = (
        G_perp_grid @ eigenvectors
    )

    # ============================================================
    # Gram matrices
    # ============================================================

    gram_raw_gaussians = (
        G_grid.conj().T @ G_grid * dA
    )

    gram_projected_gaussians = (
        G_perp_grid.conj().T @ G_perp_grid * dA
    )

    gram_contracted = (
        contracted_eigenvectors.conj().T
        @ contracted_eigenvectors
        * dA
    )

    # ============================================================
    # Simultaneous diagonalization checks
    # ============================================================

    V_inv = np.linalg.inv(eigenvectors)

    D_x = V_inv @ A_x @ eigenvectors
    D_y = V_inv @ A_y @ eigenvectors

    offdiag_x = D_x - np.diag(np.diag(D_x))
    offdiag_y = D_y - np.diag(np.diag(D_y))

    # ============================================================
    # Print all diagnostics before any plots
    # ============================================================

    np.set_printoptions(
        precision=4,
        suppress=True,
        linewidth=200,
    )

    print("=" * 80)
    print("2D HYDROGEN + PROJECTED GAUSSIAN BASIS")
    print("=" * 80)
    print("Highest principal shell:", n_principal)
    print("Number of hydrogen orbitals:", n_hydrogen)
    print("Number of projected Gaussian basis functions:", n_basis)
    print("Condition number of projected center matrix:", condition_number)

    print("\nHydrogen orbitals:")
    for i, label in enumerate(hydrogen_labels):
        print(f"{i:4d}: {label}")

    print("\n" + "=" * 80)
    print("HYDROGEN ORBITAL GRAM MATRIX")
    print("=" * 80)
    print(hydrogen_gram)

    print("\n" + "=" * 80)
    print("MAX OVERLAP OF PROJECTED GAUSSIANS WITH HYDROGEN SUBSPACE")
    print("=" * 80)
    print(np.max(np.abs(hydrogen_projected_overlap)))

    print("\n" + "=" * 80)
    print("COMMUTATOR CHECK")
    print("=" * 80)
    print("||[A_x, A_y]|| =", np.linalg.norm(commutator))

    print("\n" + "=" * 80)
    print("SORTED X EIGENVALUES")
    print("=" * 80)
    print(eigenvalues_x)

    print("\n" + "=" * 80)
    print("SORTED Y EIGENVALUES")
    print("=" * 80)
    print(eigenvalues_y)

    print("\n" + "=" * 80)
    print("GRAM MATRIX OF RAW GAUSSIANS")
    print("=" * 80)
    print(gram_raw_gaussians)

    print("\n" + "=" * 80)
    print("GRAM MATRIX OF HYDROGEN-PROJECTED GAUSSIANS")
    print("=" * 80)
    print(gram_projected_gaussians)

    print("\n" + "=" * 80)
    print("GRAM MATRIX OF CONTRACTED DIAGONALIZED FUNCTIONS")
    print("=" * 80)
    print(gram_contracted)

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
    # Select Gaussian closest to box center
    # ============================================================

    center_basis_idx = np.argmin(
        center_x**2 + center_y**2
    )

    center_basis_x = center_x[center_basis_idx]
    center_basis_y = center_y[center_basis_idx]

    # Match its center to the nearest recovered simultaneous
    # (X,Y) eigenvalue pair.
    distance_to_eigenvalue = (
        (eigenvalues_x - center_basis_x)**2
        + (eigenvalues_y - center_basis_y)**2
    )

    center_eigen_idx = np.argmin(
        distance_to_eigenvalue
    )

    # ============================================================
    # Plot sum of RAW radial Gaussians
    # ============================================================

    raw_gaussian_sum = np.sum(
        G_grid,
        axis=1,
    ).reshape(
        n_grid_y,
        n_grid_x,
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        GRID_X,
        GRID_Y,
        raw_gaussian_sum,
        cmap="viridis",
    )

    ax.scatter(
        center_x,
        center_y,
        np.zeros(n_basis),
        s=20,
    )

    ax.set_title("Sum of Raw Radial Gaussian Basis Functions")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Amplitude")

    plt.tight_layout()
    plt.show()

    # ============================================================
    # Plot sum of HYDROGEN-PROJECTED Gaussians
    # ============================================================

    projected_gaussian_sum = np.real_if_close(
        np.sum(
            G_perp_grid,
            axis=1,
        )
    ).reshape(
        n_grid_y,
        n_grid_x,
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        GRID_X,
        GRID_Y,
        projected_gaussian_sum,
        cmap="viridis",
    )

    ax.scatter(
        center_x,
        center_y,
        np.zeros(n_basis),
        s=20,
    )

    ax.set_title("Sum of Gaussians After Hydrogen Projection")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Amplitude")

    plt.tight_layout()
    plt.show()

    # ============================================================
    # Plot selected projected basis function
    # ============================================================

    center_projected_gaussian = np.real_if_close(
        G_perp_grid[:, center_basis_idx]
    ).reshape(
        n_grid_y,
        n_grid_x,
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        GRID_X,
        GRID_Y,
        center_projected_gaussian,
        cmap="Oranges",
    )

    ax.set_title(
        "Selected Gaussian After Projection Against Hydrogen Orbitals\n"
        f"Original center = ({center_basis_x:.4f}, {center_basis_y:.4f})"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Amplitude")

    plt.tight_layout()
    plt.show()

    # ============================================================
    # Plot radial Gaussian centers in the X-Y plane
    # ============================================================

    plt.figure(figsize=(8, 8))

    plt.scatter(
        center_x,
        center_y,
        s=40,
        label="Gaussian centers",
    )

    plt.scatter(
        center_basis_x,
        center_basis_y,
        s=180,
        marker="*",
        label="Selected center",
    )

    plt.title("Radial Gaussian Function Centers")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ============================================================
    # Plot corresponding simultaneously diagonalized function
    # ============================================================

    center_contracted = np.real_if_close(
        contracted_eigenvectors[:, center_eigen_idx]
    ).reshape(
        n_grid_y,
        n_grid_x,
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        GRID_X,
        GRID_Y,
        center_contracted,
        cmap="viridis",
    )

    ax.set_title(
        "Simultaneously Diagonalized Projected Function\n"
        f"X eigenvalue = {eigenvalues_x[center_eigen_idx]:.4f}, "
        f"Y eigenvalue = {eigenvalues_y[center_eigen_idx]:.4f}"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Amplitude")

    plt.tight_layout()
    plt.show()

    # ============================================================
    # Plot sum of all diagonalized projected basis functions
    # ============================================================

    diagonalized_sum = np.real_if_close(
        np.sum(
            contracted_eigenvectors,
            axis=1,
        )
    ).reshape(
        n_grid_y,
        n_grid_x,
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        GRID_X,
        GRID_Y,
        diagonalized_sum,
        cmap="viridis",
        alpha=0.85,
    )

    ax.scatter(
        center_x,
        center_y,
        np.zeros_like(center_x),
        s=25,
        label="Function centers",
    )

    ax.set_title(
        "Sum of Diagonalized Hydrogen-Orthogonal Basis Functions"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Sum")
    ax.legend()

    plt.tight_layout()
    plt.show()

    return {
        "function_centers": function_centers,
        "hydrogen_labels": hydrogen_labels,
        "H_grid": H_grid,
        "G_grid": G_grid,
        "G_perp_grid": G_perp_grid,
        "G_perp_centers": G_perp_centers,
        "X": X,
        "Y": Y,
        "A_x": A_x,
        "A_y": A_y,
        "eigenvalues_x": eigenvalues_x,
        "eigenvalues_y": eigenvalues_y,
        "eigenvectors": eigenvectors,
        "contracted_eigenvectors": contracted_eigenvectors,
        "gram_raw_gaussians": gram_raw_gaussians,
        "gram_projected_gaussians": gram_projected_gaussians,
        "gram_contracted": gram_contracted,
        "hydrogen_projected_overlap": hydrogen_projected_overlap,
        "GRID_X": GRID_X,
        "GRID_Y": GRID_Y,
    }


if __name__ == "__main__":
    # Example:
    #
    # N=2 includes all 2D hydrogen states in shells N=1 and N=2:
    # 1 state from N=1 and 3 states from N=2, for 4 total.
    #
    results = run_hydrogen_projected_gaussians(
        n_principal=2,
        box_size=20.0,
        n_grid_x=180,
        n_grid_y=180,
        gaussian_width=1.0,
        radial_spacing=1.25,
        n_rings=6,
    )
