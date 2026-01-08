import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage.draw import ellipse


# LLM
def discrete_imshow(data, ax=None, cmap_name="viridis", show_colorbar=True, max_colors=10):
    """
    Display an image with a discrete colormap based on unique values in the data.

    Parameters:
        data: 2D array with discrete values
        ax: matplotlib axis (optional, creates new figure if None)
        cmap_name: base colormap name to sample colors from
        show_colorbar: whether to show colorbar with tick labels
        max_colors: maximum number of discrete colors/ticks (bins data if exceeded)

    Returns:
        im: the imshow object
    """
    unique_vals = np.unique(data[~np.isnan(data) & ~np.isinf(data)])
    n_unique = len(unique_vals)

    # If too many unique values, bin the data
    if n_unique > max_colors:
        # Create evenly spaced bins
        bin_edges = np.linspace(unique_vals.min(), unique_vals.max(), max_colors + 1)
        tick_vals = (bin_edges[:-1] + bin_edges[1:]) / 2  # bin centers
        n_colors = max_colors
        boundaries = bin_edges
    else:
        tick_vals = unique_vals
        n_colors = n_unique
        # Create boundaries between unique values
        boundaries = np.concatenate([
            [unique_vals[0] - 0.5],
            (unique_vals[:-1] + unique_vals[1:]) / 2,
            [unique_vals[-1] + 0.5]
        ])

    # Sample colors from the base colormap
    base_cmap = plt.cm.get_cmap(cmap_name, n_colors)
    colors = [base_cmap(i) for i in range(n_colors)]
    discrete_cmap = ListedColormap(colors)

    norm = BoundaryNorm(boundaries, n_colors)

    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(data, cmap=discrete_cmap, norm=norm)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, ticks=tick_vals)
        cbar.ax.set_yticklabels([f"{v:.2g}" for v in tick_vals])

    return im

def make_ellipse(
    intensity: float = 1.0,
    axis_a: float = 0.5,
    axis_b: float = 0.5,
    center_x: float = 0.5,
    center_y: float = 0.5,
    theta: float = 0.0,  # in degrees
    shape: tuple = (251, 251),
):
    # Convert theta from degrees to radians for skimage.draw.ellipse
    rr, cc = ellipse(center_y * shape[1], center_x * shape[0], axis_a * shape[0], axis_b * shape[1], shape=shape, rotation=np.radians(theta))
    img = np.zeros(shape, dtype=np.float32)
    img[rr, cc] = intensity
    return img

def make_ellipse_thickness(
    thickness: float = 0.1,
    intensity: float = 1.0,
    axis_a: float = 0.5,
    axis_b: float = 0.5,
    center_x: float = 0.5,
    center_y: float = 0.5,
    theta: float = 0.0,  # in degrees
    shape: tuple = (251, 251),
):
    return make_ellipse(
        intensity=intensity,
        axis_a=axis_a,
        axis_b=axis_b,
        center_x=center_x,
        center_y=center_y,
        theta=theta,
        shape=shape,
    ) - make_ellipse(
        intensity=intensity,
        axis_a=max(0, axis_a - thickness),
        axis_b=max(0, axis_b - thickness),
        center_x=center_x,
        center_y=center_y,
        theta=theta,
        shape=shape,
    )

def make_rectangle(
    intensity: float = 1.0,
    width: float = 0.5,
    height: float = 0.5,
    center_x: float = 0.5,
    center_y: float = 0.5,
    shape: tuple = (251, 251),
):
    img = np.zeros(shape, dtype=np.float32)
    start_x = int((center_x - width / 2) * shape[0])
    end_x = int((center_x + width / 2) * shape[0])
    start_y = int((center_y - height / 2) * shape[1])
    end_y = int((center_y + height / 2) * shape[1])
    img[start_y:end_y, start_x:end_x] = intensity
    return img


# LLM
def smooth_zero_center(sinogram, sigma=10.0):
    """
    Smooth the transition from non-zero boundary values towards zero in the center.
    Only modifies pixels that are exactly zero - non-zero values are preserved.

    For each column, finds the upper and lower boundaries of the zero region,
    then applies Gaussian decay from the boundary values towards zero.

    Parameters:
        sinogram: 2D array with zero region in vertical center
        sigma: Gaussian decay parameter (larger = slower decay towards zero)

    Returns:
        Smoothed sinogram with filled center region (non-zero values unchanged)
    """
    result = sinogram.copy()
    n_rows, n_cols = sinogram.shape
    center = n_rows // 2

    for col in range(n_cols):
        column = sinogram[:, col]

        # Find upper boundary (last non-zero before center, searching from top)
        upper_boundary = None
        for i in range(center, -1, -1):
            if column[i] != 0:
                upper_boundary = i
                break

        # Find lower boundary (first non-zero after center, searching from bottom)
        lower_boundary = None
        for i in range(center, n_rows):
            if column[i] != 0:
                lower_boundary = i
                break

        # Skip if no zero region found
        if upper_boundary is None or lower_boundary is None:
            continue
        if upper_boundary >= lower_boundary - 1:
            continue

        # Get boundary values
        upper_val = column[upper_boundary]
        lower_val = column[lower_boundary]

        # Fill ONLY the zero region with Gaussian decay towards zero
        for i in range(upper_boundary + 1, lower_boundary):
            # Only modify if the original value is zero
            if sinogram[i, col] != 0:
                continue

            # Distance from each boundary
            dist_from_upper = i - upper_boundary
            dist_from_lower = lower_boundary - i

            # Gaussian decay towards zero from each boundary
            contribution_upper = upper_val * np.exp(-(dist_from_upper**2) / (2 * sigma**2))
            contribution_lower = lower_val * np.exp(-(dist_from_lower**2) / (2 * sigma**2))

            # Sum the contributions (both decay towards zero)
            result[i, col] = contribution_upper + contribution_lower

    return result


def make_rocket_phantom(N: int = 251, use_inf: bool = True):
    # Try to construct a phantom that would lead to exterior problem
    X = make_ellipse_thickness(intensity=1.0, thickness=0.05, axis_a=0.44, axis_b=0.3, center_x=0.5, center_y=0.5, shape=(N, N))

    X += make_ellipse(intensity=3.0, axis_a=0.08, axis_b=0.05, center_x=0.5, center_y=0.2, shape=(N, N))
    X += make_ellipse(intensity=5.0, axis_a=0.03, axis_b=0.03, center_x=0.40, center_y=0.22, shape=(N, N))
    X += make_ellipse(intensity=2.0, axis_a=0.03, axis_b=0.03, center_x=0.60, center_y=0.22, shape=(N, N))
    X += make_ellipse(intensity=4.0, axis_a=0.03, axis_b=0.03, center_x=0.40, center_y=0.33, shape=(N, N))

    # Impenetrable center circle
    INTENSITY_CENTER_SENTINEL = 100
    X += make_ellipse_thickness(thickness=0.025, intensity=INTENSITY_CENTER_SENTINEL, axis_a=0.15, axis_b=0.2, center_x=0.5, center_y=0.5, theta=30, shape=(N, N))

    if use_inf:
        X[X==INTENSITY_CENTER_SENTINEL] = np.inf

    # Bottom grill
    X += make_rectangle(intensity=2.5, width=0.03, height=0.1, center_x=0.4, center_y=0.78, shape=(N, N))
    X += make_rectangle(intensity=3.5, width=0.03, height=0.1, center_x=0.5, center_y=0.78, shape=(N, N))
    X += make_rectangle(intensity=2.5, width=0.03, height=0.1, center_x=0.6, center_y=0.7, shape=(N, N))

    # Center treasure
    X += make_ellipse(intensity=4.0, axis_a=0.1, axis_b=0.03, center_x=0.5, center_y=0.5, shape=(N, N))
    X += make_ellipse(intensity=4.0, axis_a=0.03, axis_b=0.1, center_x=0.5, center_y=0.5, shape=(N, N))

    return X

def make_simple_rocket_phantom(N: int = 251, use_inf: bool = True):
    # Try to construct a phantom that would lead to exterior problem
    X = make_ellipse_thickness(intensity=1.0, thickness=0.05, axis_a=0.44, axis_b=0.3, center_x=0.5, center_y=0.5, shape=(N, N))

    X += make_ellipse(intensity=3.0, axis_a=0.08, axis_b=0.06, center_x=0.5, center_y=0.2, shape=(N, N))
    X += make_ellipse(intensity=4.0, axis_a=0.05, axis_b=0.05, center_x=0.37, center_y=0.31, shape=(N, N))

    # Impenetrable center circle
    INTENSITY_CENTER_SENTINEL = 100
    X += make_ellipse_thickness(thickness=0.025, intensity=INTENSITY_CENTER_SENTINEL, axis_a=0.15, axis_b=0.2, center_x=0.5, center_y=0.5, theta=30, shape=(N, N))

    if use_inf:
        X[X==INTENSITY_CENTER_SENTINEL] = np.inf

    # Bottom grill
    X += make_rectangle(intensity=3.5, width=0.20, height=0.075, center_x=0.5, center_y=0.78, shape=(N, N))
    
    # Center treasure
    X += make_ellipse(intensity=4.0, axis_a=0.1, axis_b=0.03, center_x=0.5, center_y=0.5, shape=(N, N))
    X += make_ellipse(intensity=4.0, axis_a=0.03, axis_b=0.1, center_x=0.5, center_y=0.5, shape=(N, N))
    
    return X

def make_rocket_phantom2(N: int = 251, use_inf: bool = True):
    X = make_ellipse_thickness(intensity=1.0, thickness=0.05, axis_a=0.44, axis_b=0.3, center_x=0.5, center_y=0.5, shape=(N, N))

    # Top circles
    X += make_ellipse(intensity=3.0, axis_a=0.08, axis_b=0.05, center_x=0.5, center_y=0.2, shape=(N, N))
    X += make_ellipse(intensity=5.0, axis_a=0.03, axis_b=0.03, center_x=0.40, center_y=0.22, shape=(N, N))
    X += make_ellipse(intensity=2.0, axis_a=0.03, axis_b=0.03, center_x=0.60, center_y=0.22, shape=(N, N))
    X += make_ellipse(intensity=4.0, axis_a=0.03, axis_b=0.03, center_x=0.40, center_y=0.33, shape=(N, N))
    
    # Impenetrable center circle
    INTENSITY_CENTER_SENTINEL = 100
    DO_INTENSITY_CENTER_INF_CONVERSION = True
    X += make_ellipse_thickness(thickness=0.01, intensity=INTENSITY_CENTER_SENTINEL, axis_a=0.15, axis_b=0.2, center_x=0.5, center_y=0.5, theta=30, shape=(N, N))
    
    if DO_INTENSITY_CENTER_INF_CONVERSION:
        X[X==INTENSITY_CENTER_SENTINEL] = np.inf

    # Bottom sticks
    X += make_ellipse(intensity=3.0, axis_a=0.01, axis_b=0.06, center_x=0.5, center_y=0.78, shape=(N, N),theta=90)
    X += make_ellipse(intensity=3.0, axis_a=0.01, axis_b=0.06, center_x=0.59, center_y=0.65, shape=(N, N),theta=30)
    
    # Center treasure
    X += make_ellipse(intensity=4.0, axis_a=0.1, axis_b=0.03, center_x=0.5, center_y=0.5, shape=(N, N))
    X += make_ellipse(intensity=4.0, axis_a=0.03, axis_b=0.1, center_x=0.5, center_y=0.5, shape=(N, N))

    return X


################################################################################################################
# Plotting
################################################################################################################

def plot_radon_rays_skimage(
    img,
    sinogram,
    theta_deg,
    threshold=0.0,
    ax=None,
    linewidth=0.5,
    alpha=0.4,
    color='yellow',
    angle_stride=1,       # ← NEW: plot every k-th angle
    detector_stride=1     # ← NEW: plot every m-th detector row
):
    """
    Plot Radon projection rays on top of an image using scikit-image geometry.

    Parameters
    ----------
    img : (N, N) array
        The phantom or reconstructed slice.
    sinogram : (N_det, N_angles)
        Parallel-beam sinogram from skimage.transform.radon.
    theta_deg : array
        Angles in degrees (as passed to radon()).
    threshold : float
        Skip rays with intensity <= threshold.
    angle_stride : int
        Plot only every angle_stride-th projection angle.
    detector_stride : int
        Plot only every detector_stride-th detector pixel (ray offset).
    """
    N = img.shape[0]
    det_pix = np.arange(N)

    # Correct detector coordinate mapping (skimage scanner geometry)
    t_vals = det_pix - (N - 1) / 2.0  

    # Convert angle array to radians
    theta_rad = np.deg2rad(theta_deg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Center image in coordinate system [-N/2, N/2]
    extent = [-N/2, N/2, -N/2, N/2]
    ax.imshow(img, cmap='gray', extent=extent)#, origin='lower')

    # Loop angles with stride
    for j in range(0, len(theta_rad), angle_stride):
        th = theta_rad[j]
        ct, st = np.cos(th), np.sin(th)

        # Loop detector pixels with stride
        for i in range(0, N, detector_stride):
            I = sinogram[i, j]
            if I <= threshold:
                continue

            t = t_vals[i]

            # Point on the line
            x0 = ct * t
            y0 = st * t

            # Direction vector perpendicular to the normal
            dx, dy = -st, ct

            # Large extent so line crosses image
            L = N * 2

            x1, y1 = x0 + L*dx, y0 + L*dy
            x2, y2 = x0 - L*dx, y0 - L*dy

            ax.plot([x1, x2], [y1, y2],
                    color=color, lw=linewidth, alpha=alpha)

    ax.set_xlim([-N/2, N/2])
    ax.set_ylim([-N/2, N/2])
    ax.set_aspect('equal')
    ax.set_title(f"Rays (angle_stride={angle_stride}, detector_stride={detector_stride})")
    return ax
def compute_coverage_from_intensity(sinogram, theta_deg, img_shape, threshold=0.0, mask_mode='inscribed'):
    """
    Compute per-pixel angular coverage from intensity sinogram (blocked rays -> intensity <= threshold).

    Parameters
    ----------
    sinogram : (N_det, N_angles) array
        Intensity-domain sinogram I (e.g., from your code). Blocked rays should be 0 (or near 0).
    theta_deg : (N_angles,) array
        Angles in degrees (as used in skimage.transform.radon).
    img_shape : (H, W)
        Image size. Typically (N, N) for your phantom.
    threshold : float
        Rays with intensity <= threshold are treated as blocked.
    mask_mode : {'none', 'inscribed'}
        If 'inscribed', pixels outside the inscribed circle are ignored (NaN coverage).

    Returns
    -------
    coverage_count : (H, W) float array
        For each pixel: number of angles that see this pixel (unblocked).
    coverage_fraction : (H, W) float array
        coverage_count / len(theta_deg) (NaN outside mask if mask_mode='inscribed').
    missing_by_angle : (N_angles,) float array
        For each angle: fraction of pixels (inside mask) that are NOT seen.
    mask : (H, W) bool
        True for pixels considered (inside mask), False otherwise.
    """
    H, W = img_shape
    assert H == W, "This helper assumes a square image (H==W) consistent with N_det."
    N = H
    N_det, N_ang = sinogram.shape
    assert N_det == N, f"N_det ({N_det}) should match image size N ({N})."

    # Pixel-center coordinates consistent with Radon geometry & detector shift
    # Centers in [- (N-1)/2, ..., + (N-1)/2]
    coords = np.arange(N) - (N - 1) / 2.0
    Xc, Yc = np.meshgrid(coords, coords)  # Xc: columns, Yc: rows (origin at center)

    # Optional inscribed-circle mask (commonly used in Radon)
    if mask_mode == 'inscribed':
        R = (N - 1) / 2.0
        mask = (Xc**2 + Yc**2) <= (R + 1e-6)**2
    else:
        mask = np.ones((N, N), dtype=bool)

    theta_rad = np.deg2rad(theta_deg)
    coverage_count = np.zeros((N, N), dtype=np.float32)
    valid_pixels = mask.sum()

    missing_by_angle = np.zeros(N_ang, dtype=np.float32)

    # Iterate angles; for each pixel, check if the nearest detector bin is unblocked
    for j, th in enumerate(theta_rad):
        ct, st = np.cos(th), np.sin(th)
        # t for each pixel center
        t_pix = Xc * ct + Yc * st
        # nearest detector sample index
        i_det = np.rint(t_pix + (N - 1) / 2.0).astype(int)

        inside = (i_det >= 0) & (i_det < N) & mask
        if not np.any(inside):
            missing_by_angle[j] = 1.0  # nothing seen at this angle
            continue

        # Look up intensity for those pixels at this angle via nearest detector bin
        # sinogram[i_det, j] with i_det being a 2D array -> result is (N,N)
        I_here = np.zeros((N, N), dtype=sinogram.dtype)
        I_here[inside] = sinogram[i_det[inside], j]

        seen = (I_here > threshold) & inside
        coverage_count[seen] += 1.0

        # Angle-wise missing fraction over the mask
        missing_by_angle[j] = 1.0 - (seen.sum() / valid_pixels)

    # Coverage fraction in [0,1]; outside mask => NaN
    with np.errstate(invalid='ignore'):
        coverage_fraction = coverage_count / float(N_ang)
        coverage_fraction[~mask] = np.nan

    return coverage_count, coverage_fraction, missing_by_angle, mask


def plot_missing_information(img, sinogram, theta_deg, threshold=0.0,
                             mask_mode='inscribed', cmap='magma'):
    """
    Make a 3-panel figure:
      (1) the image,
      (2) coverage fraction map (0..1),
      (3) missing fraction per angle (line + polar).
    """
    N = img.shape[0]
    extent = [-N/2, N/2, -N/2, N/2]

    # Compute coverage from intensity sinogram
    cov_count, cov_frac, missing_by_angle, mask = compute_coverage_from_intensity(
        sinogram, theta_deg, img_shape=img.shape, threshold=threshold, mask_mode=mask_mode
    )

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1.2, 1.2, 1.2, 1.0], wspace=0.25, hspace=0.35)

    ax_img = fig.add_subplot(gs[:, 0])
    ax_cov = fig.add_subplot(gs[:, 1])
    ax_line = fig.add_subplot(gs[0, 2])
    ax_polar = fig.add_subplot(gs[1, 2], projection='polar')
    ax_colorbar = fig.add_subplot(gs[:, 3])

    # (1) Image for context
    ax_img.imshow(img, cmap='gray', extent=extent)#, origin='lower')
    ax_img.set_title('Image (context)')
    ax_img.set_aspect('equal')
    ax_img.set_xlim(extent[:2])
    ax_img.set_ylim(extent[2:])

    # (2) Coverage fraction map
    im = ax_cov.imshow(cov_frac, cmap=cmap, extent=extent, vmin=0.0, vmax=1.0, origin='lower')
    ax_cov.set_title('Coverage fraction per pixel\n(1.0 = seen from all angles)')
    ax_cov.set_aspect('equal')
    ax_cov.set_xlim(extent[:2])
    ax_cov.set_ylim(extent[2:])

    # Outline pixels that are NOT fully seen (cov < 1)
    try:
        import numpy as np
        levels = [1.0]
        cf = cov_frac.copy()
        # Matplotlib contour ignores NaNs automatically
        ax_cov.contour(cf, levels=levels, colors='white', linewidths=0.8, alpha=0.9,
                       extent=extent)
    except Exception:
        pass

    # Colorbar
    cbar = fig.colorbar(im, cax=ax_colorbar)
    cbar.set_label('Coverage fraction')

    # (3a) Missing fraction per angle (Cartesian)
    ax_line.plot(theta_deg, missing_by_angle, color='crimson', lw=1.5)
    ax_line.set_xlabel('Angle (degrees)')
    ax_line.set_ylabel('Missing fraction')
    ax_line.set_title('Missing fraction vs. angle')
    ax_line.set_xlim(theta_deg[0], theta_deg[-1] + (theta_deg[1]-theta_deg[0]))

    # (3b) Same in polar to show which directions are missing
    th_rad = np.deg2rad(theta_deg)
    ax_polar.plot(th_rad, missing_by_angle, color='crimson', lw=1.5)
    ax_polar.set_theta_zero_location('E')  # 0° to the right
    ax_polar.set_theta_direction(-1)       # increasing clockwise (Radon convention)
    ax_polar.set_title('Missing fraction (polar)', va='bottom')

    fig.suptitle(f"Information coverage (threshold={threshold}, mask_mode='{mask_mode}')", y=1.02, fontsize=12)
    plt.tight_layout()
    return {
        "coverage_count": cov_count,
        "coverage_fraction": cov_frac,
        "missing_by_angle": missing_by_angle,
        "mask": mask,
        "fig": fig
    }

def missing_angles_for_pixel(sinogram, theta_deg, pixel_rc, threshold=0.0):
    """
    Return a boolean array seen[j] indicating if pixel_rc is seen at angle j
    (i.e., the ray through that pixel has intensity > threshold).

    Parameters
    ----------
    sinogram : (N_det, N_angles) array
        Intensity-domain sinogram (blocked rays -> 0).
    theta_deg : (N_angles,) array
        Projection angles in degrees (as in skimage.radon).
    pixel_rc : (row, col)
        Pixel index coordinates in the image (0..N-1).
    threshold : float
        Intensity threshold; <= threshold is considered blocked.

    Returns
    -------
    seen : (N_angles,) bool array
        True if the pixel is seen (unblocked) at that angle.
    """
    N_det, N_ang = sinogram.shape
    r, c = pixel_rc
    assert 0 <= r < N_det and 0 <= c < N_det, "pixel indices must be within image bounds"
    # Pixel-centered coords in [- (N-1)/2, ..., + (N-1)/2]
    x = c - (N_det - 1) / 2.0
    y = r - (N_det - 1) / 2.0

    theta_rad = np.deg2rad(theta_deg)
    ct = np.cos(theta_rad)
    st = np.sin(theta_rad)

    # Signed distance for this pixel at each angle
    t_pix = x * ct + y * st
    # Nearest detector index
    i_det = np.rint(t_pix + (N_det - 1) / 2.0).astype(int)
    inside = (i_det >= 0) & (i_det < N_det)

    seen = np.zeros(N_ang, dtype=bool)
    seen[inside] = sinogram[i_det[inside], np.arange(N_ang)[inside]] > threshold
    return seen


# def plot_missing_angles_for_pixel(sinogram, theta_deg, pixel_rc, threshold=0.0,
#                                   ax_line=None, ax_polar=None, title_prefix='Pixel'):
#     """
#     Plot line + polar charts of missing angles for a pixel.
#     """
#     seen = missing_angles_for_pixel(sinogram, theta_deg, pixel_rc, threshold=threshold)
#     missing = ~seen
#     th_rad = np.deg2rad(theta_deg)

#     if ax_line is None or ax_polar is None:
#         fig = plt.figure(figsize=(12, 4))
#         ax_line = fig.add_subplot(1, 2, 1)
#         ax_polar = fig.add_subplot(1, 2, 2, projection='polar')

#     # Line plot (0/1 visibility)
#     ax_line.step(theta_deg, seen.astype(float), where='mid', color='tab:blue')
#     ax_line.set_ylim(-0.05, 1.05)
#     ax_line.set_xlabel('Angle (degrees)')
#     ax_line.set_ylabel('Seen = 1 / Missing = 0')
#     ax_line.set_title(f'{title_prefix} {pixel_rc} — seen/missing vs angle')

#     # Polar plot: highlight missing angles
#     ax_polar.plot(th_rad[seen], np.ones(seen.sum()), '.', color='tab:green', alpha=0.7, label='seen')
#     ax_polar.plot(th_rad[missing], np.ones(missing.sum()), 'x', color='crimson', alpha=0.8, label='missing')
#     ax_polar.set_theta_zero_location('E')  # 0° to the right
#     ax_polar.set_theta_direction(-1)       # increase clockwise
#     ax_polar.set_yticks([])
#     ax_polar.set_title('Missing angles (polar)')
#     ax_polar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
#     return seen

def visible_mask_for_angle(img_shape, sinogram, theta_deg, angle, threshold=0.0, mask_mode='inscribed'):
    """
    Compute visibility mask for a given angle.

    Parameters
    ----------
    img_shape : (N, N)
        Shape of the image.
    sinogram : (N, N_angles)
        Intensity sinogram.
    theta_deg : (N_angles,) array
        Angles in degrees.
    angle : float or int
        If float, interpreted in degrees and the nearest angle index is used.
        If int, directly used as angle index.
    threshold : float
        Intensity threshold.
    mask_mode : {'inscribed', 'none'}
        If 'inscribed', ignore pixels outside inscribed circle.

    Returns
    -------
    visible : (N, N) bool array
        True where pixel is seen at the given angle.
    j_idx : int
        Angle index used.
    """
    N = img_shape[0]
    assert img_shape[0] == img_shape[1] == sinogram.shape[0]
    N_ang = sinogram.shape[1]

    # Resolve angle index
    if isinstance(angle, (int, np.integer)):
        j_idx = int(angle)
    else:
        # nearest angle index
        j_idx = int(np.argmin(np.abs(theta_deg - angle)))
    assert 0 <= j_idx < N_ang

    coords = np.arange(N) - (N - 1) / 2.0
    Xc, Yc = np.meshgrid(coords, coords)  # centered pixel coordinates
    th = np.deg2rad(theta_deg[j_idx])
    ct, st = np.cos(th), np.sin(th)

    # t for each pixel center and nearest detector bin
    t_pix = Xc * ct + Yc * st
    i_det = np.rint(t_pix + (N - 1) / 2.0).astype(int)
    inside = (i_det >= 0) & (i_det < N)

    I_here = np.zeros((N, N), dtype=sinogram.dtype)
    I_here[inside] = sinogram[i_det[inside], j_idx]
    visible = (I_here > threshold) & inside

    if mask_mode == 'inscribed':
        R = (N - 1) / 2.0
        circle = (Xc**2 + Yc**2) <= (R + 1e-6)**2
        visible &= circle

    return visible, j_idx


def plot_visible_mask_for_angle(img, sinogram, theta_deg, angle, threshold=0.0,
                                mask_mode='inscribed', alpha=0.35, cmap='Greens', ax=None):
    """
    Overlay visibility mask for a given angle on the image.
    """
    N = img.shape[0]
    extent = [-N/2, N/2, -N/2, N/2]
    visible, j_idx = visible_mask_for_angle(img.shape, sinogram, theta_deg, angle, threshold, mask_mode)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(img, cmap='gray', extent=extent)
    ax.imshow(visible.astype(float), cmap=cmap, extent=extent, origin='lower', alpha=alpha, vmin=0, vmax=1)
    ax.set_aspect('equal')
    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
    ax.set_title(f'Visible pixels at angle {theta_deg[j_idx]:.2f}° (idx {j_idx})')
    return ax

# --- Core helper: is a pixel seen at each angle? ---
def missing_angles_for_pixel(sinogram, theta_deg, pixel_rc, threshold=0.0):
    """
    Return boolean array seen[j] indicating if pixel_rc is seen at angle j
    (ray intensity > threshold), using scikit-image geometry.
    """
    N_det, N_ang = sinogram.shape
    r, c = pixel_rc
    assert 0 <= r < N_det and 0 <= c < N_det, "pixel indices must be within image bounds"

    # Pixel-centered coords in [-(N-1)/2, ..., +(N-1)/2]
    x = c - (N_det - 1) / 2.0
    y = r - (N_det - 1) / 2.0

    theta_rad = np.deg2rad(theta_deg)
    ct = np.cos(theta_rad)
    st = np.sin(theta_rad)

    # Signed distance for this pixel at each angle
    t_pix = x * ct + y * st
    # Nearest detector index
    i_det = np.rint(t_pix + (N_det - 1) / 2.0).astype(int)
    inside = (i_det >= 0) & (i_det < N_det)

    seen = np.zeros(N_ang, dtype=bool)
    idx = np.arange(N_ang)
    seen[inside] = sinogram[i_det[inside], idx[inside]] > threshold
    return seen


# ---------- Core helpers ----------

def _line_segment_in_rect(ct, st, t, xmin, xmax, ymin, ymax, eps=1e-12):
    """
    Return the clipped segment endpoints for x*ct + y*st = t within the rectangle.
    """
    pts = []

    # Intersect with x = xmin and x = xmax
    if abs(st) > eps:
        y = (t - xmin * ct) / st
        if ymin - 1e-9 <= y <= ymax + 1e-9: pts.append((xmin, y))
        y = (t - xmax * ct) / st
        if ymin - 1e-9 <= y <= ymax + 1e-9: pts.append((xmax, y))

    # Intersect with y = ymin and y = ymax
    if abs(ct) > eps:
        x = (t - ymin * st) / ct
        if xmin - 1e-9 <= x <= xmax + 1e-9: pts.append((x, ymin))
        x = (t - ymax * st) / ct
        if xmin - 1e-9 <= x <= xmax + 1e-9: pts.append((x, ymax))

    # Deduplicate
    uniq = []
    for p in pts:
        if all(np.hypot(p[0]-q[0], p[1]-q[1]) > 1e-6 for q in uniq):
            uniq.append(p)

    if len(uniq) < 2:
        return None

    # Choose farthest two along line direction (-st, ct)
    v = np.array([-st, ct])
    proj = [p[0]*v[0] + p[1]*v[1] for p in uniq]
    i_min, i_max = int(np.argmin(proj)), int(np.argmax(proj))
    return uniq[i_min], uniq[i_max]


def _missing_angles_for_pixel(sinogram, theta_deg, pixel_rc, threshold=0.0):
    """
    For a chosen pixel (row, col) return boolean array seen[j] indicating if
    the ray through that pixel at angle j has intensity > threshold.
    Uses scikit-image Radon geometry (degrees; detector coordinate is centered indices).
    """
    N_det, N_ang = sinogram.shape
    r, c = pixel_rc
    assert 0 <= r < N_det and 0 <= c < N_det, "pixel indices must be within image bounds"

    # Pixel-centered coords in [-(N-1)/2, ..., +(N-1)/2]
    x = c - (N_det - 1) / 2.0
    y = r - (N_det - 1) / 2.0

    theta_rad = np.deg2rad(theta_deg)
    ct = np.cos(theta_rad)
    st = np.sin(theta_rad)

    # Signed distance for this pixel at each angle
    t_pix = x * ct + y * st
    # Nearest detector index
    i_det = np.rint(t_pix + (N_det - 1) / 2.0).astype(int)
    inside = (i_det >= 0) & (i_det < N_det)

    seen = np.zeros(N_ang, dtype=bool)
    idx = np.arange(N_ang)
    seen[inside] = sinogram[i_det[inside], idx[inside]] > threshold
    return seen


def _update_angle_plots(seen, theta_deg, ax_line, ax_polar, pixel_rc):
    """Update the line+polar panels to show seen/missing angles for one pixel."""
    missing = ~seen
    th_rad = np.deg2rad(theta_deg)

    # Line plot (0/1 visibility)
    ax_line.clear()
    ax_line.step(theta_deg, seen.astype(float), where='mid', color='tab:blue')
    ax_line.set_ylim(-0.05, 1.05)
    ax_line.set_xlabel('Angle (degrees)')
    ax_line.set_ylabel('Seen = 1 / Missing = 0')
    ax_line.set_title(f'Pixel {pixel_rc} — seen/missing vs angle')

    # Polar plot: highlight missing angles
    ax_polar.clear()
    ax_polar.plot(th_rad[seen], np.ones(seen.sum()), '.', color='tab:green', alpha=0.85, label='seen')
    ax_polar.plot(th_rad[missing], np.ones(missing.sum()), 'x', color='crimson', alpha=0.95, label='missing')
    ax_polar.set_theta_zero_location('E')  # 0° to the right
    ax_polar.set_theta_direction(-1)       # increase clockwise
    ax_polar.set_yticks([])
    ax_polar.set_title('Missing angles (polar)')
    ax_polar.legend(loc='upper right', fontsize=9, framealpha=0.7, bbox_to_anchor=(1.2, 1.05))


# ---------- Integrated interactive picker with ray overlay ----------
# def interactive_missing_angles_picker(
#     img,
#     sinogram,
#     theta_deg,
#     threshold=0.0,
#     mask_mode='inscribed',
#     marker_color='yellow',
#     figsize=(10, 5.5),
#     dpi=110,
#     plot_rays=True,
#     show_missing_rays=False,
#     angle_stride_overlay=5,
#     max_rays_overlay=120,
#     ray_seen_color='cyan',
#     ray_missing_color='crimson',
#     ray_linewidth=0.9,
#     ray_alpha=0.5,
# ):
#     """
#     Fully integrated interactive picker compatible with image_utils.py.

#     Provides:
#       - Click-to-select pixel
#       - Seen/missing angle analysis (line + polar)
#       - Ray overlay for selected pixel
#       - Layout safe for use inside modules

#     Works best with: %matplotlib widget
#     """

#     import numpy as np
#     import matplotlib.pyplot as plt
#     from matplotlib.gridspec import GridSpec

#     N = img.shape[0]
#     assert img.shape[0] == img.shape[1] == sinogram.shape[0], \
#         "Expected square image and sinogram with matching size."

#     # --- coordinate extents (centered) ---
#     extent = [-N/2, N/2, -N/2, N/2]
#     coords = np.arange(N) - (N - 1)/2
#     Xc, Yc = np.meshgrid(coords, coords)

#     # --- inscribed mask (optional) ---
#     if mask_mode == 'inscribed':
#         circle_mask = (Xc**2 + Yc**2) <= ((N-1)/2 + 1e-6)**2
#     else:
#         circle_mask = np.ones_like(img, bool)

#     # --- Figure layout (safe for modules) ---
#     plt.rcParams['figure.dpi'] = dpi
#     fig = plt.figure(figsize=figsize, constrained_layout=True)

#     gs = GridSpec(2, 3, figure=fig, width_ratios=[1.3, 1, 1])

#     ax_img  = fig.add_subplot(gs[:, 0])        # full left column
#     ax_line = fig.add_subplot(gs[0, 1])        # top-right
#     ax_polar= fig.add_subplot(gs[1, 1], projection='polar')  # bottom-right

#     # --- Try to expand to cell width (ipympl only) ---
#     try:
#         fig.canvas.layout.width = '100%'
#     except Exception:
#         pass

#     # ---------- helper: classify angles for selected pixel ----------
#     def missing_angles_for_pixel(pixel_rc):
#         r, c = pixel_rc
#         x = c - (N-1)/2
#         y = r - (N-1)/2
#         th = np.deg2rad(theta_deg)
#         ct, st = np.cos(th), np.sin(th)
#         t_pix = x*ct + y*st
#         i_det = np.rint(t_pix + (N-1)/2).astype(int)
#         inside = (i_det >= 0) & (i_det < N)
#         seen = np.zeros_like(th, dtype=bool)
#         idx = np.arange(len(theta_deg))
#         seen[inside] = sinogram[i_det[inside], idx[inside]] > threshold
#         return seen

#     # ---------- helper: clipped line segment ----------
#     def line_seg(ct, st, t):
#         xmin,xmax,ymin,ymax = extent[0], extent[1], extent[2], extent[3]
#         pts=[]
#         eps=1e-12
#         if abs(st)>eps:
#             y=(t - xmin*ct)/st
#             if ymin<=y<=ymax: pts.append((xmin,y))
#             y=(t - xmax*ct)/st
#             if ymin<=y<=ymax: pts.append((xmax,y))
#         if abs(ct)>eps:
#             x=(t - ymin*st)/ct
#             if xmin<=x<=xmax: pts.append((x,ymin))
#             x=(t - ymax*st)/ct
#             if xmin<=x<=xmax: pts.append((x,ymax))
#         if len(pts)<2: return None
#         uniq=[]
#         for p in pts:
#             if all(np.hypot(p[0]-q[0], p[1]-q[1])>1e-6 for q in uniq):
#                 uniq.append(p)
#         if len(uniq)<2: return None
#         v=np.array([-st,ct])
#         proj=[p[0]*v[0]+p[1]*v[1] for p in uniq]
#         return uniq[int(np.argmin(proj))], uniq[int(np.argmax(proj))]

#     # ---------- helper: plot seen/missing ----------
#     def update_angle_plots(seen, pixel_rc):
#         ax_line.clear()
#         ax_polar.clear()
#         th = np.deg2rad(theta_deg)
#         missing = ~seen

#         # line plot
#         ax_line.step(theta_deg, seen.astype(float), where='mid', color='tab:blue')
#         ax_line.set_ylim(-0.05, 1.05)
#         ax_line.set_title(f"Pixel {pixel_rc}")

#         # polar plot
#         ax_polar.plot(th[seen], np.ones(seen.sum()), '.', color='green')
#         ax_polar.plot(th[missing], np.ones(missing.sum()), 'x', color='red')
#         ax_polar.set_theta_zero_location('E')
#         ax_polar.set_theta_direction(-1)
#         ax_polar.set_yticks([])
#         ax_polar.set_title("Missing angles")

#     # ---------- helper: draw rays ----------
#     ray_artists=[]

#     def clear_rays():
#         for a in ray_artists:
#             try: a.remove()
#             except Exception: pass
#         ray_artists.clear()

#     th = np.deg2rad(theta_deg)
#     ct_all = np.cos(th)
#     st_all = np.sin(th)

#     def draw_rays(pixel_rc, seen_mask):
#         if not plot_rays: return
#         r,c = pixel_rc
#         x_pix = c - (N-1)/2
#         y_pix = r - (N-1)/2

#         idx = np.arange(len(theta_deg))[::angle_stride_overlay]
#         idx_seen = idx[seen_mask[idx]]
#         idx_miss = idx[~seen_mask[idx]] if show_missing_rays else []

#         # cap total drawn rays
#         if max_rays_overlay:
#             idx_seen = idx_seen[:max_rays_overlay]

#         # seen rays
#         for j in idx_seen:
#             ct,st = ct_all[j], st_all[j]
#             t = x_pix*ct + y_pix*st
#             seg = line_seg(ct,st,t)
#             if seg:
#                 (x1,y1),(x2,y2)=seg
#                 ln, = ax_img.plot([x1,x2], [y1,y2],
#                                   color=ray_seen_color,
#                                   lw=ray_linewidth,
#                                   alpha=ray_alpha)
#                 ray_artists.append(ln)

#     # ---------- plot base image ----------
#     ax_img.imshow(img, cmap='gray', extent=extent)#, origin='lower')
#     if mask_mode == 'inscribed':
#         ax_img.imshow((~circle_mask).astype(float),
#                       cmap='gray', extent=extent, origin='lower',
#                       alpha=0.1, vmin=0, vmax=1)
#     ax_img.set_aspect('equal')

#     # marker
#     marker, = ax_img.plot([], [], '+', color=marker_color, ms=12, mew=2)

#     # ---------- update on click ----------
#     def update_pixel(rc):
#         r,c = rc
#         if not (0 <= r < N and 0 <= c < N): return

#         # marker coords
#         x = c - (N-1)/2
#         y = r - (N-1)/2
#         marker.set_data([x],[y])

#         seen = missing_angles_for_pixel(rc)
#         update_angle_plots(seen, rc)

#         clear_rays()
#         draw_rays(rc, seen)

#         fig.canvas.draw_idle()

#     # initialize with center pixel
#     update_pixel((N//2, N//2))

#     def onclick(event):
#         if event.inaxes is not ax_img: return
#         if event.xdata is None: return
#         c = int(round(event.xdata + (N-1)/2))
#         r = int(round(event.ydata + (N-1)/2))
#         update_pixel((r,c))

#     fig.canvas.mpl_connect("button_press_event", onclick)
#     return fig


def interactive_missing_angles_picker(
    img,
    sinogram,
    theta_deg,
    threshold=0.0,
    mask_mode='inscribed',
    marker_color='yellow',
    figsize=(10, 5.5),
    dpi=110,
    plot_rays=True,
    show_missing_rays=False,
    angle_stride_overlay=5,
    max_rays_overlay=120,
    ray_seen_color='cyan',
    ray_missing_color='crimson',
    ray_linewidth=0.9,
    ray_alpha=0.5,
):
    """
    Interactive picker compatible with image_utils.py:
      • Click to select pixel
      • Top-right line plot: sinogram value along the ray through that pixel
        (NaN where blocked or outside detector range)
      • Bottom-right polar plot: seen/missing angles
      • Optional ray overlay through selected pixel

    Geometry: scikit-image Radon
      - theta_deg in degrees
      - detector coordinate uses centered indices: t = i - (N-1)/2
      - line: x cos θ + y sin θ = t

    If your 'sinogram' is intensity, the line plot shows intensity.
    If your 'sinogram' is attenuation, it shows attenuation (path integral).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    N = img.shape[0]
    assert img.shape[0] == img.shape[1] == sinogram.shape[0], \
        "Expected square image and sinogram with matching size."

    # --- coordinate extents (centered) ---
    extent = [-N/2, N/2, -N/2, N/2]
    coords = np.arange(N) - (N - 1)/2
    Xc, Yc = np.meshgrid(coords, coords)

    # --- inscribed mask (optional) ---
    if mask_mode == 'inscribed':
        circle_mask = (Xc**2 + Yc**2) <= ((N-1)/2 + 1e-6)**2
    else:
        circle_mask = np.ones_like(img, bool)

    # --- Figure layout (safe for modules) ---
    plt.rcParams['figure.dpi'] = dpi
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.3, 1, 1])

    ax_img  = fig.add_subplot(gs[:, 0])                     # full left column
    ax_line = fig.add_subplot(gs[0, 1])                     # top-right (now shows values)
    ax_polar= fig.add_subplot(gs[1, 1], projection='polar') # bottom-right (seen/miss)

    try:
        # If running in ipympl (%matplotlib widget), fill cell width
        fig.canvas.layout.width = '100%'
    except Exception:
        pass

    # ---------- helpers: geometry ----------
    th = np.deg2rad(theta_deg)
    ct_all = np.cos(th)
    st_all = np.sin(th)

    def pixel_center_rc_to_xy(rc):
        r, c = rc
        return c - (N - 1)/2.0, r - (N - 1)/2.0

    # ---------- helper: compute seen mask & per-angle value ----------
    def per_angle_value_and_seen(pixel_rc):
        """
        For each angle j:
          - Find detector row i_det near the ray that goes through pixel_rc.
          - Return val[j] = sinogram[i_det, j] (NaN if outside or blocked).
          - Return seen[j] boolean if intensity > threshold (or just val > threshold).
        """
        r, c = pixel_rc
        x_pix, y_pix = pixel_center_rc_to_xy(pixel_rc)

        # t for each angle at the pixel
        t_pix = x_pix * ct_all + y_pix * st_all

        # nearest detector index
        i_det = np.rint(t_pix + (N - 1) / 2.0).astype(int)
        inside = (i_det >= 0) & (i_det < N)

        val = np.full_like(th, np.nan, dtype=float)
        idx = np.arange(len(theta_deg))
        val[inside] = sinogram[i_det[inside], idx[inside]].astype(float)

        # seen mask based on threshold on raw values (intensity or attenuation)
        seen = np.zeros_like(th, dtype=bool)
        seen[inside] = val[inside] > threshold
        # Set blocked rays to NaN so the line plot shows gaps
        val[~seen] = 0 #np.nan
        return val, seen

    # ---------- helper: clipped line segment ----------
    def line_seg(ct, st, t):
        xmin,xmax,ymin,ymax = extent[0], extent[1], extent[2], extent[3]
        pts=[]
        eps=1e-12
        if abs(st)>eps:
            y=(t - xmin*ct)/st
            if ymin<=y<=ymax: pts.append((xmin,y))
            y=(t - xmax*ct)/st
            if ymin<=y<=ymax: pts.append((xmax,y))
        if abs(ct)>eps:
            x=(t - ymin*st)/ct
            if xmin<=x<=xmax: pts.append((x,ymin))
            x=(t - ymax*st)/ct
            if xmin<=x<=xmax: pts.append((x,ymax))
        # Deduplicate
        uniq=[]
        for p in pts:
            if all(np.hypot(p[0]-q[0], p[1]-q[1])>1e-6 for q in uniq):
                uniq.append(p)
        if len(uniq)<2: return None
        v=np.array([-st,ct])
        proj=[p[0]*v[0]+p[1]*v[1] for p in uniq]
        return uniq[int(np.argmin(proj))], uniq[int(np.argmax(proj))]

    # ---------- helper: update plots ----------
    def update_angle_plots_with_values(values, seen, pixel_rc):
        # Line plot of the actual sinogram value (NaN where blocked/missing)
        ax_line.clear()
        ax_line.plot(theta_deg, values, color='tab:blue', lw=1.5)
        ax_line.set_xlabel("Angle (deg)")
        ax_line.set_ylabel("Sinogram value\n(along ray through pixel)")
        ax_line.set_title(f"Pixel {pixel_rc} — per-angle value")
        ax_line.set_xlim(0,180)
        # Optional: show baseline at 0 (handy for intensity)
        ax_line.axhline(0.0, color='k', lw=0.6, alpha=0.4)

        # Polar plot: seen/missing (unchanged)
        ax_polar.clear()
        th_rad = np.deg2rad(theta_deg)
        ax_polar.plot(th_rad[seen],  np.ones(seen.sum()),  '.', color='green',  alpha=0.85, label='seen')
        ax_polar.plot(th_rad[~seen], np.ones((~seen).sum()), 'x', color='red', alpha=0.95, label='missing')
        ax_polar.set_theta_zero_location('E')
        ax_polar.set_theta_direction(-1)
        ax_polar.set_yticks([])
        ax_polar.set_title("Missing angles")
        ax_polar.legend(loc='upper right', fontsize=9, framealpha=0.7, bbox_to_anchor=(1.2, 1.05))

    # ---------- helper: rays overlay ----------
    ray_artists=[]

    def clear_rays():
        for a in ray_artists:
            try: a.remove()
            except Exception: pass
        ray_artists.clear()

    def draw_rays(pixel_rc, seen_mask):
        if not plot_rays: return
        r,c = pixel_rc
        x_pix, y_pix = pixel_center_rc_to_xy(pixel_rc)

        idx = np.arange(len(theta_deg))[::max(1, int(angle_stride_overlay))]
        idx_seen = idx[seen_mask[idx]]
        idx_miss = idx[~seen_mask[idx]] if show_missing_rays else []

        # cap total drawn rays
        if max_rays_overlay is not None:
            # simple cap: prioritize seen rays
            idx_seen = idx_seen[:max_rays_overlay]
            rem = max(0, max_rays_overlay - len(idx_seen))
            idx_miss = idx_miss[:rem]

        for j in idx_seen:
            ct,st = ct_all[j], st_all[j]
            t = x_pix*ct + y_pix*st
            seg = line_seg(ct,st,t)
            if seg:
                (x1,y1),(x2,y2)=seg
                ln, = ax_img.plot([x1,x2], [y1,y2],
                                  color=ray_seen_color,
                                  lw=ray_linewidth,
                                  alpha=ray_alpha)
                ray_artists.append(ln)

        for j in idx_miss:
            ct,st = ct_all[j], st_all[j]
            t = x_pix*ct + y_pix*st
            seg = line_seg(ct,st,t)
            if seg:
                (x1,y1),(x2,y2)=seg
                ln, = ax_img.plot([x1,x2], [y1,y2],
                                  color=ray_missing_color,
                                  lw=ray_linewidth,
                                  alpha=ray_alpha,
                                  linestyle='--')
                ray_artists.append(ln)

    # ---------- base image ----------
    ax_img.imshow(img, cmap='gray', extent=extent)#, origin='lower')
    if mask_mode == 'inscribed':
        ax_img.imshow((~circle_mask).astype(float),
                      cmap='gray', extent=extent,
                      alpha=0.10, vmin=0, vmax=1)
    ax_img.set_aspect('equal')
    ax_img.set_xlim(extent[:2]); ax_img.set_ylim(extent[2:])
    ax_img.set_title('Click to pick pixel (rays overlay)')

    # marker
    marker, = ax_img.plot([], [], '+', color=marker_color, ms=12, mew=2)

    # ---------- update on click ----------
    def update_pixel(rc):
        r,c = rc
        if not (0 <= r < N and 0 <= c < N): return

        # marker coords
        x, y = pixel_center_rc_to_xy(rc)
        marker.set_data([x],[y])

        # per-angle value (from sinogram) and seen mask
        values, seen = per_angle_value_and_seen(rc)
        update_angle_plots_with_values(values, seen, rc)

        clear_rays()
        draw_rays(rc, seen)

        fig.canvas.draw_idle()

    # initialize with center pixel
    update_pixel((N//2, N//2))

    def onclick(event):
        if event.inaxes is not ax_img: return
        if event.xdata is None: return
        c = int(round(event.xdata + (N-1)/2))
        r = int(round(event.ydata + (N-1)/2))
        update_pixel((r,c))

    fig.canvas.mpl_connect("button_press_event", onclick)
    return fig


>>>>>>> 1e0a019 (phantoms and fancy plotting)
