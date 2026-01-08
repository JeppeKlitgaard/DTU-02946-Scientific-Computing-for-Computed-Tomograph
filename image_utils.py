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
