import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial.transform import Rotation as R
import nibabel as nib

def main(degrees=0):
    yangle = degrees
    # Create a 3D array for the image (256x256x72 dimensions)
    image_shape = (256, 256, 72)
    
    # Set parameters
    resolution = 1.  # mm per voxel
    area = 20  # mm^2
    radius = np.sqrt(area / np.pi)
    radius_voxels = int(radius / resolution)
    
    # Cylinder length parameters
    length_straight = 60  # mm
    length_angled = 40  # mm
    total_length = length_straight + length_angled  # mm
    
    # Convert from mm to voxels
    length_straight_voxels = int(length_straight / resolution)
    length_angled_voxels = int(length_angled / resolution)
    
    # Center of the image
    center = (image_shape[0]//2, image_shape[1]//2, image_shape[2]//2)
    
    # Create cylinder using distance-based method (more efficient)
    img = create_cylinder(
        image_shape=image_shape,
        center=center,
        radius_voxels=radius_voxels,
        length_straight_voxels=length_straight_voxels,
        length_angled_voxels=length_angled_voxels
    )
    
    # Apply rotation using inverse mapping (eliminates gaps)
    rot_angles = [np.radians(10), np.radians(10), np.radians(yangle)]  # x, y, z in radians
    rotated_img = rotate_volume_inverse(img, center, rot_angles)
    
    # Visualize results
    #visualize_cylinder(rotated_img, center, image_shape, 
    #                  radius, radius_voxels, resolution, 
    #                  total_length, length_straight, length_angled,
    #                  rot_angles)
    
    # Save the result as a NIfTI file
    nifti_img = nib.Nifti1Image(rotated_img, np.eye(4))
    nib.save(nifti_img, f'straight_cylinder_rad-{yangle}.nii.gz')

    return nifti_img

def create_cylinder(image_shape, center, radius_voxels, 
                    length_straight_voxels, length_angled_voxels):
    """
    Create a cylinder with a straight segment followed by an angled segment.
    Uses a parametric approach for a more accurate and continuous cylinder.
    """
    # Initialize empty volume
    img = np.zeros(image_shape, dtype=np.uint8)

    # Center of the image
    center = (image_shape[0]//2, image_shape[1]//2, image_shape[2]//2)
    
    # Create first segment: along y-axis
    for y in range(center[1] - length_straight_voxels//2, center[1] + length_straight_voxels//2):
        for x in range(center[0] - radius_voxels, center[0] + radius_voxels):
            for z in range(center[2] - radius_voxels, center[2] + radius_voxels):
                # Check if point is within cylinder radius
                if (x - center[0])**2 + (z - center[2])**2 <= radius_voxels**2:
                    img[x, y, z] = 255
    
    
    # Define the angled segment (45 degrees in xy-plane)
    # Second segment: 45 degrees angle in the xy plane
    # Starting from the end of the first segment
    y_start = center[1] + length_straight_voxels//2

    # Calculate step sizes for 45-degree angle
    # For a 45-degree angle, we move equally in both x and y directions
    step_count = length_angled_voxels  # Number of steps to take
    dx = step_count / np.sqrt(2)  # x component
    dy = step_count / np.sqrt(2)  # y component

    # Convert to integers for indexing
    dx_voxels = int(dx)
    dy_voxels = int(dy)

    # For each step along the angled path
    for i in range(step_count):
        # Calculate position along the angled path
        # Linear interpolation from start to end
        progress = i / step_count
        x_pos = int(center[0] + progress * dx_voxels)
        y_pos = int(y_start + progress * dy_voxels)
        
        # Draw a circle at this position (perpendicular to the path)
        for x in range(x_pos - radius_voxels, x_pos + radius_voxels):
            for y in range(y_pos - radius_voxels, y_pos + radius_voxels):
                for z in range(center[2] - radius_voxels, center[2] + radius_voxels):
                    # Calculate distance from center of cylinder at this point
                    # For simplicity, we're using a basic distance calculation
                    # A more accurate approach would project the point onto the cylinder axis
                    dx_local = x - x_pos
                    dy_local = y - y_pos
                    
                    # Project point onto plane perpendicular to cylinder axis
                    # At 45 degrees, the axis direction is (1,1,0) normalized
                    axis = np.array([1, 1, 0]) / np.sqrt(2)
                    point = np.array([dx_local, dy_local, 0])
                    
                    # Projection formula: point - (point·axis)*axis
                    projection = point - np.dot(point, axis) * axis
                    
                    # Calculate distance from axis
                    distance = np.linalg.norm(projection)
                    
                    # Check if within radius
                    if distance <= radius_voxels and 0 <= x < image_shape[0] and 0 <= y < image_shape[1] and 0 <= z < image_shape[2]:
                        img[x, y, z] = 255
    
    return img


def rotate_volume_inverse(volume, center, angles):
    """
    Rotate a volume using inverse mapping to avoid gaps.
    
    Args:
        volume: Input 3D volume
        center: Center point of rotation (x, y, z)
        angles: Rotation angles in radians [x, y, z]
        
    Returns:
        Rotated volume
    """
    # Create rotation matrix
    rotation = R.from_euler('xyz', angles)
    rot_matrix = rotation.as_matrix()
    
    # Get volume dimensions
    h, w, d = volume.shape
    
    # Create output volume
    rotated = np.zeros_like(volume)
    
    # Create coordinate grids
    x_grid, y_grid, z_grid = np.meshgrid(
        np.arange(h),
        np.arange(w),
        np.arange(d),
        indexing='ij'
    )
    
    # Center coordinates
    x_centered = x_grid - center[0]
    y_centered = y_grid - center[1]
    z_centered = z_grid - center[2]
    
    # Stack coordinates for vectorized rotation
    coords = np.stack([x_centered, y_centered, z_centered], axis=-1)
    
    # Apply inverse rotation (transpose of rotation matrix)
    inv_rot_matrix = rot_matrix.T
    
    # Reshape for matrix multiplication
    orig_shape = coords.shape[:-1]
    coords_flat = coords.reshape(-1, 3)
    
    # Apply inverse rotation
    rotated_coords_flat = np.dot(coords_flat, inv_rot_matrix)
    
    # Reshape back to original shape
    rotated_coords = rotated_coords_flat.reshape(orig_shape + (3,))
    
    # Add center offset back
    src_x = np.round(rotated_coords[..., 0] + center[0]).astype(int)
    src_y = np.round(rotated_coords[..., 1] + center[1]).astype(int)
    src_z = np.round(rotated_coords[..., 2] + center[2]).astype(int)
    
    # Create mask for valid coordinates
    valid_mask = (
        (src_x >= 0) & (src_x < h) &
        (src_y >= 0) & (src_y < w) &
        (src_z >= 0) & (src_z < d)
    )
    
    # Copy values from source to rotated volume
    rotated[x_grid[valid_mask], y_grid[valid_mask], z_grid[valid_mask]] = \
        volume[src_x[valid_mask], src_y[valid_mask], src_z[valid_mask]]
    
    return rotated


def visualize_cylinder(img, center, image_shape, radius, radius_voxels, resolution, 
                      total_length, length_straight, length_angled, rot_angles):
    """
    Visualize the cylinder in 2D and 3D
    """
    # 1. Visualize a slice of the 3D image
    plt.figure(figsize=(10, 10))
    plt.imshow(img[:, :, center[2]], cmap='gray')
    plt.title('Middle Slice of Angled Cylinder (XY plane)')
    plt.axis('equal')
    plt.colorbar()
    plt.savefig('angled_cylinder_slice.png', dpi=300)

    # 2. Create 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Get the coordinates of non-zero voxels
    x, y, z = np.where(img > 0)

    # Plot only a subset of points to make the visualization clearer
    stride = 10  # Plot every 10th point
    ax.scatter(x[::stride], y[::stride], z[::stride], c='b', marker='.', alpha=0.5)

    # Create second plot for the outline
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')

    # Find edge points (using gradient-based method)
    from scipy import ndimage
    edges = np.zeros_like(img)
    
    # Use gradient magnitude to find edges
    for axis in range(3):
        gradient = ndimage.sobel(img, axis=axis).astype(float)
        edges = np.maximum(edges, np.abs(gradient))
    
    edge_threshold = 0  # Any non-zero value is an edge
    edge_mask = (edges > edge_threshold) & (img > 0)
    edge_x, edge_y, edge_z = np.where(edge_mask)
    
    # Plot edge points
    if len(edge_x) > 0:
        # Use stride to reduce number of points
        ax2.scatter(edge_x[::stride], edge_y[::stride], edge_z[::stride], 
                   c='r', marker='.', alpha=0.8)
        ax2.set_title('Outline of Angled Cylinder (Rotated)')
    else:
        ax2.set_title('Outline detection failed - showing full model')
        ax2.scatter(x[::stride], y[::stride], z[::stride], c='b', marker='.', alpha=0.5)

    # Format both 3D plots
    for axes in [ax, ax2]:
        axes.set_xlabel('X axis')
        axes.set_ylabel('Y axis')
        axes.set_zlabel('Z axis')
        
        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        axes.set_xlim(mid_x - max_range, mid_x + max_range)
        axes.set_ylim(mid_y - max_range, mid_y + max_range)
        axes.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title('Angled Cylinder (Rotated)')

    plt.savefig('angled_cylinder_3d.png', dpi=300)
    plt.savefig('angled_cylinder_outline.png', dpi=300)

    # Print information
    print(f"Angled cylinder created with dimensions {image_shape}")
    print(f"Cylinder radius: {radius:.2f} mm ({radius_voxels} voxels)")
    print(f"Resolution: {resolution} mm per voxel")
    print(f"Total cylinder length: {total_length} mm")
    print(f"First segment length: {length_straight} mm")
    print(f"Second segment length (45-degree angle): {length_angled} mm")
    rot_degrees = [np.degrees(angle) for angle in rot_angles]
    print(f"Rotations applied: {rot_degrees[0]:.2f}° (X), {rot_degrees[1]:.2f}° (Y), {rot_degrees[2]:.2f}° (Z)")


if __name__ == "__main__":
    for yangle in range(0, 45, 10):
        print(f"Creating cylinder with rotation angle: {yangle} degrees")
        main(yangle)
