import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

def straighten_optical_nerve(pcd_path, output_path):
    """
    Straightens a curved 3D point cloud of an optical nerve.

    Args:
        pcd_path: Path to the input point cloud file (e.g., .ply, .pcd).
        output_path: Path to save the straightened point cloud.
    """

    # Load the point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # 1. Estimate the nerve's central curve (e.g., using a spline or polynomial fit)
    #    This is a crucial step and might require some experimentation depending on the nerve's shape.
    #    A simple approach (if the curve is relatively smooth) is to use a moving average.

    # Simple Moving Average (adjust window size as needed)
    window_size = 20  # Example window size
    smoothed_points = []
    for i in range(window_size // 2, len(points) - window_size // 2):
        window = points[i - window_size // 2:i + window_size // 2]
        center = np.mean(window, axis=0)
        smoothed_points.append(center)
    smoothed_points = np.array(smoothed_points)

    # More robust curve fitting methods (if SMA is insufficient):
    # - Use a spline library like `scipy.interpolate.splprep` and `splev`
    # - Fit a polynomial to the points (e.g., using `numpy.polyfit`)

    # 2. Calculate the tangent vectors along the central curve.
    tangents = []
    for i in range(len(smoothed_points) - 1):
        tangent = smoothed_points[i+1] - smoothed_points[i]
        tangent /= np.linalg.norm(tangent)  # Normalize
        tangents.append(tangent)
    tangents.append(tangents[-1]) # Add last tangent to make lengths equal

    # 3. Rotate each point to align the local tangent with a desired direction (e.g., the z-axis).

    straightened_points = []
    for i in range(len(points)):
        # Find the closest point on the smoothed curve
        distances = np.linalg.norm(points[i] - smoothed_points, axis=1)
        closest_index = np.argmin(distances)
        closest_point = smoothed_points[closest_index]
        tangent = tangents[closest_index]


        # Calculate rotation
        z_axis = np.array([0, 0, 1])  # Desired direction (z-axis)
        rotation_axis = np.cross(tangent, z_axis)
        if np.linalg.norm(rotation_axis) < 1e-6: # Handle case where vectors are parallel
          rotation = Rotation.from_matrix(np.eye(3)) # No rotation needed
        else:
          rotation_angle = np.arccos(np.dot(tangent, z_axis))
          rotation = Rotation.from_rotvec(rotation_angle * rotation_axis / np.linalg.norm(rotation_axis))


        # Translate point to origin, rotate, and translate back
        rotated_point = rotation.apply(points[i] - closest_point) + closest_point
        straightened_points.append(rotated_point)

    straightened_points = np.array(straightened_points)

    # Create and save the straightened point cloud
    straightened_pcd = o3d.geometry.PointCloud()
    straightened_pcd.points = o3d.utility.Vector3dVector(straightened_points)
    o3d.io.write_point_cloud(output_path, straightened_pcd)

# Example usage:
input_pcd = "path/to/your/nerve.ply"  # Replace with your input file path
output_pcd = "path/to/output/straightened_nerve.ply"  # Replace with your desired output path
straighten_optical_nerve(input_pcd, output_pcd)

print(f"Straightened point cloud saved to: {output_pcd}")