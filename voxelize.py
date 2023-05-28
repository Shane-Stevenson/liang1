import open3d as o3d
import numpy as np

def voxelize(points : list, voxel_size : int):
    # Initialize a point cloud object
    pcd = o3d.geometry.PointCloud()

    # Add the points, colors and normals as Vectors
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = []
    normals = []
    for i in points:
        colors.append([.5, .5, .5])
        normals.append([.5, .5, .5])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    #Create a voxel grid from the point cloud with a voxel_size of 0.01
    voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=voxel_size)

    # Initialize a visualizer object
    vis = o3d.visualization.Visualizer()
    # Create a window, name it and scale it
    vis.create_window(window_name='Bunny Visualize', width=800, height=600)

    # Add the voxel grid to the visualizer
    vis.add_geometry(voxel_grid)

    # We run the visualizater
    vis.run()
    # Once the visualizer is closed destroy the window and clean up
    vis.destroy_window()
