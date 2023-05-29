import open3d as o3d
import numpy as np

def voxelize(points : list, voxel_size : int):
    # Initialize a point cloud object
    pcd = o3d.geometry.PointCloud()

    # Add the points, colors and normals as Vectors
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = []
    for i in points:
        colors.append([.5, .5, .5])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    #Create a voxel grid from the point cloud with a voxel_size of voxel_size and with a minimum bound of 0 and maximum bound of 10
    mins = np.array([0, 0, 0])      #min and max could be changed to function arguments for more manuverability
    maxs = np.array([10, 10, 10])
    voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size, mins, maxs)

    for v in voxel_grid.get_voxels():
        print(v.grid_index)

    #make new pointcloud and then a new voxelGrid
    fullPoints = []
    fullColors = []
    xyz = []
    for x in range(int(maxs[0]/voxel_size)):
        for y in range(int(maxs[1]/voxel_size)):
            for z in range(int(maxs[2]/voxel_size)):
                xyz.append([x*voxel_size, y*voxel_size, z*voxel_size])
    
    #If the voxel exists in the one we created from the original point cloud, color it and add it to the new voxel grid
    #If the voxel does not exist in the original voxel grid, color it black and add it to the new voxel grid
    #The colors is how the AI will distinguish voxels that represent points and voxels that represent empty space
    bools = voxel_grid.check_if_included(o3d.utility.Vector3dVector(xyz))
    for i in range(len(bools)):
        if(bools[i]):
            fullPoints.append(xyz[i])
            fullColors.append([.5, .5, .5])
        else:
            fullPoints.append(xyz[i])
            fullColors.append([0, 0, 0])

    fullpcd = o3d.geometry.PointCloud()
    fullpcd.points = o3d.utility.Vector3dVector(fullPoints)
    fullpcd.colors = o3d.utility.Vector3dVector(fullColors)
    full_voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(fullpcd, voxel_size, mins, maxs)

    # for v in full_voxel_grid.get_voxels():
    #     print(v.grid_index)
    print('min: ', full_voxel_grid.get_min_bound())
    print('max: ', full_voxel_grid.get_max_bound())

    # Initialize a visualizer object
    vis = o3d.visualization.Visualizer()
    # Create a window, name it and scale it
    vis.create_window(window_name='Open 3D visualizer', width=800, height=600)

    # Add the voxel grid to the visualizer
    vis.add_geometry(full_voxel_grid)

    # run the visualizater
    vis.run()
    # Once the visualizer is closed destroy the window and clean up
    vis.destroy_window()
