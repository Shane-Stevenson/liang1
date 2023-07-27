import open3d as o3d
import random
import numpy as np
import tools.voxelization as voxelization
import json
import tools.lasTools as lasTools
import tools.dataTools as dataTools
import csv

def voxel_to_csv_segmented(out_location : str, colors : list, label : int):
    """
    Writes segmented voxel data to a csv file
    """
    row = []
    with open(out_location, 'a') as f:
        writer = csv.writer(f)
        for i in colors:
            if i[0] == 0:
                row.append(0)
            else:
                row.append(1)
        for i in label:
            row.append(i) 
        writer.writerow(row)

def csv_to_voxel_segmented(v : list, bound : int, height : int, voxel_size : float):
    """
    recieves a 1D array representation of a voxel and its bounding boxes and constructs an o3d.VoxelGrid object from the given array.
    Returns a tuple containing the voxel, and its label
    """
    x = 0
    y = 0
    z = 0
    points = []
    colors = []
    #Remove the label and store it
    label = []
    while (len(v) != int((bound*bound*height)/voxel_size**3)): #144000
        label.append(eval(v.pop()))

    for i in range(len(v)):
        if int(v[i]) == 1:
            points.append([x, y, z])
            colors.append([.9-z/height, .9-z/height, .9-z/height])
        
        z += voxel_size
        if z >= height:
            y += voxel_size
            z = 0
        
        if y >= bound:
            x += voxel_size
            y = 0
        
    # quit()
    #Create a point cloud and then initialize a VoxelGrid
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    mins = np.array([0, 0, 0])
    maxs = np.array([bound, bound, height])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(point_cloud, voxel_size, mins, maxs)

    return (voxel_grid, label)

def rotate_90_segmented(v : o3d.geometry.VoxelGrid, label : list, bound : int, height : int, voxel_size : float):
    """
    This function recieves a voxel grid and rotates it 90 degrees, and then returns the new rotated voxel grid
    """
    zxy = dataTools.voxel_to_zxy(v, bound, height, voxel_size)
    print(label)
    #rotate zxy
    for i in range(len(zxy)):
        rotated = list(reversed(list(zip(*zxy[i]))))
        zxy[i] = rotated
    
    points = []
    x = 0
    y = 0
    z = 0
    for i in range(int(bound*bound*height/(voxel_size**3))):
        if(zxy[int(z/voxel_size)][int(x/voxel_size)][(int(y/voxel_size))]):
            points.append([x, y, z])  
        z += voxel_size
        if z >= height:
            y += voxel_size
            z = 0
        
        if y >= bound:
            x += voxel_size
            y = 0

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    mins = np.array([0, 0, 0])      #min and max could be changed to function arguments for more manuverability
    maxs = np.array([bound, bound, height])
    rotated_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(point_cloud, voxel_size, mins, maxs)

    #rotate label
    ret = []
    for i in label:
        temp = [j-(bound-1)/2 for j in i]
        temp[0], temp[1] = -1*temp[1], temp[0]
        temp[2], temp[3] = -1*temp[3], temp[2]
        temp = [j + (bound-1)/2 for j in temp]
        temp[0], temp[2] = temp[2], temp[0]
        ret.append(temp)
    
    return rotated_voxel_grid, ret

def mirror_segmented(v : o3d.geometry.VoxelGrid, label : list, bound : int, height : int, voxel_size : float):
    """
    This function recives a voxel grid, mirrors it across the y axis (I think), and returns the new voxel grid along with the mirrored labels
    """
    zxy = dataTools.voxel_to_zxy(v, bound, height, voxel_size)
        
    #mirror zxy
    for i in range(len(zxy)):
        mirrored = list(reversed(zxy[i]))
        zxy[i] = mirrored
    
    points = []
    x = 0
    y = 0
    z = 0
    for i in range(int(bound*bound*height/(voxel_size**3))):
        if(zxy[int(z/voxel_size)][int(x/voxel_size)][(int(y/voxel_size))]):
            points.append([x, y, z])  
        z += voxel_size
        if z >= height:
            y += voxel_size
            z = 0
        
        if y >= bound:
            x += voxel_size
            y = 0

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    mins = np.array([0, 0, 0])      #min and max could be changed to function arguments for more manuverability
    maxs = np.array([bound, bound, height])
    rotated_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(point_cloud, voxel_size, mins, maxs)

    #mirror label
    ret = []
    for i in label:
        temp = [j-(bound-1)/2 for j in i]
        temp[0] *= -1
        temp[2] *= -1
        temp = [j + (bound-1)/2 for j in temp]
        temp[0], temp[2] = temp[2], temp[0]
        ret.append(temp)

    return rotated_voxel_grid, ret

def affine_augment_segmented(in_file : str, out_file : str, bound : int, height : int, voxel_size : float):
    """
    This function recieves a csv file of voxel grids, (created by voxel_to_csv()) and generates 7 new voxel grids by rotating the 
    original one 90 degrees 4 times, and then mirroring it and rotating it again 3 times.
    """
    file = open(in_file)
    csvreader = csv.reader(file)
    csv_voxel_grid = list(csvreader)

    for i in csv_voxel_grid:
        voxel, label = csv_to_voxel_segmented(i, bound, height, voxel_size)
        for j in range(4):
            full = voxelization.fill_voxel_grid(voxel, bound, height, voxel_size)[1]
            voxel_to_csv_segmented(out_file, full, label)
            voxel, label = rotate_90_segmented(voxel, label, bound, height, voxel_size)
        
        voxel, label = mirror_segmented(voxel, label, bound, height, voxel_size)
        for j in range(4):
            full = voxelization.fill_voxel_grid(voxel, bound, height, voxel_size)[1]
            voxel_to_csv_segmented(out_file, full, label)
            voxel, label = rotate_90_segmented(voxel, label, bound, height, voxel_size)

def voxel_1d_to_4D(in_file : str, out_file : str, bound : int, height : int, voxel_size : float):
    """
    Given 1 dimensional voxel grid data from dataTools.voxel_to_csv(), this function creates a 4D array for use
    in Pytorch's 3D convolutional layers
    """
    reader = csv.reader(open(in_file))
    voxels = list(reader)
    writer = csv.writer(open(out_file, 'w'))
    for i in voxels:
        v, l = csv_to_voxel_segmented(i, bound, height, voxel_size)
        a = dataTools.voxel_to_zxy4D(v, bound, height, voxel_size)
        a = (np.transpose(np.array(a), (3, 1, 2, 0)))
        a = a.tolist()
        for j in l:
            a.append(j)
        writer.writerow(a)
    