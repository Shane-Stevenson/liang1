import open3d as o3d
import random
import numpy as np
import tools.voxelization as voxelization
import json
import math
import tools.lasTools as lasTools
import csv

def voxel_to_csv(out_location : str, colors : list, label : int):
    """
    Recieves a 'colors' array outputted by voxelization.fill_voxel_grid(), and creates a 1D array representation of the 
    voxel grid and writes it to 'out_location'. Additionally the label is appended to the end
    """
    row = []
    with open(out_location, 'a') as f:
        writer = csv.writer(f)
        for i in colors:
            if i[0] == 0:
                row.append(0)
            else:
                row.append(1)

        row.append(label) # 0 represents no tree, 1 represents 1 tree, 2 represents multiple trees
        writer.writerow(row)
    
def voxel_to_1D(v : o3d.geometry.VoxelGrid, bound : int, height : int, voxel_size : int):
    c = voxelization.fill_voxel_grid(v, bound, height, voxel_size)[1]
    l = []
    for i in c:
        if i[0] == 0:
            l.append(0)
        else:
            l.append(1)
        
    return l

def csv_to_voxel(voxel_in_csv_format : list, bound : int, height : int, voxel_size : float):
    """
    recieves a 1D array representation of a voxel and constructs an o3d.VoxelGrid object from the given array.
    Returns a tuple containing the voxel, and its label
    """
    x = 0
    y = 0
    z = 0
    points = []
    colors = []
    #Remove the label and store it
    label = voxel_in_csv_format.pop()

    for i in range(len(voxel_in_csv_format)):
        if int(voxel_in_csv_format[i]) == 1:
                points.append([x, y, z])
                colors.append([.9-z/height, .9-z/height, .9-z/height])
        
        z += voxel_size
        if z >= height:
            y += voxel_size
            z = 0
        
        if y >= bound:
            x += voxel_size
            y = 0
        
    #Create a point cloud and then initialize a VoxelGrid
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    mins = np.array([0, 0, 0])      #min and max could be changed to function arguments for more manuverability
    maxs = np.array([bound, bound, height])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(point_cloud, voxel_size, mins, maxs)

    return (voxel_grid, label)

def createTrainingData(positive_csv : str, negative_csv : str, out_location, count : int):
    """
    This function accepts two csv files and produces one csv file which contains a random sample of size n from each csv file.
    It is to be used on two csv files that contain trees and non-trees positions respectively to produce a random sample with an 
    equal amount of each

    Parameters:
    -----------

    positive_csv : str
        this file contains voxel grids that contain trees in csv format

    negative_csvc : str
        this file contains voxel grids that do not contain trees in csv format

    out_location : str
        location for the random sample to be saved

    count : int
        The size of the random sample to be taken from both win_csv and loss_csv
    """
    rowCount = 0
    for row in open(positive_csv):
        rowCount+= 1
    positive = rowCount
    rowCount = 0
    for row in open(negative_csv):
        rowCount+= 1
    negative = rowCount

    with open(positive_csv, 'r') as w:
        with open(negative_csv, 'r') as l:
            with open(out_location, 'w') as o:
                positiveReader = csv.reader(w)
                negativeReader = csv.reader(l)
                writer = csv.writer(o)

                winList = list(positiveReader)
                lossList = list(negativeReader)

                for i in random.sample(range(0, positive), count):
                    writer.writerow(winList[i])

                for i in random.sample(range(0, negative), count):
                    writer.writerow(lossList[i])

def affine_augment_csv_file(in_file : str, out_file : str, bound : int, height : int, voxel_size : float):
    """
    This function recieves a csv file of voxel grids, (created by voxel_to_csv()) and generates 7 new voxel grids by rotating the 
    original one 90 degrees 4 times, and then mirroring it and rotating it again 3 times.
    """
    file = open(in_file)
    csvreader = csv.reader(file)
    csv_voxel_grid = list(csvreader)

    for i in csv_voxel_grid:
        tuple = csv_to_voxel(i, bound, height, voxel_size)
        voxel = tuple[0]
        label = tuple[1]
        for j in range(4):
            full = voxelization.fill_voxel_grid(voxel, bound, height, voxel_size)[1]
            voxel_to_csv(out_file, full, label)
            voxel = rotate_90(voxel, bound, height, voxel_size)
        
        voxel = mirror(voxel, bound, height, voxel_size)
        for j in range(4):
            full = voxelization.fill_voxel_grid(voxel, bound, height, voxel_size)[1]
            voxel_to_csv(out_file, full, label)
            voxel = rotate_90(voxel, bound, height, voxel_size)
    
def random_augment_csv(in_file : str, out_file : str, bound : int, height : int, voxel_size : float, thousandths : int, iterations : int):
    """
    This function recieves a csv file of voxel grids, (created by voxel_to_csv()) and generates n new voxel grids by 
    randomly selecting voxels adjacent to those already on, and randomly turning them on or off (depending on their current state)
    """
    file = open(in_file)
    csvreader = csv.reader(file)
    csv_voxel_grid = list(csvreader)

    for i in csv_voxel_grid:
        tuple = csv_to_voxel(i, bound, height, voxel_size)
        voxel = tuple[0]
        label = tuple[1]
        full = voxelization.fill_voxel_grid(voxel, bound, height, voxel_size)[1]
        voxel_to_csv(out_file, full, label)
        for j in range(iterations):
            vnew = generate_new_data(voxel, bound, height, voxel_size, thousandths)
            full = voxelization.fill_voxel_grid(vnew, bound, height, voxel_size)[1]
            voxel_to_csv(out_file, full, label)

def voxel_to_zxy(v : o3d.geometry.VoxelGrid, bound : int, height : int, voxel_size : float):
    """
    This function recieves a voxel grid and returns a 3 dimensional array whose indices are z,x,y. Thus, calling 
    zyx[2][0][3] returns a one or zero at z=3, x=1, y=4 depending on whether on not that voxel is turned on
    """
    xyz = []
    z=0
    x=0
    y=0
    for i in range(int(bound*bound*height/(voxel_size**3))):
        xyz.append([x, y, z])  
        z += voxel_size
        if z >= height:
            y += voxel_size
            z = 0
        
        if y >= bound:
            x += voxel_size
            y = 0
    bools = v.check_if_included(o3d.utility.Vector3dVector(xyz))
    zxy = []
    for i in range(int(height/voxel_size)):
        zxy.append([])
        for j in range(int(bound/voxel_size)):
            zxy[i].append([])
            for k in range(int(bound/voxel_size)):
                zxy[i][j].append(0)

    for i in range(len(bools)):
        if bools[i]:
            zxy[int(xyz[i][2]/voxel_size)][int(xyz[i][0]/voxel_size)][int(xyz[i][1]/voxel_size)] = 1

    return zxy

def voxel_to_zxy4D(v : o3d.geometry.VoxelGrid, bound : int, height : int, voxel_size : float):
    """
    This function recieves a voxel grid and returns a 3 dimensional array whose indices are z,x,y. Thus, calling 
    zyx[2][0][3] returns a [0] or [1] at z=3, x=1, y=4 depending on whether on not that voxel is turned on.

    This function returns a 4D list because pyTorch's 3dConv function requires a specific input shape that is 
    not satisfied with the original voxel_to_zxy() function
    """
    xyz = []
    z=0
    x=0
    y=0
    for i in range(int(bound*bound*height/(voxel_size**3))):
        xyz.append([x, y, z])  
        z += voxel_size
        if z >= height:
            y += voxel_size
            z = 0
        
        if y >= bound:
            x += voxel_size
            y = 0
    
    bools = v.check_if_included(o3d.utility.Vector3dVector(xyz))
    
    zxy = []
    for i in range(int(height/voxel_size)):
        zxy.append([])
        for j in range(int(bound/voxel_size)):
            zxy[i].append([])
            for k in range(int(bound/voxel_size)):
                zxy[i][j].append([0])

    for i in range(len(bools)):
        if bools[i]:
            zxy[int(xyz[i][2]/voxel_size)][int(xyz[i][0]/voxel_size)][int(xyz[i][1]/voxel_size)] = [1]

    return zxy

def rotate_90(v : o3d.geometry.VoxelGrid, bound : int, height : int, voxel_size : float):
    """
    This function recieves a voxel grid and rotates it 90 degrees, and then returns the new rotated voxel grid
    """
    zxy = voxel_to_zxy(v, bound, height, voxel_size)
    #rotate zxy
    for i in range(len(zxy)):
        rotated = list(reversed(list(zip(*zxy[i]))))
        zxy[i] = rotated
    
    points = []
    for z in range(int(height/voxel_size)):
        for x in range(int(bound/voxel_size)):
            for y in range(int(bound/voxel_size)):
                if zxy[z][x][y] == 1:
                    points.append([x, y, z])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    mins = np.array([0, 0, 0])      #min and max could be changed to function arguments for more manuverability
    maxs = np.array([bound, bound, height])
    rotated_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(point_cloud, voxel_size, mins, maxs)

    return rotated_voxel_grid

def mirror(v : o3d.geometry.VoxelGrid, bound : int, height : int, voxel_size : float):
    """
    This function recives a voxel grid, mirrors it across the x axis (I think), and returns the new voxel grid 
    """
    zxy = voxel_to_zxy(v, bound, height, voxel_size)
        
    #mirror zxy
    for i in range(len(zxy)):
        mirrored = list(reversed(zxy[i]))
        zxy[i] = mirrored
    
    points = []
    for z in range(int(height/voxel_size)):
        for x in range(int(bound/voxel_size)):
            for y in range(int(bound/voxel_size)):
                if zxy[z][x][y] == 1:
                    points.append([x, y, z])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    mins = np.array([0, 0, 0])      #min and max could be changed to function arguments for more manuverability
    maxs = np.array([bound, bound, height])
    rotated_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(point_cloud, voxel_size, mins, maxs)

    return rotated_voxel_grid

def generate_new_data(v : o3d.geometry.VoxelGrid, bound : int, height : int, voxel_size : float, percent : int):
    """
    This function recieves a voxel grid and injects noise into it by randomly selecting voxels adjacent to given voxels
    that are already on, and turns adjacent ones on or off respective of their current state. The new voxel grid is then
    returned
    """
    zxy = voxel_to_zxy(v, bound, height, voxel_size)
    points = []
    
    count = 0
    for i in range(int(height/voxel_size)):
        for j in range(int(bound/voxel_size)):
            for k in range(int(bound/voxel_size)):
                if i <= 3:
                    continue
                if(zxy[i][j][k] == 1):
                    points.append([i, j, k]) 

    manipulations = [ [-1, -1, +1], [0, -1, +1], [+1, -1, +1], 
                    [-1, 0, +1], [0, 0, +1], [+1, 0, +1], 
                    [-1, +1, +1], [0, +1, +1], [+1, +1, +1], 
                    [-1, -1, 0], [0, -1, 0], [+1, -1, 0],
                    [-1, 0, 0], [+1, 0, 0], 
                    [-1, +1, 0], [0, +1, 0], [+1, +1, 0], 
                    [-1, -1, -1], [0, -1, -1], [+1, -1, -1],
                    [-1, 0, -1], [0, 0, -1], [+1, 0, -1], 
                    [-1, +1, -1], [0, +1, -1], [+1, +1, -1] ]

    for i in points:
        for j in manipulations:
            if random.randint(0,1000) < percent:
                if i[0]+j[0] < height and i[0]+j[0] >= 0 and i[1]+j[1] < bound and  i[1]+j[1] < bound >= 0 and i[2]+j[2] < bound and i[2]+j[2] >= 0:
                    if zxy[i[0]+j[0]][i[1]+j[1]][i[2]+j[2]] == 0:
                        zxy[i[0]+j[0]][i[1]+j[1]][i[2]+j[2]] = 1
                    else:
                        zxy[i[0]+j[0]][i[1]+j[1]][i[2]+j[2]] = 0

    points = []
    for z in range(int(height/voxel_size)):
        for x in range(int(bound/voxel_size)):
            for y in range(int(bound/voxel_size)):
                if zxy[z][x][y] == 1:
                    points.append([x, y, z])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    mins = np.array([0, 0, 0])      #min and max could be changed to function arguments for more manuverability
    maxs = np.array([bound, bound, height])
    rotated_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(point_cloud, voxel_size, mins, maxs)

    return rotated_voxel_grid

def split_csv_file(in_file : str, train_file : str, test_file : str, train_ratio : float):
    """
    This function will split one file into two with sizes determined by train_ration.
    Used to create a train set and a validation set
    """
    reader = csv.reader(open(in_file))
    vals = list(reader)
    switchVal = math.ceil(len(vals) * train_ratio)
    writer1 = csv.writer(open(train_file, 'w'))
    writer2 = csv.writer(open(test_file, 'w'))
    idx = 0
    while idx < switchVal:
        writer1.writerow(vals[idx])
        idx+=1
    
    while idx % 8 != 0:
        writer1.writerow(vals[idx])
        idx+=1
    
    while idx < len(vals):
        writer2.writerow(vals[idx])
        idx+=1
           
def ceiling_on_csv(in_file : str, out_file : str, bound : int, height : int, voxel_size : float):
    """
    This function accepts a csv file containing voxels, and places a ceiling on each voxel and ouputs the new voxels in another
    location
    """
    file = open(in_file)
    csvreader = csv.reader(file)
    csv_voxel_grid = list(csvreader)

    for i in csv_voxel_grid:
        if len(i) == 0:
            continue
        x = 0
        y = 0
        z = 0
        points = []
        #Remove the label and store it
        label = i.pop()

        for j in range(len(i)):
            if int(i[j]) == 1:
                if z < height/voxel_size:
                    points.append([x, y, z])
            
            z += voxel_size
            if z >= bound/voxel_size:
                y += voxel_size
                z = 0
            
            if y >= bound/voxel_size:
                x += voxel_size
                y = 0
            
        #Create a point cloud and then initialize a VoxelGrid
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        mins = np.array([0, 0, 0])      #min and max could be changed to function arguments for more manuverability
        maxs = np.array([bound, bound, height])
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(point_cloud, voxel_size, mins, maxs)
        
        c = voxelization.fill_voxel_grid(voxel_grid, bound, height, voxel_size)[1]
        voxel_to_csv(out_file, c, label)

def merge_csv_files(files : list, out_location : str):
    """
    This function recieves a list of strings of file locations and merges the csv files into one file.
    """
    merge = open(out_location, 'w')
    writer = csv.writer(merge)

    for i in files:
        file = open(i)
        reader = csv.reader(file)
        voxels = list(reader)
        for j in voxels:
            writer.writerow(j)

def create_bounding_box(voxel_in_csv_format : list, coords : list, bound : int, height : int, voxel_size : float):
    """
    This function recieves a voxel in csv format as well as a list of coordinates (bounding boxes) and creates a new voxel grid where the 
    bounding boxes are visualized. This is used for easy data visualization

    coords = list N,4
    """
    x = 0
    y = 0
    z = 0
    points = []
    colors = []

    print(coords)
    for i in range(len(voxel_in_csv_format)):
        temp = False
        for j in coords:
            if x == j[0] and y == j[1] or x == j[2] and y == j[3]:
                temp = True
                break
        if int(voxel_in_csv_format[i]) == 1 or (temp):
            points.append([x, y, z])
            colors.append([.9-z/height, .9-z/height, .9-z/height])
        
        z += voxel_size
        if z >= height:
            y += voxel_size
            z = 0
        
        if y >= bound:
            x += voxel_size
            y = 0

    #Create a point cloud and then initialize a VoxelGrid
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    mins = np.array([0, 0, 0])      #min and max could be changed to function arguments for more manuverability
    maxs = np.array([bound, bound, height])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(point_cloud, voxel_size, mins, maxs)

    return voxel_grid

def fill_voxel_boundary_boxes(in_file : str, out_file : str, padding : int):
    """
    Once segment_over_las_with_shp is ran, the various voxel grids in the csv file will have a varying amount of bounding boxes. This function
    iterates through the csv file and creates a new csv file where each voxel grid has the same size label. Voxels with a non-maximum amount
    of bounding boxes have their label's extra space filled with zeroes to represent no bounding box.

    This is used to create uniform data so that a segmentation AI can use this preprocessed data
    """
    reader = csv.reader(open(in_file))
    vs = list(reader)
    m = 0
    for i in vs:
        count = 0
        idx = len(i)-1
        while type(eval(i[idx])) == tuple:
            count += 1
            idx -= 1
        m = max(count, m)
    print(m)
    writer = csv.writer(open(out_file, 'w'))
    for i in vs:
        count = 0
        idx = len(i)-1
        while type(eval(i[idx])) == tuple:
            count += 1
            idx -= 1
        iter = 0
        while iter < m - count:
            i.append((0,0))
            iter += 1
        writer.writerow(i)



