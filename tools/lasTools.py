import laspy
import fiona
import csv
import geopandas as gpd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tools.voxelization as voxelization
import tools.dataTools as dt
import random

def iterate_over_las(las_file : str, bound : int, voxel_size : float):

    """
    This function iterates through a .las file and returns a tuple consisting of a list of voxel grids, as well as a
    parallel list containing each voxel grid's x, y maximums and minimums
    """

    #Load LiDAR data
    las = laspy.read(las_file)
    print(las)

    #Recording the data in a numpy array
    point_data = np.stack([las.x, las.y, las.z, las.classification], axis = 0).transpose((1,0))
    print(point_data)

    XMIN = 671821.2000000001
    XMAX = 673308.23
    YMIN = 3675320.3000000003
    YMAX = 3677078.63

    print('looping...')


    boxes = []
    for i in range(int((XMAX-XMIN)/bound + 1)):
        boxes.append([])
        for j in range(int((YMAX-YMIN)/bound + 1)):
            boxes[i].append([])
    
    print(len(boxes[0]))

    #initialize a 222x262 2D array for the colors
    colors = []
    for i in range(int((XMAX-XMIN)/bound + 1)):
        colors.append([])
        for j in range(int((YMAX-YMIN)/bound + 1)):
            colors[i].append([])

    for i in point_data:
        xIndex = math.floor((i[0] - XMIN)/bound)
        yIndex = math.floor((i[1] - YMIN)/bound)


        #Add the points and colors to boxes
        match i[3]:
            case 1:
                boxes[xIndex][yIndex].append([i[0], i[1], i[2], 0.416, 0.424, 0.471])
            case 2:
                boxes[xIndex][yIndex].append([i[0], i[1], i[2], 0.541, 0.357, 0.173])
            case 3:
                boxes[xIndex][yIndex].append([i[0], i[1], i[2], 0.329, 0.62, 0.8])
            case 4: 
                boxes[xIndex][yIndex].append([i[0], i[1], i[2], 0.137, 0.922, 0.216])
            case 5:
                boxes[xIndex][yIndex].append([i[0], i[1], i[2], 0.29, 0.69, 0.333])
            case 6:
                boxes[xIndex][yIndex].append([i[0], i[1], i[2], 0.922, 0.149, 0.149])
            case 7:
                boxes[xIndex][yIndex].append([i[0], i[1], i[2], 0.541, 0.145, 0.145])
            case 8: 
                boxes[xIndex][yIndex].append([i[0], i[1], i[2], 0.165, 0.843, 0.878])
            case 9:
                boxes[xIndex][yIndex].append([i[0], i[1], i[2], 0.898, 0.941, 0.192])
            case 10:
                boxes[xIndex][yIndex].append([i[0], i[1], i[2], 0.337, 0.251, 0.749])
            case 11:
                boxes[xIndex][yIndex].append([i[0], i[1], i[2], 0.282, 0.286, 0.322])
            case 12: 
                boxes[xIndex][yIndex].append([i[0], i[1], i[2], 0.949, 0.314, 0.651])

    #Initialize x,y,z, and colors lists
    x_data = []
    y_data = []
    z_data = []
    colors = []
    voxel_grid_list = []
    xy = []
    count = 0
    for i in boxes:
        count += 1
        for j in i:
            #Clear the lists when moving to a new box
            x_data.clear()
            y_data.clear()
            z_data.clear()
            colors.clear()
            for k in j:
                #fill the lists with all the values in the current box
                x_data.append(k[0])
                y_data.append(k[1])
                z_data.append(k[2])
                colors.append([k[3],k[4], k[5]])
                if k[3] == 0.137 or k[3] == 0.29:
                    green = True
            
            #If there is no x or y data, or if the boxe's x or y length is short it means we are on an edge piece and we should skip
            if(len(x_data) == 0 or len(y_data) == 0 or max(x_data) - min(x_data) <  (bound - 2) or max(y_data) - min(y_data) < (bound - 2)):
                continue
            
            # Print the data of the current box
            # print('point count: ', len(x_data))
            # print('xmin: ', min(x_data), 'xmax: ', max(x_data), 'diff: ', max(x_data) - min(x_data))
            # print('ymin: ', min(y_data), 'ymax: ', max(y_data), 'diff: ', max(y_data) - min(y_data))
            # print('zmin: ', min(z_data), 'zmax: ', max(z_data), 'diff: ', max(z_data) - min(z_data))

            
            # #draw the points using matplotlib
            # fig = plt.figure(figsize=(5, 5))
            # ax = fig.add_subplot(111, projection="3d")
            # ax.scatter(x_data, y_data, z_data)
            # ax.set_axis_off()
            # plt.show()

            #voxelize
            points = []
            xmin = min(x_data)
            ymin = min(y_data)
            zmin = min(z_data)
            for i in range(len(x_data)):
                points.append([x_data[i] - xmin, y_data[i] - ymin, z_data[i] - zmin])

            xy.append([min(x_data), min(y_data), max(x_data), max(y_data)])
            voxel_grid_list.append(voxelization.voxelize(points, colors, voxel_size, bound))

    
    return (voxel_grid_list, xy)

def get_voxelization(xmin : float, ymin: float, bound : int, voxel_size : float, las_file : str):
    """
    Given a x and y coordinate, a voxel grid is constructed of size bound x bound x bound whose origin is xmin and ymin.
    The voxel grid is returned
    """
    xmax = xmin + bound
    ymax = ymin + bound
    las = laspy.read(las_file)
    #Recording the data in a numpy array
    point_data = np.stack([las.x, las.y, las.z, las.classification], axis = 0).transpose((1,0))

    x_data = []
    y_data = []
    z_data = []
    colors = []
    l = []
    for i in point_data:
        if i[0] > xmin and i[0] <  xmax and i[1] > ymin and i[1] < ymax:
            if i[3] == 5:
                l.append(i[2])
            x_data.append(i[0])
            y_data.append(i[1])
            z_data.append(i[2])
            match i[3]:
                case 1:
                    colors.append([0.416, 0.424, 0.471])
                case 2:
                    colors.append([0.541, 0.357, 0.173])
                case 3:
                    colors.append([0.329, 0.62, 0.8])
                case 4: 
                    colors.append([0.137, 0.922, 0.216])
                case 5:
                    colors.append([0.29, 0.69, 0.333])
                case 6:
                    colors.append([0.922, 0.149, 0.149])
                case 7:
                    colors.append([0.541, 0.145, 0.145])
                case 8: 
                    colors.append([0.165, 0.843, 0.878])
                case 9:
                    colors.append([0.898, 0.941, 0.192])
                case 10:
                    colors.append([0.337, 0.251, 0.749])
                case 11:
                    colors.append([0.282, 0.286, 0.322])
                case 12: 
                    colors.append([0.949, 0.314, 0.651])
                case _:
                    colors.append([0, 0, 0])

    points = []
    zmin = min(z_data)
    for i in range(len(x_data)):
        points.append([x_data[i] - xmin, y_data[i] - ymin, z_data[i] - zmin])
    voxel = voxelization.voxelize(points, colors, voxel_size, bound)

    return voxel

def get_voxelization_heat(xmin : float, ymin: float, bound : int, voxel_size : float, las_file : str):
    """
    Given a x and y coordinate, a voxel grid is constructed of size bound x bound x bound whose origin is xmin and ymin.
    Each voxel will be colored according ot its height making the voxel easier to see.
    The voxel grid is returned
    """
    xmax = xmin + bound
    ymax = ymin + bound
    las = laspy.read(las_file)
    #Recording the data in a numpy array
    point_data = np.stack([las.x, las.y, las.z, las.classification], axis = 0).transpose((1,0))

    x_data = []
    y_data = []
    z_data = []
    classes = []
    colors = []
    for i in point_data:
        if i[0] > xmin and i[0] <  xmax and i[1] > ymin and i[1] < ymax:
            x_data.append(i[0])
            y_data.append(i[1])
            z_data.append(i[2])
            classes.append(i[3])
    xarr = np.array(x_data)
    yarr = np.array(y_data)
    zarr = np.array(z_data)
    carr = np.array(classes)
    #Associate x_data and y_data lists with z_data, then sort the all of them in descending z_data order
    idx = np.flip(np.argsort(zarr))
    xarr = np.array(xarr)[idx]
    yarr = np.array(yarr)[idx]
    zarr = np.array(zarr)[idx]
    carr = np.array(carr)[idx]

    red=0
    green=0
    blue=0

    rscale=2.5
    gscale=0
    bscale=0.5

    for i in range(len(z_data)):
        #Modify how the gradient changes here
        if red + (1/len(z_data)) < 0.95: # Don't want white voxels
            red = red + (1/len(z_data)) * rscale
        if green + (1/len(z_data)) < 0.95:
            green = green + (1/len(z_data)) * gscale
        if blue + (1/len(z_data)) < 0.95:
            blue = blue + (1/len(z_data)) * bscale
        if carr[i] == 6 or carr[i] == 2:
            colors.append([0.95, 0.95, 0.95])
        else:
            colors.append([red, green, blue])
        rscale = rscale + 0.00025 # 0.0025, 0.00075
        gscale = gscale + 0.000125 # 0.00125, 0.000125
        bscale = bscale + 0.0200 # 0.0125, 0.02
    points = []
    #print(len(colors) == len(x_data))
    zmin = min(z_data)
    for i in range(len(x_data)):
        points.append([xarr[i] - xmin, yarr[i] - ymin, zarr[i] - zmin])
    voxel = voxelization.voxelize(points, colors, voxel_size, bound)

    #print(max(z_data) - min(z_data))

    return voxel

def iterate_over_las_with_bounds(in_file : str, out_file : str, xmin : float, ymin : float, xmax : float, ymax : float, bound : int, voxel_size : float):
    """
    function will utilize iterate_over_las() and use boundries to section off a part of a .las file.
    The user is then prompted to label each voxel grid. The grids are then saved to out_file in csv format.
    """
    count = 0
    (voxels, xy) = iterate_over_las(in_file, 30, 1)

    zipped = list(zip(voxels, xy))
    random.shuffle(zipped)
    voxels, xy = zip(*zipped)
    print(len(voxels))
    for i in range(len(xy)):
        if xy[i][0] >  xmin and xy[i][2] < xmax and xy[i][1] > ymin and xy[i][3] < ymax:
            print("(", xy[i][0], ", ", xy[i][1], "), (", xy[i][2], ", ", xy[i][3], ")")
            voxelization.visualize_voxel_grid(voxels[i])
            usr = input('How many trees are contained in that voxel grid?')
            if usr == 'u' :
                continue
            c = voxelization.fill_voxel_grid(voxels[i], 30, 20, 1)[1]
            dt.voxel_to_csv(out_file, c, usr)
            print(i, "/", len(voxels))

def iterate_over_las_with_shp(las_file : str, shp_file : str, out_file : str, xmin : float, ymin : float, xmax : float, ymax : float, bound : int, height : int, voxel_size : float):
    
    """
    Given a .las file and a .shp file this function iterates over the las file and writes voxel grids to a csv labeled with the number of .shp points
    each the voxel grid. Used to get labeled data for a regression AI
    """

    (voxels, xy) = iterate_over_las(las_file, bound, voxel_size)

    voxels_in_bounds = []

    m = 0
    total = 0
    iter = 1
    
    for i in range(len(xy)):
        if xy[i][0] >  xmin and xy[i][2] < xmax and xy[i][1] > ymin and xy[i][3] < ymax:
            iter += 1
            print("(", xy[i][0], ", ", xy[i][1], "), (", xy[i][2], ", ", xy[i][3], ")")
            voxels_in_bounds.append(voxels[i])
            count = 0
            for feat in fiona.open(shp_file):
                if feat['properties']['xcoord'] > xy[i][0] and feat['properties']['xcoord'] < xy[i][2] and feat['properties']['ycoord'] > xy[i][1] and feat['properties']['ycoord'] < xy[i][3]:
                    count += 1
            if count > m:
                m = count
            total += count
            if count <= 7:
                c = voxelization.fill_voxel_grid(voxels[i], bound, height, voxel_size)[1]
                dt.voxel_to_csv(out_file, c, count)

def segment_over_las_with_shp(las_file : str, shp_file : str, out_file : str, xmin : float, ymin : float, xmax : float, ymax : float, bound : int, height : int, voxel_size : float):

    """
    Given a .las file and a .shp file this function iterates over the las file and writes voxel grids to a csv file labeled with their corresponding
    bounding boxes gathered from the .shp file. Used for getting labeled data for segmentation
    """

    (voxels, xy) = iterate_over_las(las_file, bound, voxel_size)

    voxels_in_bounds = []

    writer = csv.writer(open(out_file, 'w'))

    #Create a dictionary containing shapefile points paired with their id's
    shpDict = dict()
    for feat in fiona.open(shp_file):
        shpDict[feat['properties']['id']] = feat

    #Iterate over the list of voxel grids
    for i in range(len(xy)):
        if xy[i][0] >  xmin and xy[i][2] < xmax and xy[i][1] > ymin and xy[i][3] < ymax:
            # print("(", xy[i][0], ", ", xy[i][1], "), (", xy[i][2], ", ", xy[i][3], ")")
            voxels_in_bounds.append(voxels[i])

            smallDict = dict()

            #Add points within this grid to a dictionary
            for j in shpDict:
                if (shpDict[j]['properties']['xcoord'] > xy[i][0] and shpDict[j]['properties']['xcoord'] < xy[i][2] and 
                    shpDict[j]['properties']['ycoord'] > xy[i][1] and shpDict[j]['properties']['ycoord'] < xy[i][3]):
                    smallDict[j] = (shpDict[j]['properties']['xcoord'], shpDict[j]['properties']['ycoord'])

            newPoints = []
            idxs = []
            for j in smallDict:
                tempTuple = (0, 0)
                if j % 2 == 1:
                    if j+1 not in smallDict:
                        if shpDict[j+1]['properties']['xcoord'] > xmin and shpDict[j+1]['properties']['xcoord'] < xmax:
                            tempTuple = (shpDict[j+1]['properties']['xcoord'], tempTuple[1])
                        else:
                            tempTuple = (xy[i][2], tempTuple[1])
                        
                        if shpDict[j+1]['properties']['ycoord'] > ymin and shpDict[j+1]['properties']['ycoord'] < ymax:
                            tempTuple = (tempTuple[0], shpDict[j+1]['properties']['ycoord'])
                        else:
                            tempTuple = (tempTuple[0], xy[i][3])
                        newPoints.append(tempTuple)
                        idxs.append(j+1)
                else:
                    if j-1 not in smallDict:
                        if shpDict[j-1]['properties']['xcoord'] > xmin and shpDict[j-1]['properties']['xcoord'] < xmax:
                            tempTuple = (shpDict[j-1]['properties']['xcoord'], tempTuple[1])
                        else:
                            tempTuple = (xy[i][0], tempTuple[1])
                        
                        if shpDict[j-1]['properties']['ycoord'] > ymin and shpDict[j-1]['properties']['ycoord'] < ymax:
                            tempTuple = (tempTuple[0], shpDict[j-1]['properties']['ycoord'])
                        else:
                            tempTuple = (tempTuple[0], xy[i][1])
                        newPoints.append(tempTuple)
                        idxs.append(j-1)



            for j in range(len(newPoints)):
                smallDict[idxs[j]] = newPoints[j]
            idxs = []
            for j in smallDict:
                idxs.append(j)
            idxs.sort()
            print(idxs)
            boundingCoord = []
            for j in idxs:
                boundingCoord.append((max(0, min(bound-1, int(smallDict[j][0] - xy[i][0]))), max(0, min(bound-1, int(smallDict[j][1] - xy[i][1])))))
            
            j = len(boundingCoord) - 1
            while j >= 0:
                if j % 2 == 0:
                    print(len(boundingCoord))
                    print(j)
                    if ((boundingCoord[j+1][0]-boundingCoord[j][0]) * (boundingCoord[j+1][1] - boundingCoord[j][1]) < 8):
                        boundingCoord.pop(j+1)
                        boundingCoord.pop(j)
                j-=1

            print(xy[i])
            print(boundingCoord)
            colors = voxelization.fill_voxel_grid(voxels[i], bound, height, voxel_size)[1]
            row = []
            for i in colors:
                if i[0] == 0:
                    row.append(0)
                else:
                    row.append(1)

            # v2 = dt.create_bounding_box(row, boundingCoord, bound, height, voxel_size)
            # voxelization.visualize_voxel_grid(v2)

            for j in boundingCoord:
                row.append(j)
            writer.writerow(row)

            #Finally add the points to the voxel in csv form and write