import laspy
import fiona
import csv
import geopandas as gpd
import math
import matplotlib.pyplot as plt
import numpy as np
import tools.voxelization as voxelization
import tools.dataTools as dt
import random
import pyviz3d.visualizer as viz
import os

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

    XMIN = las.header.mins[0]
    XMAX = las.header.maxs[0]
    YMIN = las.header.mins[1]
    YMAX = las.header.maxs[1]
    print(XMIN)

    print('looping...')

    #'boxes' is a list containing the las file's point cloud data, but cut up into smaller grids. Say for example 
    #the .las file is 90x90, and the bound chosen is 30, then boxes would be 3x3, where boxes[0][0] contains a list 
    #of the points in the las file that have x values below 30, and y values below 30. In this example,
    #boxes[1][2] would contain points whith x values above 30 but below 60 and y values above 60 but below 90.

    boxes = []
    for i in range(int((XMAX-XMIN)/bound + 1)):
        boxes.append([])
        for j in range(int((YMAX-YMIN)/bound + 1)):
            boxes[i].append([])

    for i in point_data:
        xIndex = math.floor((i[0] - XMIN)/bound)
        yIndex = math.floor((i[1] - YMIN)/bound)

        #Add the points and colors to boxes, match statement to get colors based on point's classification
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
            
            #If there is no x or y data, or if the box's x or y length is short it means we are on an edge piece and we should skip
            if(len(x_data) == 0 or len(y_data) == 0 or max(x_data) - min(x_data) <  (bound - 2) or max(y_data) - min(y_data) < (bound - 2)):
                continue
            
            # Print the data of the current box
            # print('point count: ', len(x_data))
            # print('xmin: ', min(x_data), 'xmax: ', max(x_data), 'diff: ', max(x_data) - min(x_data))
            # print('ymin: ', min(y_data), 'ymax: ', max(y_data), 'diff: ', max(y_data) - min(y_data))
            # print('zmin: ', min(z_data), 'zmax: ', max(z_data), 'diff: ', max(z_data) - min(z_data))

            # if count % 20 == 0:
            #     #draw the points using matplotlib
            #     fig = plt.figure(figsize=(5, 5))
            #     ax = fig.add_subplot(111, projection="3d")
            #     ax.scatter(x_data, y_data, z_data)
            #     ax.set_axis_off()
            #     plt.show()
            # count+=1

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

    #Add points and colors if to a list if they are within the bounds
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

    #Add points and colors to lists if they are within the bounds
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

    Upon future endeavors we found that labeling voxel grids with this strategy was extremely inefficient.
    Rather than promping users for each voxel grid, data should be creted using a shp file to denote trees
    and then processed via functions like "iterate_over_las_with_shp"
    """

    #Call iterate_over_las to get the voxels and xy coordinates
    (voxels, xy) = iterate_over_las(in_file, 30, 1)

    zipped = list(zip(voxels, xy))
    random.shuffle(zipped)
    voxels, xy = zip(*zipped)
    print(len(voxels))
    #Iterate through the voxels and prompt the user for how many trees are contained.
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

    #Call iterate_over_las to get voxel grids and xy coordinates
    (voxels, xy) = iterate_over_las(las_file, bound, voxel_size)

    voxels_in_bounds = []

    m = 0
    total = 0
    iter = 1
    
    #For each voxel grid we iterate throught he points in the shp file and count the number of points that fall
    #Within the bounds of the voxel grid, thus counting the number of trees. (Each point represents one tree)
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

    """
    It is important to note how our shpfiles have been organized. Essentially, each point has an ID, if this ID is
    odd, it is the bottom left point of a bounding box. If the ID is even, it is the top right of a bounding box.
    Say point n has ID 5, then its corresponding bounding box coordinate would be point m with ID 6, point n's iD + 1
    """

    #Call iterate_over_las to get voxel grids and xy coordinates
    (voxels, xy) = iterate_over_las(las_file, bound, voxel_size)


    voxels_in_bounds = []

    writer = csv.writer(open(out_file, 'w'))

    #Create a dictionary containing shapefile points paired with their id's
    shpDict = dict()
    for feat in fiona.open(shp_file):
        shpDict[feat['properties']['id']] = feat

    count = 0
    """
    Iterate over the list of voxel grids
    For each voxel grid, add the points that fall within the voxel grid to a dictionary. Then iterate over that dictionary and
    if one point does not have its twin point, add its twin to the dictionary. This point will be out of bounds so we clamp
    its x and y values to the bounds of the voxel grid. This strategy results in small parts of trees being labeled as full 
    trees which presents problems with this data.
    """
    for i in range(len(xy)):
        if xy[i][0] >  xmin and xy[i][2] < xmax and xy[i][1] > ymin and xy[i][3] < ymax:
            count+=1

            voxels_in_bounds.append(voxels[i])

            smallDict = dict()

            #Add points within this grid to a dictionary
            for j in shpDict:
                if (shpDict[j]['properties']['xcoord'] > xy[i][0] and shpDict[j]['properties']['xcoord'] < xy[i][2] and 
                    shpDict[j]['properties']['ycoord'] > xy[i][1] and shpDict[j]['properties']['ycoord'] < xy[i][3]):
                    smallDict[j] = (shpDict[j]['properties']['xcoord'], shpDict[j]['properties']['ycoord'])
            
            # if len(smallDict) == 0:
            #     continue
            newPoints = []
            idxs = []

            #Here is where we add points to the dictionary if their twin is within the dictionary. x and y values must be clamped
            for j in smallDict:
                tempTuple = (0, 0)
                if j % 2 == 1:
                    if j+1 not in smallDict:
                        if shpDict[j+1]['properties']['xcoord'] > xy[i][0] and shpDict[j+1]['properties']['xcoord'] < xy[i][2]:
                            tempTuple = (shpDict[j+1]['properties']['xcoord'], tempTuple[1])
                        else:
                            tempTuple = (xy[i][2], tempTuple[1])
                        
                        if shpDict[j+1]['properties']['ycoord'] > xy[i][1] and shpDict[j+1]['properties']['ycoord'] < xy[i][3]:
                            tempTuple = (tempTuple[0], shpDict[j+1]['properties']['ycoord'])
                        else:
                            tempTuple = (tempTuple[0], xy[i][3])
                        newPoints.append(tempTuple)
                        idxs.append(j+1)
                else:
                    if j-1 not in smallDict:
                        if shpDict[j-1]['properties']['xcoord'] > xy[i][0] and shpDict[j-1]['properties']['xcoord'] < xy[i][2]:
                            tempTuple = (shpDict[j-1]['properties']['xcoord'], tempTuple[1])
                        else:
                            tempTuple = (xy[i][0], tempTuple[1])
                        
                        if shpDict[j-1]['properties']['ycoord'] > xy[i][1] and shpDict[j-1]['properties']['ycoord'] < xy[i][3]:
                            tempTuple = (tempTuple[0], shpDict[j-1]['properties']['ycoord'])
                        else:
                            tempTuple = (tempTuple[0], xy[i][1])
                        newPoints.append(tempTuple)
                        idxs.append(j-1)

            #Here all the twins that were not in the original dictionary are added
            for j in range(len(newPoints)):
                smallDict[idxs[j]] = newPoints[j]
            idxs = []

            #Finally all of the points are sorted by ID and the bounding boxes are placed into a list which will be used as the voxel grid's label
            for j in smallDict:
                idxs.append(j)
            idxs.sort()
            boundingCoord = []
            for j in idxs:
                if j % 2 == 0: continue
                boundingCoord.append([max(0, min(bound-1, int(smallDict[j][0] - xy[i][0]))), max(0, min(bound-1, int(smallDict[j][1] - xy[i][1]))),
                                      max(0, min(bound-1, int(smallDict[j+1][0] - xy[i][0]))), max(0, min(bound-1, int(smallDict[j+1][1] - xy[i][1])))])

            for j in boundingCoord:
                if (j[2] - j[0]) * (j[3] - j[1]) < 8:
                    boundingCoord.remove(j)

            boundingCoord.sort(key=lambda x: (x[0], x[1]))
            colors = voxelization.fill_voxel_grid(voxels[i], bound, height, voxel_size)[1]
            row = []
            #First we write the voxel grid to the csv file
            for i in colors:
                if i[0] == 0:
                    row.append(0)
                else:
                    row.append(1)

            #Then we write the bounding coordinates to the end of the voxel grid
            for j in boundingCoord:
                row.append(j)
            writer.writerow(row)

            # v2 = dt.create_bounding_box(row, boundingCoord, bound, height, voxel_size)
            # voxelization.visualize_voxel_grid(v2)

def segment_over_las_with_shp_selective(las_file : str, shp_file : str, out_file : str, xmin : float, ymin : float, xmax : float, ymax : float, bound : int, height : int, voxel_size : float):

    """
    Given a .las file and a .shp file this function iterates over the las file and writes voxel grids to a csv file labeled with their corresponding
    bounding boxes gathered from the .shp file. Used for getting labeled data for segmentation. The difference between this and 
    segment_over_las_with_shp() is that this function will only include bounding boxes for trees who's majority falls within the
    voxel grid. This way the data is of better quality as a small section of a tree will not be labeled.
    """

    (voxels, xy) = iterate_over_las(las_file, bound, voxel_size)
    
    voxels_in_bounds = []

    writer = csv.writer(open(out_file, 'w'))

    #Create a dictionary containing shapefile points paired with their id's
    shpDict = dict()
    for feat in fiona.open(shp_file):
        shpDict[feat['properties']['id']] = feat

    print([shpDict[i]['properties']['xcoord'], shpDict[i]['properties']['ycoord']] for i in shpDict)
    count = 0
    #Iterate over the list of voxel grids
    for i in range(len(xy)):
        if xy[i][0] >  xmin and xy[i][2] < xmax and xy[i][1] > ymin and xy[i][3] < ymax:
            count+=1

            voxels_in_bounds.append(voxels[i])

            smallDict = dict()


            newPoints = []
            idxs = []
            tuple1 = (0,0)
            tuple2 = (0,0)

            #Here we find what twin points must be added to the shp dictionary, add them and clamp their x, y values to the voxel grid's
            for j in shpDict:
                if j % 2 == 1:
                    tuple1 = (0, 0)
                    if shpDict[j]['properties']['xcoord'] > xy[i][0] and shpDict[j]['properties']['xcoord'] < xy[i][2]:
                        tuple1 = (shpDict[j]['properties']['xcoord'], tuple1[1])
                    else:
                        if shpDict[j]['properties']['xcoord'] < xy[i][0]:
                            tuple1 = (xy[i][0], tuple1[1])
                        else:
                            tuple1 = (xy[i][2], tuple1[1])
                    if shpDict[j]['properties']['ycoord'] > xy[i][1] and shpDict[j]['properties']['ycoord'] < xy[i][3]:
                        tuple1 = (tuple1[0], shpDict[j]['properties']['ycoord'])
                    else:
                        if shpDict[j]['properties']['ycoord'] < xy[i][1]:
                            tuple1 = (tuple1[0], xy[i][1])
                        else:
                            tuple1 = (tuple1[0], xy[i][3])
                else:
                    tuple2 = (0, 0)
                    if shpDict[j]['properties']['xcoord'] > xy[i][0] and shpDict[j]['properties']['xcoord'] < xy[i][2]:
                        tuple2 = (shpDict[j]['properties']['xcoord'], tuple2[1])
                    else:
                        if shpDict[j]['properties']['xcoord'] < xy[i][0]:
                            tuple2 = (xy[i][0], tuple2[1])
                        else:
                            tuple2 = (xy[i][2], tuple2[1])
                    
                    if shpDict[j]['properties']['ycoord'] > xy[i][1] and shpDict[j]['properties']['ycoord'] < xy[i][3]:
                        tuple2 = (tuple2[0], shpDict[j]['properties']['ycoord'])
                    else:
                        if shpDict[j]['properties']['ycoord'] < xy[i][1]:
                            tuple2 = (tuple2[0], xy[i][1])
                        else:
                            tuple2 = (tuple2[0], xy[i][3])

                    """
                    Here we decide what points to eliminate by comparing their clamped labels to the real labels and eliminate
                    clamped labels if the IoU is less than .5. This helps us eliminate labels for trees that are only partially
                    within the voxel grid
                    """
                    if ((tuple2[0] - tuple1[0]) * (tuple2[1] - tuple1[1])) / ((shpDict[j]['properties']['xcoord'] - shpDict[j-1]['properties']['xcoord']) * (shpDict[j]['properties']['ycoord'] - shpDict[j-1]['properties']['ycoord']))  > .5:
                        newPoints.append(tuple1)
                        newPoints.append(tuple2)
                        idxs.append(j-1)
                        idxs.append(j)

            #Here the points are added to the dictionary
            for j in range(len(newPoints)):
                smallDict[idxs[j]] = newPoints[j]
            idxs = []

            #each point's ID's are added to a list which is then iterated over to collect the correct bounding boxes
            for j in smallDict:
                idxs.append(j)
            idxs.sort()
            boundingCoord = []
            for j in idxs:
                if j % 2 == 0: continue
                boundingCoord.append([max(0, min(bound-1, int(smallDict[j][0] - xy[i][0]))), max(0, min(bound-1, int(smallDict[j][1] - xy[i][1]))),
                                      max(0, min(bound-1, int(smallDict[j+1][0] - xy[i][0]))), max(0, min(bound-1, int(smallDict[j+1][1] - xy[i][1])))])
                # boundingCoord.append((max(0, min(bound-1, int(smallDict[j][0] - xy[i][0]))), max(0, min(bound-1, int(smallDict[j][1] - xy[i][1])))))

            #We also eliminate bounding boxes that are small, in this case the boxes with an area less than 8
            for j in boundingCoord:
                if (j[2] - j[0]) * (j[3] - j[1]) < 8:
                    boundingCoord.remove(j)

            #Here we sort the bounding boxes by the x value of their lower left corner first, then the y value of their lower left corner
            boundingCoord.sort(key=lambda x: (x[0], x[1]))
            colors = voxelization.fill_voxel_grid(voxels[i], bound, height, voxel_size)[1]
            row = []

            #Finally the voxel grid is created and added to the given csv file along with the corresponding labels
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

def view_as_pointcloud_with_bounds(xmin : float, ymin : float, bound : int, las_file : str, shp_file : str):
    """
    This function will collect the points that fall within a 30x30 collumn originating from the xmin and y min,
    and visualize the point cloud
    """

    xmax = xmin + bound
    ymax = ymin + bound
    las = laspy.read(las_file)

    point_data = np.stack([las.x, las.y, las.z, las.classification], axis = 0).transpose((1,0))

    x_data = []
    y_data = []
    z_data = []

    for i in point_data:
        if i[0] > xmin and i[0] <  xmax and i[1] > ymin and i[1] < ymax:
            x_data.append(i[0])
            y_data.append(i[1])
            z_data.append(i[2])


    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_data, y_data, z_data)
    ax.set_axis_off()
    plt.show()

def get_pointclouds_with_shp(las_file : str, bound : int, shp_file : str, xmin : float, ymin : float, xmax : float, ymax : float):
    """
    This function divides the passed las file into point clouds of size bound x bound, and organize the saving location
    to be in the same format as ISBNet requires the s3dis data to be in. Essentially the goal is to treat our data the same way 
    that s3dis data is treated in order to make training on our data compatible with ISBNet's code
    """

    #Load LiDAR data
    las = laspy.read(las_file)
    print(las)

    #Recording the data in a numpy array
    point_data = np.stack([las.x, las.y, las.z, las.classification], axis = 0).transpose((1,0))
    print(point_data)

    XMIN = las.header.mins[0]
    XMAX = las.header.maxs[0]
    YMIN = las.header.mins[1]
    YMAX = las.header.maxs[1]
    print(XMIN)

    print('looping...')

    shpDict = dict()
    for feat in fiona.open(shp_file):
        shpDict[feat['properties']['id']] = feat

    #Create the boxes as described in iterate_over_las()
    boxes = []
    for i in range(int((XMAX-XMIN)/bound + 1)):
        boxes.append([])
        for j in range(int((YMAX-YMIN)/bound + 1)):
            boxes[i].append([])

    for i in point_data:
        if i[0] < xmin or i[0] > xmax or i[1] < ymin or i[1] > ymax:
            continue
        xIndex = math.floor((i[0] - XMIN)/bound)
        yIndex = math.floor((i[1] - YMIN)/bound) 
        boxes[xIndex][yIndex].append([i[0], i[1], i[2], i[3]])

    gridCount = 1
    for i in boxes:
        for j in i:
            if len(j) == 0:
                continue
            d = np.zeros((len(j), 6)) #MAKE Nx6 and color by class, green for vegetation, black for anything else (X,Y,Z,R,G,B)
            for idx, k in enumerate(j):
                d[idx][0] = k[0]
                d[idx][1] = k[1]
                d[idx][2] = k[2]
                if k[3] == 4 or k[3] == 5:
                    d[idx][3] = 0
                    d[idx][4] = 128
                    d[idx][5] = 0
                else:
                    d[idx][3] = 0
                    d[idx][4] = 0
                    d[idx][5] = 0
            
            #If the point cloud is not shaped correctly it is a edge piece and we skip
            if max(d[:, 0]) - min(d[:, 0]) < bound-5 or max(d[:, 1]) - min(d[:, 1]) < bound-5:
                continue

            xy = [min(d[:, 0]), min(d[:, 1]), max(d[:, 0]), max(d[:, 1])]

            smallDict = dict()
            for m in shpDict:
                if (shpDict[m]['properties']['xcoord'] > xy[0] and shpDict[m]['properties']['xcoord'] < xy[2] and 
                    shpDict[m]['properties']['ycoord'] > xy[1] and shpDict[m]['properties']['ycoord'] < xy[3]):
                    smallDict[m] = (shpDict[m]['properties']['xcoord'], shpDict[m]['properties']['ycoord'])
            
            #Upon adding the correct shp file points to smallDict, shpHelper is called to change smallDict to contain the correct points
            shpHelper(shpDict, smallDict, xy)
            
            #Next, points that fall within the bounds to twin shp file points are labeled as a tree instance within the 
            #annotations folder
            treeCount = 1
            for m in smallDict:
                if m % 2 == 0: continue
                instance = np.zeros_like(d)
                idx2 = 0
                for n in d:
                    if n[0] > smallDict[m][0] and n[1] > smallDict[m][1] and n[0] < smallDict[m+1][0] and n[1] < smallDict[m+1][1]:
                        instance[idx2][0] = n[0]
                        instance[idx2][1] = n[1]
                        instance[idx2][2] = n[2]
                        instance[idx2][3] = n[3]
                        instance[idx2][4] = n[4]
                        instance[idx2][5] = n[5]
                        idx2+=1

                zero_rows_mask = np.all(instance[:, -6:] == 0, axis=1)
                instance = instance[~zero_rows_mask]
                if len(instance) == 0 or len(instance) == 1: continue
                #Points are "normalized" by subtracting the minimum value of each dimension from the corresponding dimensions
                #(otherwise x,y,z values would be as they are listed in the las file)
                instance[:, 0] -= min(d[:, 0])
                instance[:, 1] -= min(d[:, 1])
                instance[:, 2] -= min(d[:, 2])
                os.makedirs('GISData/pointCloudGrids/Area_2/grid_' + str(gridCount) + '/annotations', exist_ok=True)
                path = os.path.join('GISData/pointCloudGrids/Area_2/grid_' + str(gridCount) + '/annotations/tree_' + str(treeCount) + '.txt')
                np.savetxt(path, instance, fmt='%.6f')
                treeCount+=1

            
            d[:, 0] -= min(d[:, 0])
            d[:, 1] -= min(d[:, 1])
            d[:, 2] -= min(d[:, 2])

            #Finally the main pointcloud is written to the correct filepath
            os.makedirs('GISData/pointCloudGrids/Area_2/grid_' + str(gridCount), exist_ok=True)
            path = os.path.join('GISData/pointCloudGrids/Area_2/grid_' + str(gridCount), 'grid_' + str(gridCount) + '.txt')
            gridCount+=1
            # print(path)
            np.savetxt(path, d, fmt='%.6f')

            if len(smallDict) > 4:
                print(len(smallDict))
                print(xy)
                print(sorted(list(smallDict.keys())))
                v = viz.Visualizer()

                v.add_points('test', d[:, :3], d[:, -3:], point_size=60, visible=False)
                v.save('uhh')
def shpHelper(shpDict : dict, smallDict : dict, xy : list):
    """
    This function adjusts the passed smallDict to contain the correct bounds for each tree instance within the point cloud bounds
    """
    keys = list(smallDict.keys())
    for j in keys:
        tempTuple=(0,0)
        if j % 2 == 1:
            if j+1 not in smallDict:
                if shpDict[j+1]['properties']['xcoord'] > xy[0] and shpDict[j+1]['properties']['xcoord'] < xy[2]:
                    tempTuple = (shpDict[j+1]['properties']['xcoord'], tempTuple[1])
                else:
                    tempTuple = (xy[2], tempTuple[1])
                
                if shpDict[j+1]['properties']['ycoord'] > xy[1] and shpDict[j+1]['properties']['ycoord'] < xy[3]:
                    tempTuple = (tempTuple[0], shpDict[j+1]['properties']['ycoord'])
                else:
                    tempTuple = (tempTuple[0], xy[3])
                smallDict[j+1] = tempTuple
        else:
            if j-1 not in smallDict:
                if shpDict[j-1]['properties']['xcoord'] > xy[0] and shpDict[j-1]['properties']['xcoord'] < xy[2]:
                    tempTuple = (shpDict[j-1]['properties']['xcoord'], tempTuple[1])
                else:
                    tempTuple = (xy[0], tempTuple[1])
            
                if shpDict[j-1]['properties']['ycoord'] > xy[1] and shpDict[j-1]['properties']['ycoord'] < xy[3]:
                    tempTuple = (tempTuple[0], shpDict[j-1]['properties']['ycoord'])
                else:
                    tempTuple = (tempTuple[0], xy[1])
                smallDict[j-1] = tempTuple