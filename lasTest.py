import laspy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from voxelize import voxelize

def iterate_over_las(las_file : str, square_length : int):

    #Load LiDAR data
    las = laspy.read(las_file)
    print(las)

    #Recording the data in a numpy array
    point_data = np.stack([las.x, las.y, las.z], axis = 0).transpose((1,0))
    print(point_data)


    XMIN = 671790.69
    XMAX = 673277.34
    YMIN = 3677052.9 
    YMAX = 3678811.21
        
    print('looping...')

    #initialize a 222x262 2D array
    boxes = []
    for i in range(222):
        boxes.append([])
        for j in range(262):
            boxes[i].append([])

    """
    The data is 1486 units long and 1758 units high. We place each point in its corresponding x-Box by dividing the difference of the 
    point's x-position and the minimum x value by square_length. This process is repeated for the y value and then the point is appended
    to boxes
    """
    for i in point_data:
        xIndex = math.floor((i[0] - XMIN)/square_length)
        yIndex = math.floor((i[1] - YMIN)/square_length)
        boxes[xIndex][yIndex].append([i[0], i[1], i[2]])

    #Initialize x,y,z lists
    x_data = []
    y_data = []
    z_data = []
    for i in boxes:
        for j in i:
            #Clear the lists when moving to a new box
            x_data.clear()
            y_data.clear()
            z_data.clear()
            for k in j:
                #fill the lists with all the values in the current box
                x_data.append(k[0])
                y_data.append(k[1])
                z_data.append(k[2])
            
            #If there is no x or y data, or if the boxe's x or y length is short it means we are on an edge piece and we should skip
            if(len(x_data) == 0 or len(y_data) == 0 or max(x_data) - min(x_data) < 8 or max(y_data) - min(y_data) < 8):
                continue
            
            #Print the data of the current box
            print('point count: ', len(x_data))
            print('xmin: ', min(x_data), 'xmax: ', max(x_data), 'diff: ', max(x_data) - min(x_data))
            print('ymin: ', min(y_data), 'ymax: ', max(y_data), 'diff: ', max(y_data) - min(y_data))
            print('zmin: ', min(z_data), 'zmax: ', max(z_data), 'diff: ', max(z_data) - min(z_data))

            #draw the points using matplotlib
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x_data, y_data, z_data)
            ax.set_axis_off()
            plt.show()

            #voxelize
            points = []
            xmin = min(x_data)
            ymin = min(y_data)
            zmin = min(z_data)
            for i in range(len(x_data)):
                points.append([x_data[i] - xmin, y_data[i] - ymin, z_data[i] - zmin])
            voxelize(points, .25)