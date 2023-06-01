import open3d as o3d
import random
import csv

def voxel_to_csv(out_location : str, colors : list):
    row = []
    with open(out_location, 'a') as f:
        type(f)
        writer = csv.writer(f)
        for i in colors:
            if i[0] == 0:
                row.append(0)
            else:
                row.append(1)

        row.append(1) # 0 represents no tree
        writer.writerow(row)

def createTrainingData(positive_csv : str, negatice_csv : str, out_location, count : int):
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
    for row in open(negatice_csv):
        rowCount+= 1
    negative = rowCount

    with open(positive_csv, 'r') as w:
        with open(negatice_csv, 'r') as l:
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
