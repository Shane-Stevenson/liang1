import open3d as o3d
import csv

def voxel_to_csv(out_location : str, colors : list):
    with open(out_location, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([j for sub in colors for j in sub])


