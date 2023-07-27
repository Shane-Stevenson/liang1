import geopandas as gpd
import numpy as np
import open3d as o3d
from tools import visualize_voxel_grid, get_voxelization

visualize_voxel_grid(get_voxelization(672710.3300000001, 3676072.0300000003, 30, .5, 'data/unt2.las'))
