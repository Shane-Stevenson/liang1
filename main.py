import geopandas as gpd
import open3d as o3d
from lasTest import iterate_over_las
import voxelize
import data

print('bruh')

init_voxels = iterate_over_las('initData.las', 10)
print(len(init_voxels))

count = 0
for i in init_voxels:
    full_voxel_colors = voxelize.fill_voxel_grid(i, .25, 10)
    data.voxel_to_csv('full_voxels.csv', full_voxel_colors)
    count+=1
    if count == 10:
        break

