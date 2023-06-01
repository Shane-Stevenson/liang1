import geopandas as gpd
import open3d as o3d
from tools.lasTest import iterate_over_las
import tools.voxelize as voxelize
import tools.data as data
import AI.ai1 as ai1

print('bruh')

# data.createTrainingData('full_tree_voxels.csv', 'full_not_tree_voxels.csv', 'train.csv', 597)

# ai1.trainAndTest('train.csv')


init_voxels = iterate_over_las('data/initData.las', 10)
# for i in init
# size = len(init_voxels)
# print(size)

# count = 0
# c = 0
# for i in init_voxels:
#     if count % 10 == 0:
#         if c == 597:
#             break
#         c+=1
#         full_voxel_colors = voxelize.fill_voxel_grid(i, .25, 10)
#         data.voxel_to_csv('full_tree_voxels.csv', full_voxel_colors)
#         if count % 100 == 0:
#             print(count)
#             print(count/size)

#     count+=1
# print(count)

