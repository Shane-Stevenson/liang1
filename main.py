import geopandas as gpd
import open3d as o3d
import tools.lasTest as lasTest
import tools.voxelize as voxelization
import tools.data as data
import AI.ai1 as ai1

print('bruh')

v = lasTest.get_voxelization(671813.03, 3677706.3299999996, 25, 1, 'data/initData.las')

voxelization.visualize_voxel_grid(v)

full = voxelization.fill_voxel_grid(v, 1, 25)

voxelization.visualize_voxel_grid(full[0])