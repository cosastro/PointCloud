# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d
import pandas
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, method):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    data = point_cloud.values
    min_d = data.min(axis=0)
    max_d = data.max(axis=0)

    D = (max_d - min_d) / leaf_size

    point_x , point_y, point_z = np.array(point_cloud.x), np.array(point_cloud.y), np.array(point_cloud.z)
    h_x, h_y, h_z = np.floor((point_x-min_d[0]) / leaf_size), np.floor((point_y-min_d[1]) / leaf_size), np.floor((point_z-min_d[2]) / leaf_size)

    h = np.array(np.floor(h_x + h_y * D[0] + h_z * D[0] * D[1]))

    # 按行连接两个矩阵 data的数据是个四维的
    data = np.c_[h, point_x, point_y, point_z]
    data = data[data[:, 0].argsort()]
    # 屏蔽结束

    if method == 'random':
        filtered_points = []
        for i in range(data.shape[0] - 1):
            if data[i][0] != data[i+1][0]:
                filtered_points.append(data[i][1:])
        filtered_points.append(data[data.shape[0]-1][1:])

    if method == 'centroid':
        filtered_points = []
        data_points = []
        for i in range(data.shape[0] - 1):
            if data[i][0] == data[i+1][0]:
                data_points.append(data[i][1:])
                continue
            if data_points == []:
                continue
            filtered_points.append(np.mean(data_points, axis=0))
            data_points = []
        filtered_points = np.array(filtered_points)

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    points = np.genfromtxt("../modelnet40_normal_resampled/car/car_0005.txt", delimiter=",")
    points = pandas.DataFrame(points[:, 0:3])
    points.columns = ['x', 'y', 'z']
    point_cloud_pynt = PyntCloud(points)

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 10.0, method='random' )
    print(filtered_cloud)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
