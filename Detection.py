# Importing the libraries 

import numpy as np 
import open3d as o3d 
import struct #interpret as binary bytes data
import matplotlib.pyplot as plt 
import pandas 
import time
import glob 

global debug 

# Binary byte data into Point cloud data

size_float = 4 
list_pcd = []

file_to_open = "/home/chinmay/Desktop/3D-Detection-Pipeline/Data/test_files/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151605047769.pcd.bin" 
file_to_save = "/home/chinmay/Desktop/3D-Detection-Pipeline/Data/test_files/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151605047769.pcd"

with open(file_to_open, "rb") as f:  # open the file to read in the binary format 
	byte = f.read(size_float*4)
	while byte:
		x,y,z,intensity = struct.unpack("ffff", byte) # f stands for flot
		list_pcd.append([x,y,z])
		byte = f.read(size_float*4)

np_pcd = np.asarray(list_pcd)
pcd = o3d.geometry.PointCloud()
v3d = o3d.utility.Vector3dVector
pcd.points = v3d(np_pcd)


o3d.io.write_point_cloud(file_to_save, pcd)
o3d.visualization.draw_geometries([pcd])


# Downsampling the point cloud data
def downsample(pcd, factor = 0.2):
	downsample_pcd = pcd.voxel_down_sample(voxel_size = factor)
	return downsample_pcd

# Segmentation of the plane between inliers and outliers : using RANSAC
def ransac(pcd, iterations = 10, tolerance = 0.25):
	# Fit a plane (ground) to the 3D plane

	plane_model, inliers = pcd.segment_plane(distance_threshold = tolerance, ransac_n = 3, num_iterations= iterations)

	inlier_cloud = pcd.select_by_index(inliers)
	outlier_cloud = pcd.select_by_index(inliers, invert = True)
	outlier_cloud.paint_uniform_color([1,0,0]) # RGB Colour
	inlier_cloud.paint_uniform_color([0,0,1]) # RGB Colour
	
	return inlier_cloud, outlier_cloud

# Clustering 
def dbscan(pcd, eps = 0.45, min_points = 7, print_progress = False, debug = False):

	verbosityLevel = o3d.utility.VerbosityLevel.Warning
	if debug:
		verbosityLevel = o3d.utility.VerbosityLevel.Debug
	with o3d.utility.VerbosityContextManager(verbosityLevel) as cm:
		labels = np.array(pcd.cluster_dbscan(eps = eps, min_points = min_points, print_progress = print_progress))
		max_labels = labels.max()
		print("Number of clusters in point cloud:" + str(max_labels))

		colors = plt.get_cmap("tab20")(labels/(max_labels if max_labels>0 else 1))
		colors[labels<0] = 0
		pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])

		return pcd, labels 

def hbdscan(pcd):
	# TODO - implement this
	pass

# Put segmented point cloud data into Bounding Boxes

def oriented_bbox(pcd, labels, min_points=30, max_points= 400):
	obbs = []
	indices = pandas.Series(range(len(labels))).groupby(labels, sort = False).apply(list).tolist()


	for i in range(len(indices)):
		num_points = len(pcd.select_by_index(indices[i]).points)

		if min_points < num_points < max_points:
			sub_cloud = pcd.select_by_index(indices[i])
			obb = sub_cloud.get_axis_aligned_bounding_box()
			obb.color = (0,0,0)
			obbs.append(obb)

	print("Number of Bounding boxes:" + str(len(obbs)))

	return obbs 

pcl_file = "/home/chinmay/Desktop/3D-Detection-Pipeline/Data/test_files/UDACITY/0000000008.pcd"

# Starting the main file 
pcd = o3d.io.read_point_cloud(pcl_file)
if __debug__:
	print(len(np.asarray(pcd.points)))
	o3d.visualization.draw_geometries([pcd])
t = time.time()

# Downsampling 
downsample_factor = 0.25
downsample_pcd = downsample(pcd,downsample_factor)
if __debug__:
    print("Downsample Time", time.time() - t)
    o3d.visualization.draw_geometries([downsample_pcd])

# RANSAC
iterations = 100
tolerance = 0.3
inlier_pts, outlier_pts = ransac(downsample_pcd, iterations = iterations, tolerance = tolerance)
if __debug__:
    print("Segmentation Time", time.time() - t)
    o3d.visualization.draw_geometries([outlier_pts, inlier_pts])

# Clustering 
t = time.time()
eps = 4
min_points = 5 
print_progress = False 
outlier_pts, labels = dbscan(outlier_pts, eps=eps, min_points=min_points, print_progress=False)
if __debug__:
    print("Clustering Time", time.time() - t)
    o3d.visualization.draw_geometries([outlier_pts, inlier_pts])

# Bounding Boxes 
t = time.time()
bboxes = oriented_bbox(outlier_pts, labels)

outlier_with_bboxes = [outlier_pts]
outlier_with_bboxes.extend(bboxes)
outlier_with_bboxes.append(inlier_pts)

if __debug__:
    print("Bounding Boxes Time", time.time() - t)
    o3d.visualization.draw_geometries(outlier_with_bboxes)

