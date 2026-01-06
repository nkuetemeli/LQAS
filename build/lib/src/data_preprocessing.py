from utils_datasets import *


classes, train_files, train_labels, test_files, test_labels = data_parser(which_model_net='ModelNet10')


# Create a point cloud from the vertices
f = np.random.choice(np.arange(len(train_files)))
f = 2707
train_file, train_label = train_files[f], train_labels[f]
print(f'Chose label {train_label}: {classes[train_label]}, \nfile {f}: {train_file}')


# Convert the triangle mesh to a point cloud
voxel_density_tensor = transform(train_file, voxel_size=1/7, visualize='pcd')
