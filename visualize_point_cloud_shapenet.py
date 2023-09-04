import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# File path
file_path = r'/home/uceeuam/graduation_project/datasets/shapenet/ShapeNet55/shapenet_pc/02691156-1a04e3eab45ca15dd86060f189eb133.npy'

# Load the npy file
data = np.load(file_path)
print(data[0,:])
# Extract X, Y, and Z coordinates from the point cloud
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the point cloud
ax.scatter(x, y, z, s=1, c='b')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud Visualization')

# Show the plot
plt.savefig('fig/point_test.png')