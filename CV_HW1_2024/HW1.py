import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from skimage import morphology
from sklearn.preprocessing import normalize
import scipy
import os
import re

image_row = 120
image_col = 120


# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')


# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')


# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row, image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')


# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z, filepath):
    Z_map = np.reshape(Z, (image_row, image_col)).copy()
    data = np.zeros((image_row * image_col, 3), dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd, write_ascii=True)


# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])


# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image_row, image_col = image.shape
    return image


# read images
def read_images(filepath):
    images = []
    for fn in [f'{filepath}/pic{_}.bmp' for _ in range(1, 7)]:
        images.append(read_bmp(fn).flatten())
    return np.array(images)


# read noisy images and remove the gaussian noise
def read_noisy_image(filepath):
    images = []
    pattern1 = morphology.square(4)
    pattern2 = morphology.disk(3)
    pattern3 = morphology.diamond(3)
    for fn in [f'{filepath}/pic{_}.bmp' for _ in range(1, 7)]:
        image = read_bmp(fn)

        image = morphology.erosion(image, pattern1)
        image = morphology.erosion(image, pattern2)

        image = morphology.dilation(image, pattern1)
        image = morphology.dilation(image, pattern2)

        image = morphology.erosion(image, pattern3)
        image = morphology.dilation(image, pattern3)

        images.append(image.flatten())

    return np.array(images)


# read and normalize light source
def read_light_sources(filepath):
    pattern1 = re.compile(r'\((.*)\)')  # this pattern can extract the value in the bracket
    pattern2 = re.compile(r',')  # split the values by comma
    with open(f'{filepath}/LightSource.txt', 'r') as f:
        light_sources = [list(map(float, pattern2.split(pattern1.findall(_)[0]))) for _ in f.readlines()]

    return normalize(np.array(light_sources), axis=1)


# read bmp and calculate normal vector
def calculate_normal_vector(images, light_sources):
    # calculate the normal vector
    G = np.linalg.lstsq(light_sources.T @ light_sources, light_sources.T @ images, rcond=None)[0]

    # normalize the normal vector
    G = G.T
    G = normalize(G)
    G = G.reshape((image_row, image_col, 3))
    return G


# calculate mask from image
def calculate_mask(image):
    mask = (image != 0)
    return mask.reshape((image_row, image_col))


# calculate the depth map from normal vector
def calculate_depth_map(G, mask):
    pixel_count = np.count_nonzero(mask)
    pixel_index = np.where(mask)

    ind = np.zeros((image_row, image_col), dtype=int)
    ind.fill(-1)

    for i, _ in enumerate(zip(*pixel_index)):
        ind[_[0], _[1]] = i

    V = np.zeros((2 * pixel_count, 1), dtype=np.float32)
    # M = np.zeros((2 * pixel_count, pixel_count))
    M = scipy.sparse.lil_matrix((2 * pixel_count, pixel_count), dtype=np.float32)

    for i in range(pixel_count):
        h = pixel_index[0][i]  # y
        w = pixel_index[1][i]  # x
        n = G[h, w]

        if w + 1 < image_col and mask[h, w + 1] >= 0:
            V[2 * i, 0] = -n[1] / n[2]

            M[2 * i, i] = -1
            M[2 * i, ind[h, w + 1]] = 1
#        elif w - 1 >= 0 and mask[h, w - 1] >= 0:
#            V[2 * i, 0] = -n[0] / n[2]

#            M[2 * i, i] = 1
#            M[2 * i, ind[h, w - 1]] = -1

        if h + 1 < image_row and mask[h + 1, w] >= 0:
            V[2 * i + 1, 0] = -n[0] / n[2]

            M[2 * i + 1, i] = -1
            M[2 * i + 1, ind[h + 1, w]] = 1
#        elif h - 1 >= 0 and mask[h - 1, w] >= 0:
#            V[2 * i + 1, 0] = -n[1] / n[2]

#            M[2 * i + 1, i] = -1
#            M[2 * i + 1, ind[h - 1, w]] = 1

    print('Solving the linear system...')
    MM = M.T @ M
    MV = M.T @ V
    z = scipy.sparse.linalg.spsolve(MM, MV)
    # z -= np.min(z)

    Z = np.zeros_like(mask, dtype=np.float32)

    for i in range(pixel_count):
        Z[pixel_index[0][i], pixel_index[1][i]] = z[i]

    return Z


def replace_outliers(Z):
    threshold_value = 60
    mask = Z > threshold_value
    idx = np.where(mask)

    for i in range(len(idx[0])):
        h, w = idx[0][i], idx[1][i]
        Z[h, w] = min(np.sqrt(threshold_value * Z[h, w]), 80)

    Z -= np.min(Z[Z > 0])

    return Z


if __name__ == '__main__':
    # bunny star venus noisy_venus
    filepath = 'star'

    images_path = f'./test/{filepath}'
    print('Reading images...')
    if 'noisy' in filepath:
        images = read_noisy_image(images_path)
    else:
        images = read_images(images_path)
    light_sources = read_light_sources(images_path)
    print('Reading images done.')

    mask = calculate_mask(images[0])

    G = calculate_normal_vector(images, light_sources)
    normal_visualization(G)
    # mask_visualization(mask)

    Z = calculate_depth_map(G, mask)
    if 'venus' in filepath:
        Z = replace_outliers(Z)

    depth_visualization(Z)

    if not os.path.exists('./output'):
        os.makedirs('./output')
    output_path = f'./output/{filepath}.ply'
    # save_ply(Z, output_path)
    # show_ply(output_path)

    # showing the windows of all visualization function
    plt.show()
