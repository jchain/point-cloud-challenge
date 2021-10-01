import numpy as np
from plyfile import PlyData, PlyElement
#import matplotlib.pyplot as plt


def open_ply(file_path):
    rdata = PlyData.read(file_path)
    points = []
    for i in range(len(rdata.elements[0].data)):
        point = rdata.elements[0].data[i]
        a = np.array(list(point))
        points.append(a)
    data = np.array(points)
    return data


def write_ply(name, data):
    tuples = []
    for point_i in range(data.shape[0]):
        tuples.append(tuple(data[point_i, :9]))

    described_data = np.array(
        tuples,
        dtype=[
            ("x", "double"),
            ("y", "double"),
            ("z", "double"),
            ("nx", "double"),
            ("ny", "double"),
            ("nz", "double"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    element = PlyElement.describe(described_data, "vertex")
    PlyData([element], text=False).write(name)

def filter_by_normal_vector(points1):
    '''
    Filter the outliers based on normal vector

            Parameters:
                    points1 (np.array): point cloud

            Returns:
                    points1 (np.array): a new point cloud with the outlier points colored by black.
    '''
    normal_vec1 = points1[0:, 3:6]   # extract the normal vector subarray
    mean1 = np.mean(normal_vec1, axis=0)
    std1 = np.std(normal_vec1, axis=0)

    a, b, c = std1
    outlier_num = 0

    # x, y, z forms a 3-variate distribution. sigma_x,y,z forms an ellipsoid in 3d space.
    # refer to the parametric equation of ellipsoid for details.
    for i in range(normal_vec1.shape[0]):
        point = points1[i]
        v = normal_vec1[i]

        vx, vy, vz = v - mean1
        v_norm = np.linalg.norm((vx, vy, vz))
        cos_theta = vz / v_norm
        sin_theta = np.sin(np.arccos(cos_theta))
        cos_phi = vx / (v_norm * sin_theta)
        sin_phi = vy / (v_norm * sin_theta)

        sigma_x = a * sin_theta * cos_phi
        sigma_y = b * sin_theta * sin_phi
        sigma_z = c * cos_theta

        sigma_norm_2 = 2*np.linalg.norm((sigma_x, sigma_y, sigma_z))

        if (v_norm > sigma_norm_2):     # outlier is defined as larger than 2x of sigma
            outlier_num += 1
            point[6], point[7], point[8] = 0, 0, 0

    return points1

# Use histogram to detect the global roof height range [lower, upper]. We can use the bounds to
# filter the outliers caused by the trees higher than the roof and the bushes lower than the roof.
# NOTE: a more rebust way is to consider the x-range and y-range of the roof and figure out the
# threshold that doesn't remove the small substructure on the roof.
def find_height_lower_and_upper_bound(points, thresh=300):
    '''
    Find the lower and upper bound of plausible roof height (z-axis)

            Parameters:
                    points (np.array): the *entire* roof point cloud
                    thresh (int): threshold for the noise level

            Returns:
                    height_lower (np.array): the lower bound of roof height.
                    height_upper (np.array): the upper bound of roof height.
    '''
    non_black_point_mask = np.where(np.logical_or(np.logical_or(points[:,6] > 0, points[:,7] > 0), points[:,8] > 0))
    non_black_points = points[non_black_point_mask]

    non_black_z = non_black_points[0:, 2:3]
    #non_black_x = non_black_points[0:, 0:1]
    #non_black_y = non_black_points[0:, 1:2]
    hist, bin_edges = np.histogram(non_black_z, 100)
    hist_x = np.where(hist > thresh)
    height_lower = bin_edges[hist_x[0][0]]
    height_upper = bin_edges[hist_x[0][-1]]

    return height_lower, height_upper

def main():
    points = open_ply("data/PointCloud.ply")

    # Your code goes here
    (row, col) = points.shape
    non_black_point_mask = np.where(np.logical_or(np.logical_or(points[:,6] > 0, points[:,7] > 0), points[:,8] > 0))
    non_black_points = points[non_black_point_mask]
    #write_ply("data/xxx_non_black_points.ply", non_black_points)

    black_point_mask = np.where(np.logical_and(np.logical_and(points[:,6] == 0, points[:,7] == 0), points[:,8] == 0))
    black_points = points[black_point_mask]
    #write_ply("data/xxx_black_points.ply", black_points)

    color_array = non_black_points[0:, 6:9]

    colormap = np.unique(color_array, axis=0)
    #print("number of unique color points: %d" % (colormap.shape[0]))

    color_points = {}   # dict to store the sets of single color points with integer keys
    for i in range(colormap.shape[0]):
        r, g, b = colormap[i]
        x = np.where(np.logical_and(np.logical_and(non_black_points[:,6] == r, non_black_points[:,7] == g), non_black_points[:,8] == b))
        unique_color_points = non_black_points[x]
        color_points[i] = unique_color_points
        #write_ply("data/xxx_{0}.ply".format(i), unique_color_points)

    #------------------------------------------------------
    # Step 1. Filter the outliers based on normal vector
    #------------------------------------------------------
    filtered_color_points = np.transpose(np.array([[],[],[],[],[],[],[],[],[]]))
    for key in color_points:
        color_points_id = key
        points1 = color_points[color_points_id]
        new_points1 = filter_by_normal_vector(points1)

        #write_ply("data/yyy_{0}.ply".format(color_points_id), new_points1)
        filtered_color_points = np.concatenate((filtered_color_points, new_points1), axis=0)

    #write_ply("data/yyy_all.ply", filtered_color_points)

    #-------------------------------------------------------------------------
    # Step 1. Filter the outliers based on global height lower and upper bound
    #-------------------------------------------------------------------------
    height_lower, height_upper = find_height_lower_and_upper_bound(filtered_color_points)

    for i in range(filtered_color_points.shape[0]):
        point = filtered_color_points[i]
        if (point[6] > 0 or point[7] > 0 or point[8] > 0):
            if (point[2] < height_lower or point[2] > height_upper):
                point[6], point[7], point[8] = 0, 0, 0

    #write_ply("data/yyy_all_1.ply", filtered_color_points)

    result_points = np.concatenate((filtered_color_points, black_points), axis=0)

    write_ply("data/Result.ply", result_points)
    return 0


if __name__ == "__main__":
    main()
