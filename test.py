import sys

import cv2
import numpy as np
import pyvista

res = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
res = cv2.flip(res, 0)

colors = cv2.imread(sys.argv[2])
colors = cv2.resize(colors, (640, 480))
# colors = cv2.flip(colors, 0)


def depth_map_to_point_cloud(depth_map):
    rows, cols = depth_map.shape
    points = []

    for y in range(rows):
        for x in range(cols):
            depth = depth_map[y, x]
            points.append([x, y, depth])

    return np.array(points)


# Convert depth map to point cloud
point_cloud = depth_map_to_point_cloud(res)


# xpoint_cloud = np.random.random((100, 3))
# point_cloud = np.asarray(res)
# breakpoint()
pdata = pyvista.PolyData(point_cloud, force_float=False)
# pdata.point_data.colors = np.asarray(colors)
# pdata["colors"] = colors
# pdata["orig_sphere"] = np.arange(307164)
# Compute the surface mesh from the point cloud using the Delaunay triangulation
surf = pdata.delaunay_2d(progress_bar=True)

# Load an image to use as a texture
# colors.transpose(Image.FLIP_TOP_BOTTOM).save("fubar.png")
image_rgb = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)

# Create a PyVista image object from the RGB image data
image_pv = pyvista.pyvista_ndarray(image_rgb)

# Create a texture from the PyVista image object
texture = pyvista.Texture(image_pv)

# Map the texture onto the PolyData object
surf.texture_map_to_plane(inplace=True, use_bounds=True)

######

# Plot the PolyData object with the texture
plotter = pyvista.Plotter(line_smoothing=True)
plotter.add_mesh(surf, texture=texture)
plotter.show()
