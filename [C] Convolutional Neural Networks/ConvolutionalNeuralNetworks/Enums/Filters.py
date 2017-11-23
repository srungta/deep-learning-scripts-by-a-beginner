import numpy as np

SOBEL = np.array([[1 ,0, -1],[1, 0, -1],[1 ,0, -1]])

horizontal_filter = np.array([[1 ,1, 1],[0, 0, 0],[-1 ,-1, -1]])

NORMAL = np.array([[1 ,0, -1],[2, 0, -2],[1 ,0, -1]])
SCHARR = np.array([[3 ,0, -3],[10, 0, -10],[3 ,0, -3]])