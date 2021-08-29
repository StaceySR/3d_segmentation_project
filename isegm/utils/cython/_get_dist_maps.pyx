import numpy as np
cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free

ctypedef struct qnode:
    int row
    int col
    int dep
    int layer
    int orig_row
    int orig_col
    int orig_dep
#
@cython.infer_types(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def get_dist_maps(np.ndarray[np.float32_t, ndim=2, mode="c"] points,
                  int height, int width, int depth, float norm_delimeter):
    cdef np.ndarray[np.float32_t, ndim=4, mode="c"] dist_maps = \
        np.full((2, height, width, depth), 1e6, dtype=np.float32, order="C")

    cdef int *dxy = [-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 1, 0, 1, 0, 0, 0, 0, 1]
    cdef int i, j, x, y, z, dx, dy, dz
    cdef qnode v
    cdef qnode *q = <qnode *> malloc((4 * height * width * depth + 1) * sizeof(qnode))
    cdef int qhead = 0, qtail = -1
    cdef float ndist

    for i in range(points.shape[0]):
        x, y, z = round(points[i, 0]), round(points[i, 1]), round(points[i, 2])
        if x >= 0:
            qtail += 1
            q[qtail].row = x
            q[qtail].col = y
            q[qtail].dep = z
            q[qtail].orig_row = x
            q[qtail].orig_col = y
            q[qtail].orig_dep = z
            if i >= points.shape[0] / 2:
                q[qtail].layer = 1
            else:
                q[qtail].layer = 0
            dist_maps[q[qtail].layer, x, y, z] = 0

    while qtail - qhead + 1 > 0:
        v = q[qhead]
        qhead += 1

        for k in range(6):
            x = v.row + dxy[3 * k]
            y = v.col + dxy[3 * k + 1]
            z = v.dep + dxy[3 * k + 2]

            ndist = ((x - v.orig_row)/norm_delimeter) ** 2 + ((y - v.orig_col)/norm_delimeter) ** 2 + ((z - v.orig_dep)/norm_delimeter) ** 2
            if (x >= 0 and y >= 0 and z >= 0 and x < height and y < width and z < depth and
                dist_maps[v.layer, x, y, z] > ndist):
                qtail += 1
                q[qtail].orig_col = v.orig_col
                q[qtail].orig_row = v.orig_row
                q[qtail].orig_dep = v.orig_dep
                q[qtail].layer = v.layer
                q[qtail].row = x
                q[qtail].col = y
                q[qtail].dep = z
                dist_maps[v.layer, x, y, z] = ndist

    free(q)
    return dist_maps
