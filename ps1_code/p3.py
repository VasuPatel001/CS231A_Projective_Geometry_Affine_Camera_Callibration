# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

'''
COMPUTE_LINE_EQUATIONS
    points - a list of two points on a line as [[x1, y1], [x2, y2]]
Returns:
    coefficients a, b, c of line ax + by + c = 0
'''
def compute_line_equation(points):
    pt1 = points[0]
    pt2 = points[1]
    #convert (x,y) points to homogenous coordinate (x,y,1)
    pt1 = [pt1[0], pt1[1], 1.0]
    pt2 = [pt2[0], pt2[1], 1.0]

    #compute line_equation (a, b, c) using cross product
    l = np.cross(pt1, pt2)
    a, b, c = l[0], l[1], l[2]
    return a, b, c

'''
COMPUTE_POINT_OF_INTERSECTION
    line1 - defined by its coefficients [a1, b1, c1]
    line2 - defined by its coefficients [a2, b2, c2]
Returns:
    point of intersection (x, y)
'''
def compute_point_of_intersection(line1, line2):
    pt = np.cross(line1, line2)
    x, y, z = pt[0], pt[1], pt[2]
    return (x/z, y/z)

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). 
            It will contain four points: two for each parallel line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''
def compute_vanishing_point(points):
    # BEGIN YOUR CODE HERE
    # ref: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    points_l1 = points[0], points[1]
    points_l2 = points[2], points[3]

    #equation of line1 & line2
    l1 = compute_line_equation(points_l1)
    l2 = compute_line_equation(points_l2)
    #vanishing point (x,y)
    x, y = compute_point_of_intersection(l1, l2)
    v = [x,y]
    return v
    # END YOUR CODE HERE

'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''
def compute_K_from_vanishing_points(vanishing_points):
    # BEGIN YOUR CODE HERE
    # form equations A.w = 0 with 4 constraints of omega (w)
    # A = np.zeros((vanishing_points.shape[0], 4), dtype=np.float32)
    A = []
    for i, point_i in enumerate(vanishing_points):
        for j, point_j in enumerate(vanishing_points):
            if i != j and j > i:
                pt_1 = [point_i[0], point_i[1], 1.0]
                pt_2 = [point_j[0], point_j[1], 1.0]
                A.append(
                    [pt_1[0] * pt_2[0] + pt_1[1] * pt_2[1],
                     pt_1[0] * pt_2[2] + pt_1[2] * pt_2[0],
                     pt_1[1] * pt_2[2] + pt_1[2] * pt_2[1],
                     pt_1[2] * pt_2[2]])

    A = np.array(A, dtype=np.float32)
    u, s, v_t = np.linalg.svd(A, full_matrices=True)
    # 4 constraints of omega (w) can be obtained as the last column of v or last row of v_transpose
    w1, w4, w5, w6 = v_t.T[:, -1]
    # form omega matrix
    w = np.array([[w1, 0., w4],
                  [0., w1, w5],
                  [w4, w5, w6]])
    # w = (K.K_transpose)^(-1)
    # K can be obtained by Cholesky factorization followed by its inverse
    K_transpose_inv = np.linalg.cholesky(w) #K_transpose_inv = L (Lower triangular matrix )
    K = np.linalg.inv(K_transpose_inv.T)
    # divide by the scaling factor
    K = K / K[-1, -1]
    return K
    # END YOUR CODE HERE

'''
COMPUTE_ANGLE_BETWEEN_PLANES
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    # BEGIN YOUR CODE HERE
    # compute vanishing line (vl_1 and vl_2) using pair of vanishing points
    vl_1 = np.array(compute_line_equation(vanishing_pair1))
    vl_2 = np.array(compute_line_equation(vanishing_pair2))
    #print(vl_1.shape, vl_2.shape)

    # compute omega inverse
    w_inv = K @ K.transpose()

    # compute angle between these two planes
    l1T_winv_l2 = vl_1.transpose() @ (w_inv @ vl_2)
    sqrt_l1T_winv_l1 = np.sqrt(vl_1.transpose() @ (w_inv @ vl_1))
    sqrt_l2T_winv_l2 = np.sqrt(vl_2.transpose() @ (w_inv @ vl_2))

    #compute angle between planes
    theta = np.arccos(l1T_winv_l2 / (sqrt_l1T_winv_l1 * sqrt_l2T_winv_l2))
    # convert the angle between planes to degrees and return
    return np.degrees(theta)
    # END YOUR CODE HERE

'''
COMPUTE_ROTATION_MATRIX_BETWEEN_CAMERAS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''
def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    # BEGIN YOUR CODE HERE
    ## estimate real-world direction vectors given vanishing points
    print("Hello:", vanishing_points1.shape)
    # first image
    d1i = []
    for v1i in vanishing_points1:
        # vanishing point (v) and 3-dimensional direction vector (d) are related as [d = K.v]
        v1i_homogeneous = np.array([v1i[0], v1i[1], 1.0])
        KinvV = np.dot(np.linalg.inv(K), v1i_homogeneous.T)
        d1i.append(KinvV / np.sqrt(
            KinvV[0] ** 2 + KinvV[1] ** 2 + KinvV[2] ** 2))  # normalize to make sure you obtain a unit vector
    d1i = np.array(d1i)

    # second image
    d2i = []
    for v2i in vanishing_points2:
        # vanishing point (v) and 3-dimensional direction vector (d) are related as [d = K.v]
        v2i_homogeneous = np.array([v2i[0], v2i[1], 1.0])
        KinvV = np.dot(np.linalg.inv(K), v2i_homogeneous.T)
        d2i.append(KinvV / np.sqrt(
            KinvV[0] ** 2 + KinvV[1] ** 2 + KinvV[2] ** 2))  # normalize to make sure you obtain a unit vector
    d2i = np.array(d2i)

    # directional vectors between image 1 and image 2 are related by a rotation (R): [d2i = R.d1i] => [R = d2i.d1i_inverse]
    R = np.dot(d2i.T, np.linalg.inv(d1i.T))
    return R
    # END YOUR CODE HERE

if __name__ == '__main__':
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[674,1826],[2456,1060],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[126,1056],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual)

    # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print()
    print("Angle between floor and box:", angle)

    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print("Rotation between two cameras:\n", rotation_matrix)
    z,y,x = mat2euler(rotation_matrix)
    print()
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))
