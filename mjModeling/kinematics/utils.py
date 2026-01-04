import numpy as np
def quat_to_mat(q):
    """Convert quaternion [w, x, y, z] to rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])

def mat_to_axisangle(R):
    """Convert rotation matrix to axis-angle"""
    angle = np.arccos((np.trace(R) - 1) / 2)
    if angle < 1e-6:
        return np.zeros(3)
    
    axis = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0], 
        R[1,0] - R[0,1]
    ]) / (2 * np.sin(angle))
    
    return axis * angle