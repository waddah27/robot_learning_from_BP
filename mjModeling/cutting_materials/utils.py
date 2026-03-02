

from mjModeling.conf import MATERIAL_GEOM
from mjModeling.kuka_iiwa_14.iiwa14_model import iiwa14

__all__ = ["get_material_geometry"]

def get_material_geometry(robot: iiwa14):
    """
    get the material geometry from robot XML scene.
    returns:
    mat_center(float): the material center
    mat_size(np.array): the material 3d dims
    surface_height(float): surface height of the material
    """
    mat_id = robot.model.geom(MATERIAL_GEOM).id
    mat_center = robot.model.geom_pos[mat_id].copy()
    mat_size = robot.model.geom_size[mat_id]
    surface_height = mat_center[2] + mat_size[2]
    return mat_center, mat_size, surface_height