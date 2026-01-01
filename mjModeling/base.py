from startup import mujoco, os
# Get the directory where your Python script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full absolute path to the STL
scalpelHadnler1_dirname = "scalpel_model/scalpel/scalpelHandler1.STL"
scalpelHandler1_path = os.path.join(base_dir, scalpelHadnler1_dirname)
scalpelHadnler2_dirname = "scalpel_model/scalpel/scalpelHandler2.STL"
scalpelHandler2_path = os.path.join(base_dir, scalpelHadnler2_dirname)
scalpel_dirname = "scalpel_model/scalpel/Scalpel.stl"
scalpel_path = os.path.join(base_dir, scalpel_dirname)
class Robot:
    def __init__(self):
        self._model = None
        self._data = None
    
    @property
    def model(self):
        return self._model
    
    def create(self, xml_path):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML not found at {xml_path}")

        spec = mujoco.MjSpec.from_file(xml_path)

        # 1. Register Mesh 1
        mesh1 = spec.add_mesh(name="mesh1", file=scalpelHandler1_path)
        mesh1.refquat = [0.707, -0.707, 0, 0]
        mesh1.scale = [0.001, 0.001, 0.001] 

        # 2. Register Mesh 2
        mesh2 = spec.add_mesh(name="mesh2", file=scalpelHandler2_path)
        mesh2.refquat = [0.0, 1.0, 0, 0] # Match the orientation of part 1
        mesh2.scale = [0.001, 0.001, 0.001]
        # 3. Register the scalpel mesh
        mesh3 = spec.add_mesh(name="mesh3", file=scalpel_path)
        #  Rotate 60 degrees To mach scalpel being perpendicular to ee  
        mesh3.refquat = [0, 0.5, 0, 0.866] 
        mesh3.scale = [0.0001, 0.0001, 0.0001]
        # 4. Setup Body Hierarchy
        ee_body = spec.body("link7")
        attach_site = spec.site("attachment_site")
        
        handler = ee_body.add_body(name="3d_printed_handler")
        # Align handler origin to the attachment site
        handler.pos =  attach_site.pos
        handler.quat = attach_site.quat
        handler_depth = -0.09
        # 5. Add Part 1 (First half of handler)
        handler.add_geom(
            name="handler_part1_geom", 
            type=mujoco.mjtGeom.mjGEOM_MESH, 
            meshname="mesh1", 
            rgba=[0.9, 0.9, 0.0, 1], # Orange
            pos = attach_site.pos + [-0.045 ,0.045, handler_depth]
        )

        # 6. Part 2 (Second half of handler)
        # We add this to the SAME body so they are fixed together.
        handler.add_geom(
            name="handler_part2_geom", 
            type=mujoco.mjtGeom.mjGEOM_MESH, 
            meshname="mesh2", 
            rgba=[0.8, 0.8, 0.8, 1], # Grey to distinguish
            pos=attach_site.pos + [-0.0175, -0.002, 0.025] 
        )
        # 7. Scalpel geometry from rigestered scalpel mesh
        handler.add_geom(
            name="Scalpel_geom", 
            type=mujoco.mjtGeom.mjGEOM_MESH, 
            meshname="mesh3", 
            rgba=[0.8, 0.8, 0.8, 1], # Grey to distinguish
            pos=attach_site.pos #+ [-0.0175, -0.002, 0.025] 
        )

        # 8. Compile model, forward it with data
        self._model = spec.compile()
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
 
 
    @property
    def data(self):
        return self._data
        
            