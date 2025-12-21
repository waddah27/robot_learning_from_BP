from startup import mujoco, os
# Get the directory where your Python script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full absolute path to the STL
stl_rel_path = "scalpel_model/scalpel/scalpelHandler1.STL"
stl_abs_path = os.path.join(base_dir, stl_rel_path)
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

        # 1. Load the existing model into a spec
        spec = mujoco.MjSpec.from_file(xml_path)
        # 1. Register the Meshes in Assets
        # 'file' should be the path to your STL files
        scalpel_handler_mesh = spec.add_mesh(name="handler_mesh", file=stl_abs_path)
        scalpel_handler_mesh.refquat = [0.707, -0.707, 0, 0]
        scalpel_handler_mesh.scale = [0.001, 0.001, 0.001] 
        # scalpel_mesh = spec.add_mesh(name="scalpel_mesh", file="scalpel.stl")
 
        # 2. Find the robot's end-effector body (usually 'link7')
        # You may need to verify the name in your specific XML
        ee_body = spec.body("link7")
        if ee_body is None:
            raise ValueError("Could not find 'link7' in the model. Check your XML body names.")

        # 3. Add a new child body for the blade
        handler = ee_body.add_body(name="3d_printed_handler")
        # Visual: Add the complex STL mesh
        handler.add_geom(name="handler_visual", type=mujoco.mjtGeom.mjGEOM_MESH, 
                     meshname="handler_mesh", rgba=[0.9, 0.9, 0.9, 1]) # White plastic
    
        handler.pos = [-0.05, 0.05, 0.0] 
        if False:
            blade_body = ee_body.add_body(name="blade_attachment")
            blade_body.pos = [0, 0, 0.05]  # Offset from the end of link7

            # 4. Add the actual geometry (the blade)
            # Using a thin box as a placeholder for a blade
            blade_geom = blade_body.add_geom(
            name="blade_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.001, 0.005, 0.1],  # [width, depth, length]
            rgba=[0.8, 0.8, 0.8, 1]   # Metallic grey
            )

        # 5. Compile the modified spec into the simulation model
        self._model = spec.compile()
        self._data = mujoco.MjData(self._model)
        
        # Initial physics forward pass
        mujoco.mj_kinematics(self._model, self._data)    
        mujoco.mj_forward(self._model, self._data)

        print(f"Blade successfully attached to {ee_body.name}")
 
    @property
    def data(self):
        return self._data
        
            