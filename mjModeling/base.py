from startup import mujoco, os
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

        # 2. Find the robot's end-effector body (usually 'link7')
        # You may need to verify the name in your specific XML
        parent_body = spec.body("link7")
        if parent_body is None:
            raise ValueError("Could not find 'link7' in the model. Check your XML body names.")

        # 3. Add a new child body for the blade
        blade_body = parent_body.add_body(name="blade_attachment")
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

        print(f"Blade successfully attached to {parent_body.name}")
 
    @property
    def data(self):
        return self._data
        
            