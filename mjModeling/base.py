from startup import mujoco, os
class Robot:
    def __init__(self):
        self._model = None
        self._data = None
    
    @property
    def model(self):
        return self._model
    
    def create(self, xml_path):
        if os.path.exists(xml_path):
            # self._model = mujoco.MjModel.from_xml_path(xml_path)
            spec = mujoco.MjSpec.from_file(xml_path)
            self._model = spec.compile()
            self._data = mujoco.MjData(self._model)
            mujoco.mj_kinematics(self._model, self._data)    
            mujoco.mj_forward(self._model, self._data)
        else:
            raise FileNotFoundError
    @property
    def data(self):
        return self._data
        
            