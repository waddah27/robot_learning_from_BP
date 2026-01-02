
class Robot:
    def __init__(self):
        self._model = None
        self._data = None
    
    @property
    def model(self):
        return self._model
    
    def create(self, xml_path):
        raise NotImplementedError()
    
    def add_scalpel(self):
        raise NotImplementedError()
    
    def add_material(self):
        raise NotImplementedError()
    
    def run_cutting_experiment(self):
        raise NotImplementedError()
 
 
    @property
    def data(self):
        return self._data
        
            