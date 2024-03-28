from functools import cached_property

class Strategy():
    def __init__(self) -> None:
        self.init_ = False
        
    def init(self, actions_dimensions):
        self.init_ = True
        self.actions_dimensions = actions_dimensions
        
    def check_initialization(self):
        if not self.init_:
            raise RuntimeError("Cannot call this method on an uninitialized utility strategy instance")
        
    @cached_property
    def dimensions(self):
        raise NotImplementedError()
        
    def convert(x):
        raise NotImplementedError()