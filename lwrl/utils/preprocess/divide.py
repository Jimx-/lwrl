from lwrl.utils.preprocess import Preprocessor


class Divide(Preprocessor):
    def __init__(self, scale):
        self.scale = scale

    def process(self, variable):
        return variable / self.scale
