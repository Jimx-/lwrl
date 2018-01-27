
class Schedule:
    def value(self, t):
        raise NotImplementedError()

class LinearSchedule(Schedule):
    def __init__(self, initial=1.0, final=0.1, steps=1000000):
        self.initial = initial
        self.final = final
        self.steps = steps

    def value(self, t):
        frac = min(1.0, float(t) / self.steps)
        return self.initial + (self.final - self.initial) * frac

def get_schedule(config):
    type_dict = {
        "linear": LinearSchedule,
    }

    return type_dict[config['type']](**config['args'])
