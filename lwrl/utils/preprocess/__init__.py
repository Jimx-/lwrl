from lwrl.utils.preprocess.preprocessor import Preprocessor
from lwrl.utils.preprocess.divide import Divide

preprocessor_dict = dict(
    divide=Divide,
)


def get_preprocessor(type, *args, **kwargs):
    return preprocessor_dict[type](*args, **kwargs)
