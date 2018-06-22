import visdom
import logging


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


def window_func(f):
    def inner_func(self, label, *args, **kwargs):
        wnd = self.get_window(label)
        opts = {'title': label}
        new_wnd = f(self, wnd, opts, *args, **kwargs)
        if wnd is None:
            self.add_window(label, new_wnd)

    return inner_func


class Visualizer(metaclass=Singleton):
    def __init__(self):
        self.vis = visdom.Visdom()
        self.logger = logging.getLogger(__name__)

        startup_sec = 1
        while not self.vis.check_connection() and startup_sec > 0:
            time.sleep(0.1)
            startup_sec -= 0.1

        if not self.vis.check_connection():
            logging.error('no connection can be formed quickly')

        self.windows = dict()

    def add_window(self, label, wnd):
        self.windows[label] = wnd

    def get_window(self, label):
        return self.windows.get(label)

    @window_func
    def text(self, wnd, opts, text, append=False):
        return self.vis.text(text, opts=opts, win=wnd, append=append)

    @window_func
    def line(self, wnd, opts, X, Y, append=False, xlabel=None, ylabel=None):
        if xlabel is not None:
            opts['xlabel'] = xlabel
        if ylabel is not None:
            opts['ylabel'] = ylabel

        update = None
        if append and wnd is not None:
            update = 'append'
        return self.vis.line(X=X, Y=Y, opts=opts, win=wnd, update=update)
