from cogstream.engine.engine import StreamType, Colorspace, Transformation, Engine
from cogstream.engine.yolo.engine import YoloEngine

name = 'engine'

__all__ = [
    'StreamType',
    'Colorspace',
    'Transformation',
    'Engine',
    'load_engine'
]

_engines = {
    'yolo': YoloEngine
}

_running = dict()


def load_engine(engine_name: str):
    global _engines, _running

    if engine_name in _running:
        return _running[engine_name]

    if engine_name in _engines:
        e = _engines[engine_name]()
        e.setup()
        _running[engine_name] = e
        return e
    else:
        raise ValueError('unknown engine %s' % engine_name)