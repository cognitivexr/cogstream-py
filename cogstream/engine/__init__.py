from cogstream.engine.engine import StreamType, Colorspace, Transformation, Engine
from cogstream.engine.yolo.engine import YoloEngine

name = 'engine'

__all__ = [
    'StreamType',
    'Colorspace',
    'Transformation',
    'Engine',
]

_engines = {
    'yolo': YoloEngine()
}


def load_engine(engine_name: str):
    global _engines

    if engine_name in _engines:
        return _engines[engine_name]
    else:
        raise ValueError('unknown engine %s' % engine_name)
