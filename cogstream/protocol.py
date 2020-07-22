"""
Protocol implementation.
"""
from abc import ABC
from typing import Dict

from PIL import Image

from cogstream.engine import StreamType, Colorspace, Transformation

'''
Basic protocol handshake:

# client: startstream (frames|video) <engine>
# server: ok w=<width> h=<height> c=<(RGB|greyscale)> t=<(scale|letterbox)>
# client: ok me
# OR client: ok you

# client/server: error [<reason>]
'''


class ProtocolError(BaseException):
    pass


class ControlMessage(ABC):
    @staticmethod
    def parse(message: str):
        raise NotImplementedError

    def serialize(self) -> str:
        raise NotImplementedError

    def __str__(self):
        return '%s%s' % (self.__class__.__name__, self.__dict__)

    def __repr__(self):
        return self.__str__()


class StartMessage(ControlMessage):

    def __init__(self, stream_type, engine) -> None:
        super().__init__()
        self.stream_type = stream_type
        self.engine = engine

    @staticmethod
    def parse(message: str):
        parts = message.strip().split(' ', 2)
        if len(parts) != 3:
            raise ProtocolError

        if parts[0] != 'startstream':
            raise ProtocolError

        try:
            stream_type = StreamType[parts[1]]
        except KeyError:
            raise ProtocolError

        return StartMessage(stream_type, parts[2])

    def serialize(self) -> str:
        return 'startstream %s %s' % (self.stream_type.name, self.engine)


class FormatMessage(ControlMessage):

    def __init__(self, width, height, colorspace=Colorspace.unknown, transformation=Transformation.unknown) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.colorspace = colorspace
        self.transformation = transformation

    @staticmethod
    def parse(message: str):
        parts = message.strip().split(' ')
        if len(parts) < 3 or len(parts) > 5:
            raise ProtocolError

        if parts[0] != 'ok':
            raise ProtocolError

        data = dict()

        for kv in parts[1:]:
            try:
                k, v = kv.split('=')
            except ValueError:
                raise ProtocolError
            data[k] = v

        try:
            width = int(data['w'])
            height = int(data['h'])
        except (KeyError, ValueError):
            raise ProtocolError

        if 'c' in data:
            try:
                colorspace = Colorspace[data['c'].lower()]
            except KeyError:
                raise ProtocolError
        else:
            colorspace = Colorspace.unknown

        if 't' in data:
            try:
                transformation = Transformation[data['t'].lower()]
            except KeyError:
                raise ProtocolError
        else:
            transformation = Transformation.unknown

        return FormatMessage(width, height, colorspace, transformation)

    def serialize(self) -> str:

        data = dict()
        data['w'] = str(self.width)
        data['h'] = str(self.height)

        if self.colorspace and self.colorspace != Colorspace.unknown:
            data['c'] = self.colorspace.name
        if self.transformation and self.transformation != Transformation.unknown:
            data['t'] = self.transformation.name

        return 'ok %s' % to_kv_string(data)


class TransformResponseMessage(ControlMessage):

    def __init__(self, server_side=False) -> None:
        self.server_side = server_side

    @staticmethod
    def parse(message: str):
        if message.strip() == 'ok you':
            return TransformResponseMessage(True)
        if message.strip() == 'ok me':
            return TransformResponseMessage(False)

        raise ProtocolError

    def serialize(self) -> str:
        if self.server_side:
            return 'ok you'
        else:
            return 'ok me'


def to_kv_string(data: Dict):
    return ' '.join(['%s=%s' % (k, v) for k, v in data.items()])


# TODO: frame stream protocol needs to be optimized (e.g. by different stream types, with an option for dynamic streams,
#  where each frame contains the frame shape, otherwise negotiate the frame format before initializing the stream).

def serialize_image(img: Image.Image):
    mode = img.mode
    w, h = img.size

    data = img.tobytes()

    prefix = ('%s,%d,%d\n' % (mode, w, h)).encode('UTF-8')

    return prefix + data


def parse_image(arr: bytes):
    header, data = arr.split(b'\n', 1)

    mode, w, h = header.decode('UTF-8').split(',')
    w = int(w)
    h = int(h)

    return Image.frombytes(mode, (w, h), data)
