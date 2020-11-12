import logging
import socket
import threading
import time
from queue import Queue

import jsonpickle
from PIL import Image

from cogstream.engine import StreamType
from cogstream.net import recv_packet, send_packet, send_message, recv_message
from cogstream.protocol import StartMessage, FormatMessage, TransformResponseMessage, serialize_image

logger = logging.getLogger(__name__)


class Client:
    def __init__(self, address, engine) -> None:
        super().__init__()
        self.address = address
        self.sock = None
        self.handshake = False
        self.engine = engine

    def open(self):
        if self.sock is not None:
            raise ValueError('already connected')

        address = self.address
        logger.info('connecting to server at %s', address)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(address)

        try:
            # client: startstream ...
            start_msg = StartMessage(StreamType.frames, self.engine)
            logger.info('sending: %s', start_msg)
            send_message(sock, start_msg)

            # server: ok w=<width> h=<height> ...
            format_msg = recv_message(sock, FormatMessage)
            logger.info('received: %s', format_msg)

            # client: ok you|me
            # TODO: check whether we can do transformation locally
            transform_msg = TransformResponseMessage(server_side=True)
            logger.info('sending: %s', transform_msg)
            send_message(sock, transform_msg)
        except:
            logger.exception('error during handshake')
            sock.close()
            return

        logger.info('client-server handshake successful')
        self.sock = sock
        self.handshake = True

    def close(self):
        if self.sock is None:
            return

        self.sock.close()

    def request(self, frame):
        payload = serialize_image(frame)
        send_packet(self.sock, payload)
        result = recv_packet(self.sock)

        # TODO: deserialize the result properly
        result = jsonpickle.decode(result.decode('UTF-8'))

        return result


class ThreadedProcessor:

    def __init__(self, client, threads=1) -> None:
        super().__init__()
        self.source = Queue()
        self.sink = Queue()
        self.client = client
        self.num_threads = threads
        self.threads = None

    def start(self):
        self.threads = [threading.Thread(target=self.run) for _ in range(self.num_threads)]

        for t in self.threads:
            t.start()

    def shutdown(self, timeout=None):
        for _ in self.threads:
            self.source.put_nowait((-1, None))

        for t in self.threads:
            try:
                t.join(timeout)
            except:
                logger.warning('timeout joining worker thread')

    def run(self):
        logger.info('starting processor ...')
        source = self.source
        client = self.client
        sink = self.sink

        if logger.isEnabledFor(logging.DEBUG):
            while True:
                then = time.time()
                i, frame = source.get()
                if i == -1:
                    logger.debug('poison received, processor returning')
                    break

                response = client.request(Image.fromarray(frame))
                logger.debug('%010d: %s (%.2fms) (qsize=%d)', i, response, ((time.time() - then) * 1000),
                             source.qsize())
                sink.put(response)
        else:
            while True:
                i, frame = source.get()
                if i == -1:
                    break

                response = client.request(Image.fromarray(frame))
                sink.put(response)
