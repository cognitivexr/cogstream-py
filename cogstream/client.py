import logging
import socket

from cogstream.engine import StreamType
from cogstream.net import recv_packet, send_packet, send_message, recv_message
from cogstream.protocol import StartMessage, FormatMessage, TransformResponseMessage

logger = logging.getLogger(__name__)


class Client:
    def __init__(self, address) -> None:
        super().__init__()
        self.address = address
        self.sock = None
        self.handshake = False

    def open(self):
        if self.sock is not None:
            raise ValueError('already connected')

        address = self.address
        logger.info('connecting to server at %s', address)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(address)

        try:
            # client: startstream ...
            start_msg = StartMessage(StreamType.frames, 'yolo')  # TODO: pass engine
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
        send_packet(self.sock, frame)
        return recv_packet(self.sock)
