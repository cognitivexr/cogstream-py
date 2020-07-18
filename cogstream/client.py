import logging
import socket

from cogstream.net import recv_packet, send_packet

logger = logging.getLogger(__name__)


class Client:
    def __init__(self, address) -> None:
        super().__init__()
        self.address = address
        self.sock = None

    def open(self):
        if self.sock is not None:
            raise ValueError('already connected')

        address = self.address
        logger.info('connecting to server at %s', address)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(address)

        try:
            sock.send(b'startstream\n')
            msg = recv_packet(sock)
            logger.info('got message %s', msg)
            if msg != b'ok!':
                logger.warning('protocol error')
                sock.close()
                return
        except:
            sock.close()
            return

        logger.info('client-server handshake complete')
        self.sock = sock

    def close(self):
        if self.sock is None:
            return

        self.sock.close()

    def request(self, frame):
        send_packet(self.sock, frame)
        return recv_packet(self.sock)
