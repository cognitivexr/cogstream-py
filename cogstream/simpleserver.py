import io
import logging
import socket
import struct
import time

from PIL import Image

from cogstream.net import recv

logger = logging.getLogger(__name__)


def parse_pil_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def recv_packet(sock):
    # Read message length and unpack it into an integer
    raw_len = recv(sock, 4)
    if not raw_len:
        return None

    # little-endian signed integer
    int_len = struct.unpack('<i', raw_len)[0]

    # Read the message data
    return recv(sock, int_len)


def start_stream(conn):
    i = 0
    while True:
        then = time.time()
        try:
            buf = recv_packet(conn)
        except ConnectionResetError:
            logger.debug('stopping stream due to ConnectionResetError')
            break
        if not buf:
            logger.debug('stopping stream')
            break

        logger.debug('receiving packet with %d bytes took %.2fms', len(buf), ((time.time() - then) * 1000))

        try:
            then = time.time()
            img = parse_pil_image(buf)

            # rotate
            img = img.rotate(180)
            img.save(f'frame_{i}.png')
            i += 1
            logger.debug('parsing image took %.2fms', ((time.time() - then) * 1000))

        except:
            logger.exception('inference: exception')
            continue


def serve(address):
    logger.info('starting server on address %s', address)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(address)
    server_socket.listen(1)

    conn = None
    try:
        while True:
            logger.info('waiting for next connection')
            conn, addr = server_socket.accept()
            logger.info('initiating handshake with %s', addr)

            # TODO: multiple connections

            logger.info('client-server handshake successful, starting stream')
            start_stream(conn)

            logger.info('closing connection %s', addr)
            conn.close()
    except KeyboardInterrupt:
        pass
    finally:
        if conn:
            conn.close()

        server_socket.close()
