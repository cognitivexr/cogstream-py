import logging
import socket
import time

from cogstream.net import send_packet, recv_packet

logger = logging.getLogger(__name__)


def start_stream(conn):
    while True:
        then = time.time()
        buf = recv_packet(conn)
        if not buf:
            logger.debug('stopping stream')
            break

        logger.debug('receiving packet with %d bytes took %.2fms', len(buf), ((time.time() - then) * 1000))

        # TODO: do inference

        # TODO: send back the result
        send_packet(conn, b'ok')


def serve(address):
    logger.info('starting server on address %s', address)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(address)
    server_socket.listen(1)

    conn = None
    try:
        while True:
            logger.info('waiting for next connection')
            conn, addr = server_socket.accept()
            logger.info('initiating handshake with %s', addr)

            data = conn.recv(1024)
            if data:
                msg = data.decode('utf-8')

                if msg.startswith('startstream'):
                    try:
                        send_packet(conn, b'ok!')
                        start_stream(conn)
                    except:
                        logger.exception('exception while streaming')
                        break
                else:
                    logger.warning('protocol error')

            logger.info('closing connection %s', addr)
            conn.close()
    except KeyboardInterrupt:
        pass
    finally:
        if conn:
            conn.close()

        server_socket.close()
