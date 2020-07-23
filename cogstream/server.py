import logging
import socket
import time

import jsonpickle

from cogstream.engine import load_engine
from cogstream.net import send_packet, recv_packet, recv_message, send_message
from cogstream.protocol import StartMessage, ProtocolError, FormatMessage, TransformResponseMessage, parse_image

logger = logging.getLogger(__name__)


def serialize_result(result):
    then = time.time()

    # TODO: serialize the result properly
    r = jsonpickle.encode(result)
    payload = r.encode('UTF-8')

    logger.debug('result serialization took %.2fms', ((time.time() - then) * 1000))

    return payload


def start_stream(conn, engine, do_transform=False):
    while True:
        then = time.time()
        buf = recv_packet(conn)
        if not buf:
            logger.debug('stopping stream')
            break

        logger.debug('receiving packet with %d bytes took %.2fms', len(buf), ((time.time() - then) * 1000))

        try:
            then = time.time()
            img = parse_image(buf)
            logger.debug('parsing image took %.2fms', ((time.time() - then) * 1000))

            result = engine.inference(img)
            logger.debug('inference: result %s', result)
        except:
            logger.exception('inference: exception')
            continue

        payload = serialize_result(result)
        send_packet(conn, payload)


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

            try:
                # client: startstream ...
                start_msg = recv_message(conn, StartMessage)
                logger.info('received: %s', start_msg)

                # server: ok w=<width> h=<height> ...
                try:
                    engine = load_engine(start_msg.engine)
                except ValueError:
                    raise ProtocolError('unknown engine %s' % start_msg.engine)

                shape = engine.shape()
                format_msg = FormatMessage(shape[0], shape[1], engine.colorspace(), engine.transformation())
                logger.info('sending: %s', format_msg)
                send_message(conn, format_msg)

                # client: ok you|me
                transform_msg = recv_message(conn, TransformResponseMessage)
                logger.info('received: %s', transform_msg)

                logger.info('client-server handshake successful, starting stream')
                start_stream(conn, engine, do_transform=transform_msg.server_side)
            except ProtocolError:
                logger.exception('protocol error, closing connection %s', addr)
                conn.close()
                continue

            logger.info('closing connection %s', addr)
            conn.close()
    except KeyboardInterrupt:
        pass
    finally:
        if conn:
            conn.close()

        server_socket.close()
