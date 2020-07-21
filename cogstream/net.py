import struct

from cogstream.protocol import ProtocolError


def send_message(sock, msg):
    payload = msg.serialize() + '\n'
    payload = payload.encode('UTF-8')
    send_packet(sock, payload)


def recv_message(sock, msg_cls):
    msg = recv_packet(sock)

    if not msg:
        raise ProtocolError('Unexpected EOF')

    msg = msg.decode('UTF-8').strip()

    if msg.startswith('error'):
        raise ProtocolError(msg)

    return msg_cls.parse(msg)


def send_packet(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def recv(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def recv_packet(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recv(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recv(sock, msglen)
