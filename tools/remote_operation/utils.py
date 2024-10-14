import socket
import pickle
import numpy as np
from loguru import logger
from typing import Optional

def send_ack(conn: socket.socket) -> None:
    conn.sendall('ACK'.encode())
    
def recv_ack(conn: socket.socket, max_data_size: int, debug: bool = False) -> None:
    ack = ""
    while ack != 'ACK':
        ack = conn.recv(max_data_size).decode()
    if debug:
        logger.debug('ACK received')

def send_err(data: Optional[Exception], conn: socket.socket, max_data_size: int) -> None:
    data = pickle.dumps(data)
    conn.sendall(data)
    recv_ack(conn, max_data_size)

def recv_err(conn: socket.socket, max_data_size: int, debug: bool = False) -> Optional[Exception]:
    data = conn.recv(max_data_size)
    assert data != b'', 'data is empty'
    data = pickle.loads(data)
    send_ack(conn)
    if debug:
        logger.debug(f'bytes received, value: {data}')
    return data

def send_int(data: int, conn: socket.socket, max_data_size: int) -> None:
    conn.sendall(str(data).encode())
    recv_ack(conn, max_data_size)

def recv_int(conn: socket.socket, max_data_size: int, debug: bool = False) -> int:
    data = conn.recv(max_data_size)
    assert data != b'', 'data is empty'
    data = int(data.decode())
    send_ack(conn)
    if debug:
        logger.debug(f'int received, value: {data}')
    return data

def send_bool(data: bool, conn: socket.socket, max_data_size: int) -> None:
    conn.sendall(str(data).encode())
    recv_ack(conn, max_data_size)

def recv_bool(conn: socket.socket, max_data_size: int, debug: bool = False) -> bool:
    data = conn.recv(max_data_size)
    assert data != b'', 'data is empty'
    data = data.decode() == 'True'
    send_ack(conn)
    if debug:
        logger.debug(f'bool received, value: {data}')
    return data

def send_array(data: np.array, dtype: np.dtype, conn: socket.socket, max_data_size: int) -> None:
    data = data.astype(dtype).tobytes()
    length = len(data)
    send_int(length, conn, max_data_size)
    
    conn.sendall(data)
    recv_ack(conn, max_data_size)
    
def recv_array(dtype: np.dtype, shape: tuple, conn: socket.socket, max_data_size: int, debug: bool = False) -> np.array:
    length = recv_int(conn, max_data_size, debug=debug)
    
    buffer = bytearray()
    while len(buffer) < length:
        data = conn.recv(max_data_size)
        assert data != b'', 'data is empty'
        buffer.extend(data)
    buffer = buffer[:length]
    data = np.frombuffer(buffer, dtype=dtype).reshape(shape)
    send_ack(conn)
    if debug:
        logger.debug(f'array received, shape: {data.shape}')
    
    return data