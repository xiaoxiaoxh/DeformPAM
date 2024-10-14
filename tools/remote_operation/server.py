import socket
import pickle
import numpy as np
import open3d as o3d
from loguru import logger
from typing import Tuple, List, Optional
from tools.remote_operation.utils import send_int, recv_bool, recv_err, send_array, recv_array

class RemoteManualOperationServer:
    def __init__(self,
                 host: str = '127.0.0.1',
                 port: int = 12000,
                 max_data_size: int = 12800,
                 timeout: int = 30,
                 debug: bool = False,
                 ):
        self.host = host
        self.port = port
        self.max_data_size = max_data_size
        self.timeout = timeout
        self.debug = debug
        self.socket, self.conn = None, None
        self.set_up_server()
        self.listen()
        
    def set_up_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.settimeout(self.timeout)
    
    def listen(self):
        self.socket.listen(1)
        logger.info(f'[Remote Manual Operation Server] listening on {self.host}:{self.port}...')
        self.conn, self.addr = self.socket.accept()
        self.conn.settimeout(self.timeout)
        logger.info(f'[Remote Manual Operation Server] connected by {self.addr[0]}:{self.addr[1]}')
        
    def close_connection(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            
    def send_pcd_and_recv_organized(self, points: np.ndarray, colors: np.ndarray):
        send_array(points, np.float64, self.conn, self.max_data_size)
        send_array(colors, np.float64, self.conn, self.max_data_size)
        organized = recv_bool(self.conn, self.max_data_size, debug=self.debug)
        return organized
        
    def send_pcd_and_recv_anno(self, points: np.ndarray, colors: np.ndarray, n: int):
        send_array(points, np.float64, self.conn, self.max_data_size)
        send_array(colors, np.float64, self.conn, self.max_data_size)
        send_int(n, self.conn, self.max_data_size)
        
        pts = recv_array(np.float64, (-1, 3), self.conn, self.max_data_size, debug=self.debug)
        pts_sel = recv_array(np.int32, (-1), self.conn, self.max_data_size, debug=self.debug)
        pts_sel = pts_sel.tolist()
        err = recv_err(self.conn, self.max_data_size, debug=self.debug)
        
        return pts, pts_sel, err
    
    
def remote_display_pcd_and_get_organized(point_cloud: o3d.geometry.PointCloud,
                       host: str = '127.0.0.1',
                       port: int = 12000,
                       debug: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[Exception]]:
    while True:
        try:
            server = None
            server = RemoteManualOperationServer(
                host=host,
                port=port,
                debug=debug
            )
            points = np.asarray(point_cloud.points)
            colors = np.asarray(point_cloud.colors)
            organized = server.send_pcd_and_recv_organized(points=points, colors=colors)
            return organized
        except Exception as e:
            logger.error(f'error when conducting remote display: {e}, continue...')
        finally:
            if server is not None:
                server.close_connection()

def remote_pick_n_points_from_pcd(point_cloud: o3d.geometry.PointCloud,
                                  n: int,
                                  host: str = '127.0.0.1',
                                  port: int = 13000,
                                  debug: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[Exception]]:
    # no while loop here, because the caller should handle the exception
    try:
        server = None
        server = RemoteManualOperationServer(
            host=host,
            port=port,
            debug=debug
        )
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        pts, pts_sel, err = server.send_pcd_and_recv_anno(points=points, colors=colors, n=n)
        return pts, pts_sel, err
    except Exception as e:
        logger.error(f'error when conducting remote manual operation: {e}')
        return None, None, e
    finally:
        if server is not None:
            server.close_connection()

if __name__ == '__main__':
    with open('/home/xuehan/DeformPAM/outputs/2024-03-26/16-16-30/pcd.pickle', 'rb') as f:
        points = pickle.load(f)
        logger.debug(f'points shape: {points.shape}')
    colors = points.copy()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    remote_display_pcd_and_get_organized(point_cloud, debug=True)
    remote_pick_n_points_from_pcd(point_cloud, 2, debug=True)