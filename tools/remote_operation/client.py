import argparse
import socket
import numpy as np
import open3d as o3d
from loguru import logger
from tools.remote_operation.utils import recv_int, send_bool, send_err, send_array, recv_array
from common.pcd_utils import pick_n_points_from_pcd

import py_cli_interaction
from rich.console import Console

class RemoteManualOperationClient():
    def __init__(self,
                 host: str = '127.0.0.1',
                 display_port: int = 12000,
                 anno_port: int = 13000,
                 max_data_size: int = 12800,
                 timeout: int = 300,
                 debug: bool = False,
                 ):
        self.host = host
        self.display_port = display_port
        self.anno_port = anno_port
        self.max_data_size = max_data_size
        self.timeout = timeout
        self.debug = debug
        self.socket = None
    
    def connect_to_server(self, mode='display'):
        host = self.host
        port = self.display_port if mode == 'display' else self.anno_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        self.socket.settimeout(self.timeout)
        logger.info(f'[Remote Manual Operation Client] connected to {host}:{port}') 
    
    def close_connection(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
            
    def recv_pcd_and_check_organized(self):
        Console().print(
                    "[instruction] Check if the object is organized")
        points = recv_array(np.float64, (-1, 3), self.socket, self.max_data_size, debug=self.debug)
        colors = recv_array(np.float64, (-1, 3), self.socket, self.max_data_size, debug=self.debug)
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([point_cloud])
        
        organized = py_cli_interaction.must_parse_cli_bool("organized enough?", default_value=False)
        send_bool(organized, self.socket, self.max_data_size)
        
    
    def recv_pcd_and_do_anno(self):
        Console().print(
                    "[instruction] Select ideal poses in order")
        
        points = recv_array(np.float64, (-1, 3), self.socket, self.max_data_size, debug=self.debug)
        colors = recv_array(np.float64, (-1, 3), self.socket, self.max_data_size, debug=self.debug)
        n = recv_int(self.socket, self.max_data_size, debug=self.debug)
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        pts, pts_sel, err = pick_n_points_from_pcd(point_cloud, n)
        pts = np.array(pts)
        pts_sel = np.array(pts_sel)

        send_array(pts, np.float64, self.socket, self.max_data_size)
        send_array(pts_sel, np.int32, self.socket, self.max_data_size)
        send_err(err, self.socket, self.max_data_size)
            
    def start_client(self):
        mode = 'display'
        while True:
            try:
                self.connect_to_server(mode=mode)
                if mode == 'display':
                    self.recv_pcd_and_check_organized()
                elif mode == 'anno':
                    self.recv_pcd_and_do_anno()
            except ConnectionRefusedError:
                continue
            except ConnectionResetError:
                continue
            except KeyboardInterrupt:
                logger.info('[Remote Manual Operation Client] exit')
                break
            finally:
                self.close_connection()
                mode = 'anno' if mode == 'display' else 'display' # change mode
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1', help='host ip')
    parser.add_argument('--display_port', type=int, default=12000, help='display port number')
    parser.add_argument('--anno_port', type=int, default=13000, help='annotation port number')
    parser.add_argument('--max_data_size', type=int, default=12800, help='max data size')
    parser.add_argument('--timeout', type=int, default=300, help='timeout')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()
    client = RemoteManualOperationClient(
        host=args.host,
        display_port=args.display_port,
        anno_port=args.anno_port,
        max_data_size=args.max_data_size,
        timeout=args.timeout,
        debug=args.debug,
    )
    client.start_client()
