import os
import sys
import time

from multiprocessing import Process, Event
import subprocess

import cv2
import mvsdk
from loguru import logger

from third_party.mvcam.vcamera import vCameraSystem

def record_worker(save_path_base, stop_record_event, encode_record_event, delete_record_event):
    # arguments -----------------
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    CAMERA_IDX = 0 if nDev == 1 else int(input("Select camera: "))

    EXPOSURE_MS = 30
    FRAME_SPEED = 0
    TRIGGER = 0
    RESOLUTION_PRESET = 0
    IMAGE_PATTERN = 'frame_%05d.png'
    # arguments -----------------

    cam_sys = vCameraSystem(exposure_ms=EXPOSURE_MS,
                            trigger=TRIGGER,
                            resolution_preset=RESOLUTION_PRESET,
                            frame_speed=FRAME_SPEED)
    camera = cam_sys[CAMERA_IDX]

    if not os.path.exists(save_path_base):
        os.makedirs(save_path_base)

    video_idx = len([file_name for file_name in os.listdir(save_path_base) if "recording_tmp" in file_name])

    tmp_path = os.path.join(save_path_base, f"recording_tmp_{video_idx}")
    os.makedirs(tmp_path, exist_ok=True)
    for file_name in os.listdir(tmp_path):
        os.remove(os.path.join(tmp_path, file_name))

    fname = ''
    timefname = os.path.join(tmp_path, 'timestamps.txt')
    with camera as c:
        logger.info("Start recording...")
        capture_cnt = 0
        while not stop_record_event.is_set():
            try:
                start_frame_t = time.time()
                img = c.read()
                img = img[:, :, ::-1]  # RGB -> BGR

                fname = os.path.join(tmp_path, IMAGE_PATTERN % capture_cnt)
                cv2.imwrite(fname, img)
                with open(timefname, 'a') as f:
                    f.write(f"file '{fname}'\n")
                    f.write(f"duration {time.time() - start_frame_t}\n")
                capture_cnt += 1

            except Exception as e:
                logger.warning(e)
        if fname != '':
            with open(timefname, 'a') as f:
                f.write(f"file '{fname}'\n")
        logger.info("Recording finished.")

    # create video
    def create_video_from_images(tmp_path, output_file_path):
        logger.info("")
        command = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', os.path.join(tmp_path, 'timestamps.txt'),
            '-vsync', 'vfr',
            '-pix_fmt', 'yuv420p',
            output_file_path
        ]
    
        logger.info("Saving video...")
        try:
            subprocess.run(command, check=True)
            logger.info("Video created successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"An error occurred: {e}")

    if encode_record_event.is_set():
        output_video_path = os.path.join(save_path_base, f'recording_{video_idx}.mp4')
        create_video_from_images(tmp_path, output_video_path)
    
    # clean tmp folder
    if delete_record_event.is_set():
        for file_name in os.listdir(tmp_path):
            os.remove(os.path.join(tmp_path, file_name))

if __name__ == '__main__':
    stop_record_event = Event()
    encode_record_event = Event()
    delete_record_event = Event()
    process = Process(target=record_worker, args=('/home/xuehan/wendi_vis_debug', stop_record_event, encode_record_event, delete_record_event))
    process.start()
    time.sleep(15)
    encode_record_event.set()
    stop_record_event.set()
    process.join()