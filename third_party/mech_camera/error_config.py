# Error description for camera
# Author: Mingshuo Han

from lib_py.base.error_base import ErrorBase

cam_code = ErrorBase("camera")
cam_code.ok = 0

# factory related
cam_code.not_created = 101
cam_code.closing_wrong_camera = 102
cam_code.cam_not_defined_in_json = 103
cam_code.value_not_in_map = 104
cam_code.no_intrinsic = 105
cam_code.wrong_data_type = 106
cam_code.wrong_switch_strategy = 107
cam_code.not_supported = 108
cam_code.get_data_not_supported = 109
cam_code.read_config_error = 110
cam_code.call_api_not_supported = 111

# OpenCV camera related
cam_code.opencv_camera_no_camera_found = 200
cam_code.opencv_camera_no_such_device = 201
cam_code.opencv_camera_not_opened = 202
cam_code.opencv_camera_setting_error = 203
cam_code.opencv_camera_not_created = 204
cam_code.opencv_camera_select_timeout_error = 220

# Basler camera related
cam_code.basler_camera_no_camera_found = 300
cam_code.basler_camera_no_such_device = 301
cam_code.basler_camera_not_opened = 302
cam_code.basler_camera_not_created = 304
cam_code.basler_camera_not_grabbing = 305
cam_code.basler_camera_restart_failed = 320

# RealSense Camera related
cam_code.realsense_camera_no_camera_found = 400
cam_code.realsense_camera_no_such_device = 401
cam_code.realsense_camera_not_created = 402
cam_code.realsense_camera_advanced_mode_not_support = 405

# HikVision Camera related
cam_code.hikvision_camera_no_camera_found = 500
cam_code.hikvision_camera_no_such_device = 501
cam_code.hikvision_camera_not_created = 504
cam_code.hikvision_camera_enumerate_device_failed = 517
cam_code.hikvision_camera_cannot_create_handle = 505
cam_code.hikvision_camera_cannot_open_device = 506
cam_code.hikvision_camera_cannot_load_feature = 507
cam_code.hikvision_camera_cannot_set_packet_size = 508
cam_code.hikvision_camera_cannot_get_packet_size = 509
cam_code.hikvision_camera_cannot_set_trigger_mode = 510
cam_code.hikvision_camera_cannot_get_payload_size = 511
cam_code.hikvision_camera_cannot_start_grabbing = 512
cam_code.hikvision_camera_cannot_stop_grabbing = 513
cam_code.hikvision_camera_cannot_close_device = 514
cam_code.hikvision_camera_cannot_get_frame = 515
cam_code.hikvsion_camera_cannot_convert_pixel_type = 516
cam_code.hikvision_camera_not_grabbing = 517
cam_code.hikvision_camera_cannot_convert_pixel_format = 518
cam_code.hikvision_camera_cannot_get_shape = 520
cam_code.hikvision_camera_not_support_camera_type = 521

# RVBust Camera related
cam_code.rvbust_camera_no_camera_found = 600
cam_code.rvbust_camera_wrong_cam_selected = 601
cam_code.rvbust_camera_not_valid = 602
cam_code.rvbust_camera_open_failed = 603
cam_code.rvbust_camera_get_intrinsics_failed = 604
cam_code.rvbust_camera_capture_failed = 605
cam_code.rvbust_camera_no_such_camera_type = 606
cam_code.rvbust_camera_auto_white_balance_failed = 607
cam_code.rvbust_camera_not_created = 608
cam_code.rvbust_camera_X1_not_support_both_camera = 609
cam_code.rvbust_camera_set_bandwidth_failed = 610
cam_code.rvbust_camera_set_transform_failed = 611

# MechEye Camera related
cam_code.mecheye_camera_no_camera_found = 1000
cam_code.mecheye_camera_no_such_device = 1001
cam_code.mecheye_camera_not_opened = 1002
cam_code.mecheye_camera_parameter_not_supported = 1003
cam_code.mecheye_internal_error = 1004

# DKam Camera related
cam_code.dkam_camera_no_camera_found = 701
cam_code.dkam_camera_sort_fail = 702
cam_code.dkam_camera_no_such_device = 703
cam_code.dkam_camera_connection_fail = 704
cam_code.dkam_camera_set_trigger_mode_fail = 705
cam_code.dkam_camera_set_trigger_mode_fail = 706
cam_code.dkam_camera_start_stream_fail = 707
cam_code.dkam_camera_start_acquisition_fail = 708
cam_code.dkam_camera_stop_stream_fail = 709
cam_code.dkam_camera_stop_acquisition_fail = 710
cam_code.dkam_camera_instance_not_created = 711
cam_code.dkam_camera_is_not_grabbing = 712
cam_code.dkam_camera_get_intrinsic_error = 713
cam_code.dkam_camera_fetch_rgb_data_failed = 714
cam_code.dkam_camera_fetch_point_data_failed = 715
cam_code.dkam_camera_destroy_camera_fail = 716
cam_code.dkam_camera_disconnect_fail = 717
cam_code.dkam_camera_fetch_gray_data_failed = 718
cam_code.dkam_camera_load_config_failed = 719
cam_code.dkam_camera_force_ip_failed = 720
cam_code.dkam_camera_ip_unrechable = 721
cam_code.dkam_camera_filter_failed = 722
cam_code.dkam_camera_get_extrinsic_error = 723
cam_code.dkam_camera_get_status_fail = 724
cam_code.dkam_camera_get_cloud_unit_failed = 725

# Orbbec Camera related
cam_code.orbbec_camera_no_camera_found = 801
cam_code.orbbec_camera_no_such_device = 802
cam_code.orbbec_camera_not_opened = 803
cam_code.orbbec_camera_cannot_get_frame = 804
cam_code.orbbec_camera_cannot_convert_to_rgb_image = 805
cam_code.orbbec_camera_get_intrinsic_failed = 806
cam_code.orbbec_camera_wrong_property = 807
cam_code.orbbec_camera_property_not_supported = 808
cam_code.orbbec_camera_frame_sync_not_supported = 809
cam_code.orbbec_camera_wrong_profile_setting = 810
cam_code.orbbec_camera_failed_to_get_rgb_frame = 811
cam_code.orbbec_camera_failed_to_get_depth_frame = 812
cam_code.orbbec_camera_not_created = 813
