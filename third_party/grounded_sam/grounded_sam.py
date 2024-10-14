import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

from typing import Optional, Tuple

class GroundedSAM:
    def __init__(self, 
                 grounding_dino_config_path: str = "data/checkpoints/GroundingDINO_SwinT_OGC.cfg.py",
                 grounding_dino_checkpoint_path: str = "data/checkpoints/groundingdino_swint_ogc.pth",
                 sam_encoder_version: str = "vit_b",
                 sam_checkpoint_path: str = "data/checkpoints/sam_vit_b_01ec64.pth",
                 classes: list = ["cloth"],
                 box_threshold: float = 0.25,
                 valid_box_range: Optional[Tuple[float, float, float, float]] = None,
                 text_threshold: float  = 0.25,
                 nms_threshold: float = 0.8,
                 use_hsv_segmentation: bool = False,
                 hsv_bounds: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None,
                 vis: bool = False) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=grounding_dino_config_path, 
                                          model_checkpoint_path=grounding_dino_checkpoint_path,
                                          device=device)

        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[sam_encoder_version](checkpoint=sam_checkpoint_path).to(torch.device(device))
        self.sam_predictor = SamPredictor(sam)

        # params
        self.classes = list(classes)
        self.box_threshold = box_threshold
        self.valid_box_range = valid_box_range
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.use_hsv_segmentation = use_hsv_segmentation
        self.hsv_bounds = hsv_bounds
        self.vis = vis

    def predict(self, image: np.ndarray) -> np.ndarray:
        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=self.classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        # filter detections by valid_box_range
        if self.valid_box_range is not None:
            print(f"Before box filtering: {len(detections.xyxy)} boxes")
            valid_idx = []
            for idx, det in enumerate(detections):
                xyxy = list(det)[0]
                if (xyxy[0] >= self.valid_box_range[0] and xyxy[1] >= self.valid_box_range[1] and
                    xyxy[2] <= self.valid_box_range[2] and xyxy[3] <= self.valid_box_range[3]):
                    valid_idx.append(idx)
            detections.xyxy = detections.xyxy[valid_idx]
            detections.confidence = detections.confidence[valid_idx]
            detections.class_id = detections.class_id[valid_idx]
            print(f"After box filtering: {len(detections.xyxy)} boxes")

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        if len(detections.__dict__) == 5:
            # for compatibility with older versions of groundingdino
            labels = [
                f"{self.classes[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _
                in detections]
        else:
            labels = [
                f"{self.classes[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _, _
                in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            self.nms_threshold
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")

        # convert detections to masks
        if self.use_hsv_segmentation:
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            lower_bound = np.array(self.hsv_bounds[0])
            upper_bound = np.array(self.hsv_bounds[1])
            
            mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            
            mask = mask.astype(bool)
            
            all_valid_area = np.zeros_like(mask, dtype=bool)
            for xyxy in detections.xyxy:
                x1, y1, x2, y2 = xyxy
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                all_valid_area[y1:y2, x1:x2] = True
                
            mask = mask & all_valid_area

            detections.mask = mask[None, ...]
        else:
            detections.mask = self.segment(
                sam_predictor=self.sam_predictor,
                image=image,
                xyxy=detections.xyxy
            )

        if self.vis:
            # annotate image with detections
            box_annotator = sv.BoxAnnotator()
            mask_annotator = sv.MaskAnnotator()
            if len(detections.__dict__) == 5:
                # for compatibility with older versions of groundingdino
                labels = [
                    f"{self.classes[class_id]} {confidence:0.2f}"
                    for _, _, confidence, class_id, _
                    in detections]
            else:
                labels = [
                    f"{self.classes[class_id]} {confidence:0.2f}"
                    for _, _, confidence, class_id, _, _
                    in detections]
            annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        if self.vis:
            # visualize the annotated grounding dino image
            cv2.imshow("groundingdino_annotated_image.jpg", annotated_frame)
            cv2.waitKey()
            # visualize the annotated grounded-sam image
            cv2.imshow("grounded_sam_annotated_image.jpg", annotated_image)
            cv2.waitKey()

        return detections.mask 

    @staticmethod
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        # Prompting SAM with detected boxes
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    
if __name__ == '__main__':
    SOURCE_IMAGE_PATH = '/home/xuehan/Desktop/PhoXiCameraCPP/ExternalCamera/Data/test0.png'

    # load image
    image = cv2.imread(SOURCE_IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    grounded_sam_model = GroundedSAM(vis=True)

    masks = grounded_sam_model.predict(image)
