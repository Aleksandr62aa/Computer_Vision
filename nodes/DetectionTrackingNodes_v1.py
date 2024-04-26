from ultralytics import YOLO
import torch
import numpy as np

from utils_local.utils import profile_time
from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement


class DetectionTrackingNodes:
    """Модуль инференса модели детекции + трекинг алгоритма"""

    def __init__(self, config) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Детекция будет производиться на {device}')

        config_yolo = config["detection_node"]
        self.model = YOLO(config_yolo["weight_pth"])
        self.model.fuse()

        self.classes = self.model.names

        self.conf = config_yolo["confidence"]
        self.iou = config_yolo["iou"]
        self.imgsz = config_yolo["imgsz"]


        self.classes_to_detect = config_yolo["classes_to_detect"]

    @profile_time
    def process(self, frame_element: FrameElement) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"DetectionTrackingNodes | Неправильный формат входного элемента {type(frame_element)}"

        frame = frame_element.frame.copy()

        track_list = self.model.track(frame, classes=self.classes_to_detect,
                                      conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                      verbose=False, persist=True, tracker="bytetrack.yaml")

        if track_list[0].boxes.id != None:
            # Получение id list
            frame_element.id_list = track_list[0].boxes.id.int().tolist()

            # Получение box list
            frame_element.tracked_xyxy = track_list[0].boxes.xyxy.int().tolist()
            frame_element.detected_xyxy = frame_element.tracked_xyxy

            # Получение object class names
            detected_cls = track_list[0].boxes.cls.int().tolist()
            frame_element.tracked_cls = [self.classes[i] for i in detected_cls]
            frame_element.detected_cls = frame_element.tracked_cls

            # Получение conf scores
            frame_element.tracked_conf = track_list[0].boxes.conf.tolist()
            frame_element.detected_conf = frame_element.tracked_conf
        else:
            frame_element.id_list = []
            frame_element.tracked_xyxy = []
            frame_element.detected_xyxy = []
            frame_element.tracked_cls = []
            frame_element.detected_cls = []
            frame_element.tracked_conf = []
            frame_element.detected_conf = []

        return frame_element