# -----------------------------------------------------------------------------------
# Company: TensorSense
# Project: LaserGaze
# File: GazeProcessor.py
# Description: This class processes video input to detect facial landmarks and estimate
#              gaze vectors using MediaPipe. The gaze estimation results are asynchronously
#              output via a callback function. This class leverages advanced facial
#              recognition and affine transformation to map detected landmarks into a
#              3D model space, enabling precise gaze vector calculation.
# Author: Sergey Kuldin
# -----------------------------------------------------------------------------------

import mediapipe as mp
import numpy as np
import cv2
import time
from landmarks import *
from face_model import *
from AffineTransformer import AffineTransformer
from EyeballDetector import EyeballDetector

model_path = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear


class GazeProcessor:
    def __init__(self, camera_idx=0, callback=None, visualization_options=None):
        self.camera_idx = camera_idx
        self.callback = callback
        self.vis_options = visualization_options
        self.left_detector = EyeballDetector(DEFAULT_LEFT_EYE_CENTER_MODEL)
        self.right_detector = EyeballDetector(DEFAULT_RIGHT_EYE_CENTER_MODEL)
        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO
        )

    async def start(self):
        with FaceLandmarker.create_from_options(self.options) as landmarker:
            cap = cv2.VideoCapture(self.camera_idx)

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                timestamp_ms = int(time.time() * 1000)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=frame)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.face_landmarks:
                    lms_s = np.array([[lm.x, lm.y, lm.z]
                                     for lm in result.face_landmarks[0]])
                    lms_2 = (lms_s[:, :2] * [frame.shape[1],
                             frame.shape[0]]).round().astype(int)

                    mp_hor_pts = [lms_s[i] for i in OUTER_HEAD_POINTS]
                    mp_ver_pts = [lms_s[i] for i in [NOSE_BRIDGE, NOSE_TIP]]
                    at = AffineTransformer(
                        lms_s[BASE_LANDMARKS, :], BASE_FACE_MODEL,
                        mp_hor_pts, mp_ver_pts,
                        OUTER_HEAD_POINTS_MODEL, [
                            NOSE_BRIDGE_MODEL, NOSE_TIP_MODEL]
                    )

                    # حساب EAR للرَمش
                    left_eye_indices = [33, 160, 158,
                                        133, 153, 144]  # LEFT_EYE contour
                    right_eye_indices = [362, 385, 387,
                                         263, 373, 380]  # RIGHT_EYE contour
                    left_eye = lms_2[left_eye_indices]
                    right_eye = lms_2[right_eye_indices]

                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear_avg = (left_ear + right_ear) / 2.0

                    BLINKING_THRESHOLD = 0.23
                    blinking = ear_avg < BLINKING_THRESHOLD

                    # تحديث مركز العين
                    left_pts = lms_s[LEFT_IRIS + ADJACENT_LEFT_EYELID_PART]
                    right_pts = lms_s[RIGHT_IRIS + ADJACENT_RIGHT_EYELID_PART]
                    self.left_detector.update(
                        [at.to_m2(p) for p in left_pts], timestamp_ms)
                    self.right_detector.update(
                        [at.to_m2(p) for p in right_pts], timestamp_ms)

                    left_gaze_vector, right_gaze_vector = None, None
                    if self.left_detector.center_detected:
                        left_eyeball_center = at.to_m1(
                            self.left_detector.eye_center)
                        left_pupil = lms_s[LEFT_PUPIL]
                        left_gaze_vector = left_pupil - left_eyeball_center
                        left_proj_point = left_pupil + left_gaze_vector * 5.0

                    if self.right_detector.center_detected:
                        right_eyeball_center = at.to_m1(
                            self.right_detector.eye_center)
                        right_pupil = lms_s[RIGHT_PUPIL]
                        right_gaze_vector = right_pupil - right_eyeball_center
                        right_proj_point = right_pupil + right_gaze_vector * 5.0

                    direction = ""
                    command = ""

                    if blinking:
                        direction = "BLINKING"
                        command = "STOP"
                    else:
                        if left_gaze_vector is not None and right_gaze_vector is not None:
                            avg_gaze = (
                                left_gaze_vector[:2] + right_gaze_vector[:2]) / 2
                        elif left_gaze_vector is not None:
                            avg_gaze = left_gaze_vector[:2]
                        elif right_gaze_vector is not None:
                            avg_gaze = right_gaze_vector[:2]
                        else:
                            avg_gaze = None

                        if avg_gaze is not None:
                            x, y = avg_gaze
                            if abs(x) > abs(y):
                                direction = "RIGHT" if x > 0 else "LEFT"
                            else:
                                direction = "DOWN" if y > 0 else "UP"

                            command = {
                                "UP": "GO FORWARD",
                                "DOWN": "GO BACKWARD",
                                "LEFT": "TURN LEFT",
                                "RIGHT": "TURN RIGHT"
                            }.get(direction, "")

                    display_text = f"{direction} / {command}"
                    print("🧭", display_text)
                    cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 255), 2, cv2.LINE_AA)

                    if self.vis_options and self.left_detector.center_detected and self.right_detector.center_detected:
                        p1 = relative(left_pupil[:2], frame.shape)
                        p2 = relative(left_proj_point[:2], frame.shape)
                        frame = cv2.line(
                            frame, p1, p2, self.vis_options.color, self.vis_options.line_thickness)
                        p1 = relative(right_pupil[:2], frame.shape)
                        p2 = relative(right_proj_point[:2], frame.shape)
                        frame = cv2.line(
                            frame, p1, p2, self.vis_options.color, self.vis_options.line_thickness)

                    elif self.vis_options:
                        text_location = (10, frame.shape[0] - 10)
                        cv2.putText(frame, "Calibration...", text_location,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.vis_options.color, 2)

                    if self.callback:
                        await self.callback(left_gaze_vector, right_gaze_vector)

                cv2.imshow('LaserGaze', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
