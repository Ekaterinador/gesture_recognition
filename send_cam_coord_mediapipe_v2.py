#!/usr/bin/env python3
from geometry_msgs import msg
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist, Vector3
from std_msgs.msg import String, Float64MultiArray
from std_msgs.msg import ColorRGBA
import tf
import numpy as np
import random
from std_msgs.msg import String, Int8, Header
import time
import serial
import cv2
import mediapipe as mp
import numpy as np

pub = rospy.Publisher('/mediapipe', Float64MultiArray, queue_size=10)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )

if __name__ == '__main__':
    print('start')
    cap = cv2.VideoCapture(0)
    rospy.init_node('camera', anonymous=True)
    # [head, neck, r shoulder, r elbow, r wrist, l shoulder, l elbow, l wrist, back]
    indexes = [[0], [12, 11], [12], [14], [16], [11], [13], [15], [24, 23]]
    rate = rospy.Rate(10)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            try:
                data = []
                msg = Float64MultiArray()
                for idx in indexes:
                    if len(idx) == 1:
                        pose = results.pose_landmarks.landmark[idx[0]]
                        data.extend([pose.x, pose.y, pose.z])
                    else:
                        pose1, pose2 = results.pose_landmarks.landmark[idx[0]], results.pose_landmarks.landmark[idx[1]]
                        data.extend([pose1.x + (pose1.x - pose2.x) / 2,
                                    pose1.y + (pose1.y - pose2.y) / 2,
                                    pose1.z + (pose1.z - pose2.z) / 2])
                print(data)
                msg.data = data
                pub.publish(msg)

            except:
                pass

    cap.release()
    cv2.destroyAllWindows()
