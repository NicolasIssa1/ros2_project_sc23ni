# Exercise 3 - If green object is detected, and above a certain size, then send a message (print or use lab2)

import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal


class colourIdentifier(Node):
    def __init__(self):
        super().__init__('cI')

        # detection flag
        self.green_found = False

        # sensitivity for green detection
        self.sensitivity = 10

        # CvBridge and subscriber
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def callback(self, data):
        try:
            # Convert ROS image into OpenCV image
            image = self.bridge.imgmsg_to_cv2(data, 'rgb8')
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Convert BGR image to HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Green colour bounds
            hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
            hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

            # Filter out everything except green
            green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)

            # Apply mask to original image
            filtered_img = cv2.bitwise_and(image, image, mask=green_mask)

            # Find contours in the green mask
            contours, _ = cv2.findContours(
                green_mask,
                mode=cv2.RETR_LIST,
                method=cv2.CHAIN_APPROX_SIMPLE
            )

            # reset flag each callback
            self.green_found = False

            if len(contours) > 0:
                # Find the largest contour
                c = max(contours, key=cv2.contourArea)

                # Check if contour is large enough
                if cv2.contourArea(c) > 500:
                    self.green_found = True

                    # Moments for center
                    M = cv2.moments(c)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                    else:
                        cx, cy = 0, 0

                    # Draw a circle around the detected contour
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    center_x, center_y = int(x), int(y)
                    radius = int(radius)

                    cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

            # Print detection result
            if self.green_found:
                self.get_logger().info('Green object detected')

            # Show windows
            cv2.namedWindow('camera_Feed', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('camera_Feed', 320, 240)
            cv2.imshow('camera_Feed', image)

            cv2.namedWindow('green_Filtered', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('green_Filtered', 320, 240)
            cv2.imshow('green_Filtered', filtered_img)

            cv2.waitKey(3)

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')


# Create a node of your class in the main and ensure it stays up and running
def main():
    def signal_handler(sig, frame):
        rclpy.shutdown()

    rclpy.init(args=None)
    cI = colourIdentifier()

    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(cI,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            continue
    except ROSInterruptException:
        pass

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()