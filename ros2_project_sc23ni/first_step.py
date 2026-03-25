import threading
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal


class colourIdentifier(Node):
    def __init__(self):
        super().__init__('cI')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.callback,
            10
        )

        self.subscription

    def callback(self, data):
        try:
            # topic is rgb8
            cv_image = self.bridge.imgmsg_to_cv2(data, 'rgb8')

            # OpenCV displays better in BGR
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            self.get_logger().info('Image received')

            cv2.namedWindow('camera_Feed', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('camera_Feed', 320, 240)
            cv2.imshow('camera_Feed', cv_image)
            cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')


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