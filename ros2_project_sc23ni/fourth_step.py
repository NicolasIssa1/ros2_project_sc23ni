#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
import time
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError


# ---------- Planned motion ----------
ESCAPE_TURN_SPEED = -0.8
ESCAPE_TURN_TIME = 2.1          # right ~95 deg

ESCAPE_FORWARD_SPEED = 0.30
ESCAPE_FORWARD_TIME = 24      # leave starting compartment

GREEN_TURN_SPEED = 0.8
GREEN_TURN_TIME = 2.0           # left ~90 deg toward green area

GREEN_FORWARD_SPEED = 0.28
GREEN_FORWARD_TIME = 3.2        # move toward green area

# ---------- Green → Red path (tune these on real map!) ----------
GREEN_REACHED_PAUSE = 1.5           # seconds to pause at green
LEAVE_GREEN_TURN_SPEED = -0.8       # clockwise to exit green area
LEAVE_GREEN_TURN_TIME = 3.2         # ~145 deg — adjust per map

PATH_GR_FWD1_SPEED = 0.18
PATH_GR_FWD1_TIME = 5.0             # forward segment 1 toward red

PATH_GR_TURN_SPEED = -0.8           # turn toward red zone
PATH_GR_TURN_TIME = 2.0             # ~90 deg — adjust per map

PATH_GR_FWD2_SPEED = 0.18
PATH_GR_FWD2_TIME = 5.0             # forward segment 2

RED_SEARCH_ROTATE_SPEED = 0.4       # slow rotation to scan for red
RED_SEARCH_TIMEOUT = 12.0           # seconds before fallback
RED_REACHED_PAUSE = 1.5             # seconds to pause at red

# ---------- Red → Blue path ----------
LEAVE_RED_TURN_SPEED = -0.8         # clockwise (right) to swing AWAY from wall
LEAVE_RED_TURN_TIME = 1.8           # ~80 deg right — clear the wall first

PATH_RB_FWD1_SPEED = 0.25
PATH_RB_FWD1_TIME = 3.5             # forward to pass the wall

PATH_RB_TURN_SPEED = 0.8            # counter-clockwise (left) toward blue
PATH_RB_TURN_TIME = 2.8             # ~125 deg left — face blue from clean angle

PATH_RB_FWD2_SPEED = 0.25
PATH_RB_FWD2_TIME = 4.0             # forward segment 2 toward blue

BLUE_SEARCH_ROTATE_SPEED = 0.4      # slow rotation to scan for blue
BLUE_SEARCH_TIMEOUT = 12.0          # seconds before fallback

# ---------- General motion ----------
FORWARD_SPEED = 0.25
TURN_SPEED = 0.9
SOFT_TURN_SPEED = 0.35

FRONT_STOP_DIST = 0.22
FRONT_AVOID_DIST = 0.65
FRONT_CLEAR_DIST = 0.90
SIDE_WALL_DIST = 0.38

# ---------- Target handling ----------
TARGET_STOP_DIST = 1.00
COLOUR_MIN_AREA = 220
CAMERA_FOV = 62.0


class ColourProject(Node):
    def __init__(self):
        super().__init__('coursework_project_node')

        self.bridge = CvBridge()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        # Lidar
        self.front_dist = 999.0
        self.left_dist = 999.0
        self.right_dist = 999.0
        self.scan_ranges = np.array([])
        self.scan_n = 0

        # Vision
        self.image_width = 640
        self.red_seen = False
        self.green_seen = False
        self.blue_seen = False

        self.blue_visible = False
        self.blue_cx = None
        self.blue_reachable = False

        self.green_visible = False
        self.green_cx = None
        self.green_reachable = False

        self.red_visible = False
        self.red_cx = None
        self.red_reachable = False

        # Control
        self.state = "escape_turn"
        self.state_start = time.time()
        self.target_reached = False
        self.green_done = False
        self.red_done = False

        self.get_logger().info("ColourProject node started")

    def set_state(self, new_state):
        if self.state != new_state:
            self.get_logger().info(f"State: {self.state} -> {new_state}")
            self.state = new_state
            self.state_start = time.time()

    def state_elapsed(self):
        return time.time() - self.state_start

    def publish_cmd(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)

    # ---------------- LIDAR ----------------
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)

        self.scan_ranges = ranges
        self.scan_n = len(ranges)

        n = self.scan_n

        fw = max(1, int(18 * n / 360))
        front_sector = np.concatenate([ranges[:fw], ranges[n - fw:]])
        self.front_dist = float(np.min(front_sector))

        ls = int(60 * n / 360)
        le = int(110 * n / 360)
        self.left_dist = float(np.min(ranges[ls:le]))

        rs = int(250 * n / 360)
        re = int(300 * n / 360)
        self.right_dist = float(np.min(ranges[rs:re]))

    def lidar_dist_at_angle(self, angle_deg):
        if self.scan_n == 0:
            return 999.0
        centre = int((angle_deg % 360) * self.scan_n / 360)
        spread = max(1, int(8 * self.scan_n / 360))
        indices = [(centre + i) % self.scan_n for i in range(-spread, spread + 1)]
        return float(np.min(self.scan_ranges[indices]))

    def is_path_clear_to_target(self, cx):
        if self.scan_n == 0 or cx is None:
            return False

        offset = (cx - self.image_width / 2.0) / (self.image_width / 2.0)
        angle_deg = -offset * (CAMERA_FOV / 2.0)
        dist = self.lidar_dist_at_angle(angle_deg)
        return dist > 0.75

    # ---------------- VISION ----------------
    def image_callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w, _ = image.shape
        self.image_width = w

        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        green_mask = cv2.inRange(hsv, np.array([45, 100, 100]), np.array([80, 255, 255]))
        blue_mask = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))

        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        self.red_visible = False
        self.green_visible = False
        self.blue_visible = False

        self.red_cx = None
        self.green_cx = None
        self.blue_cx = None

        self.red_reachable = False
        self.green_reachable = False
        self.blue_reachable = False

        def biggest_blob(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, None
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < COLOUR_MIN_AREA:
                return None, None
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
            else:
                cx = w // 2
            return area, cx

        red_area, red_cx = biggest_blob(red_mask)
        green_area, green_cx = biggest_blob(green_mask)
        blue_area, blue_cx = biggest_blob(blue_mask)

        if red_area is not None:
            self.red_seen = True
            self.red_visible = True
            self.red_cx = red_cx
            self.red_reachable = self.is_path_clear_to_target(red_cx)

        if green_area is not None:
            self.green_seen = True
            self.green_visible = True
            self.green_cx = green_cx
            self.green_reachable = self.is_path_clear_to_target(green_cx)

        if blue_area is not None:
            self.blue_seen = True
            self.blue_visible = True
            self.blue_cx = blue_cx
            self.blue_reachable = self.is_path_clear_to_target(blue_cx)

        # Simple display
        cv2.putText(image, f"state: {self.state}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"R:{self.red_seen} G:{self.green_seen} B:{self.blue_seen}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"front:{self.front_dist:.2f}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.green_visible:
            txt = "GREEN clear" if self.green_reachable else "GREEN blocked"
            cv2.putText(image, txt, (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.red_visible:
            txt = "RED clear" if self.red_reachable else "RED blocked"
            cv2.putText(image, txt, (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if self.blue_visible:
            txt = "BLUE clear" if self.blue_reachable else "BLUE blocked"
            cv2.putText(image, txt, (10, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("camera_feed", image)
        cv2.waitKey(1)

    # ---------------- CONTROL ----------------
    def steer_to_cx(self, cx, base_speed=0.12):
        error_x = cx - (self.image_width / 2.0)
        norm_error = error_x / (self.image_width / 2.0)
        angular = -0.8 * norm_error
        angular = max(min(angular, 0.8), -0.8)
        self.publish_cmd(base_speed, angular)

    def control_loop(self):
        if self.target_reached:
            self.stop_robot()
            return

        # Highest priority: reachable blue (only after red is done)
        if self.red_done and self.blue_visible and self.blue_reachable and self.state != "approach_blue":
            self.set_state("approach_blue")

        # 1) initial perfect path
        if self.state == "escape_turn":
            if self.state_elapsed() < ESCAPE_TURN_TIME:
                self.publish_cmd(0.0, ESCAPE_TURN_SPEED)
            else:
                self.set_state("escape_forward")
            return

        if self.state == "escape_forward":
            if self.front_dist < FRONT_STOP_DIST:
                self.set_state("planned_left_turn_to_green")
                return

            if self.state_elapsed() < ESCAPE_FORWARD_TIME:
                self.publish_cmd(ESCAPE_FORWARD_SPEED, 0.0)
            else:
                self.set_state("planned_left_turn_to_green")
            return

        # 2) deterministic turn toward green
        if self.state == "planned_left_turn_to_green":
            if self.state_elapsed() < GREEN_TURN_TIME:
                self.publish_cmd(0.0, GREEN_TURN_SPEED)
            else:
                self.set_state("go_to_green_zone")
            return

        # 3) drive toward green area
        if self.state == "go_to_green_zone":
            if self.green_visible and self.green_reachable:
                self.set_state("approach_green")
                return

            if self.front_dist < FRONT_AVOID_DIST:
                self.set_state("explore_turn")
                return

            if self.state_elapsed() < GREEN_FORWARD_TIME:
                self.publish_cmd(GREEN_FORWARD_SPEED, 0.0)
            else:
                self.set_state("explore")
            return

        # 4) approach green cube
        if self.state == "approach_green":
            if not (self.green_visible and self.green_reachable):
                self.set_state("explore")
                return

            if self.front_dist <= TARGET_STOP_DIST:
                self.stop_robot()
                self.set_state("green_reached")
                return

            self.steer_to_cx(self.green_cx, base_speed=0.10)
            return

        # ---- green → red planned path ----

        # 4b) green reached — pause
        if self.state == "green_reached":
            self.stop_robot()
            if self.state_elapsed() >= GREEN_REACHED_PAUSE:
                self.green_done = True
                self.get_logger().info("GREEN CUBE REACHED — marked done")
                self.set_state("leave_green_turn")
            return

        # 4c) leave green area — turn
        if self.state == "leave_green_turn":
            if self.state_elapsed() < LEAVE_GREEN_TURN_TIME:
                self.publish_cmd(0.0, LEAVE_GREEN_TURN_SPEED)
            else:
                self.set_state("path_to_red_1")
            return

        # 4d) path to red — forward segment 1
        if self.state == "path_to_red_1":
            if self.front_dist < FRONT_STOP_DIST:
                self.set_state("path_to_red_avoid")
                return
            if self.red_visible and self.red_reachable:
                self.set_state("approach_red")
                return
            if self.state_elapsed() < PATH_GR_FWD1_TIME:
                angular = 0.0
                if self.left_dist < SIDE_WALL_DIST:
                    angular = -SOFT_TURN_SPEED
                elif self.right_dist < SIDE_WALL_DIST:
                    angular = SOFT_TURN_SPEED
                self.publish_cmd(PATH_GR_FWD1_SPEED, angular)
            else:
                self.set_state("path_to_red_turn")
            return

        # 4e) path to red — turn toward red zone
        if self.state == "path_to_red_turn":
            if self.state_elapsed() < PATH_GR_TURN_TIME:
                self.publish_cmd(0.0, PATH_GR_TURN_SPEED)
            else:
                self.set_state("path_to_red_2")
            return

        # 4f) path to red — forward segment 2
        if self.state == "path_to_red_2":
            if self.front_dist < FRONT_STOP_DIST:
                self.set_state("path_to_red_avoid")
                return
            if self.red_visible and self.red_reachable:
                self.set_state("approach_red")
                return
            if self.state_elapsed() < PATH_GR_FWD2_TIME:
                angular = 0.0
                if self.left_dist < SIDE_WALL_DIST:
                    angular = -SOFT_TURN_SPEED
                elif self.right_dist < SIDE_WALL_DIST:
                    angular = SOFT_TURN_SPEED
                self.publish_cmd(PATH_GR_FWD2_SPEED, angular)
            else:
                self.set_state("search_red")
            return

        # 4g) search for red — slow rotation in place
        if self.state == "search_red":
            if self.red_visible and self.red_reachable:
                self.set_state("approach_red")
                return
            if self.state_elapsed() < RED_SEARCH_TIMEOUT:
                self.publish_cmd(0.0, RED_SEARCH_ROTATE_SPEED)
            else:
                self.set_state("explore")
            return

        # 4h) wall avoidance during path to red
        if self.state == "path_to_red_avoid":
            turn_dir = 1.0 if self.left_dist > self.right_dist else -1.0
            if self.front_dist > FRONT_CLEAR_DIST:
                self.set_state("search_red")
            else:
                self.publish_cmd(0.0, turn_dir * TURN_SPEED)
            return

        # 4i) approach red cube
        if self.state == "approach_red":
            if not (self.red_visible and self.red_reachable):
                if self.state_elapsed() > 3.0:
                    self.set_state("search_red")
                else:
                    self.stop_robot()
                return

            if self.front_dist <= TARGET_STOP_DIST:
                self.stop_robot()
                self.set_state("red_reached")
                return

            self.steer_to_cx(self.red_cx, base_speed=0.10)
            return

        # 4j) red reached — pause
        if self.state == "red_reached":
            self.stop_robot()
            if self.state_elapsed() >= RED_REACHED_PAUSE:
                self.red_done = True
                self.get_logger().info("RED CUBE REACHED — marked done")
                self.set_state("leave_red_turn")
            return

        # ---- red → blue planned path ----

        # 5a) leave red area — turn toward blue zone
        if self.state == "leave_red_turn":
            if self.state_elapsed() < LEAVE_RED_TURN_TIME:
                self.publish_cmd(0.0, LEAVE_RED_TURN_SPEED)
            else:
                self.set_state("path_to_blue_1")
            return

        # 5b) path to blue — forward segment 1
        if self.state == "path_to_blue_1":
            if self.front_dist < FRONT_STOP_DIST:
                self.set_state("path_to_blue_avoid")
                return
            if self.blue_visible and self.blue_reachable:
                self.set_state("approach_blue")
                return
            if self.state_elapsed() < PATH_RB_FWD1_TIME:
                angular = 0.0
                if self.left_dist < SIDE_WALL_DIST:
                    angular = -SOFT_TURN_SPEED
                elif self.right_dist < SIDE_WALL_DIST:
                    angular = SOFT_TURN_SPEED
                self.publish_cmd(PATH_RB_FWD1_SPEED, angular)
            else:
                self.set_state("path_to_blue_turn")
            return

        # 5c) path to blue — corrective turn
        if self.state == "path_to_blue_turn":
            if self.state_elapsed() < PATH_RB_TURN_TIME:
                self.publish_cmd(0.0, PATH_RB_TURN_SPEED)
            else:
                self.set_state("path_to_blue_2")
            return

        # 5d) path to blue — forward segment 2
        if self.state == "path_to_blue_2":
            if self.front_dist < FRONT_STOP_DIST:
                self.set_state("path_to_blue_avoid")
                return
            if self.blue_visible and self.blue_reachable:
                self.set_state("approach_blue")
                return
            if self.state_elapsed() < PATH_RB_FWD2_TIME:
                angular = 0.0
                if self.left_dist < SIDE_WALL_DIST:
                    angular = -SOFT_TURN_SPEED
                elif self.right_dist < SIDE_WALL_DIST:
                    angular = SOFT_TURN_SPEED
                self.publish_cmd(PATH_RB_FWD2_SPEED, angular)
            else:
                self.set_state("search_blue")
            return

        # 5e) search for blue — slow rotation in place
        if self.state == "search_blue":
            if self.blue_visible and self.blue_reachable:
                self.set_state("approach_blue")
                return
            if self.state_elapsed() < BLUE_SEARCH_TIMEOUT:
                self.publish_cmd(0.0, BLUE_SEARCH_ROTATE_SPEED)
            else:
                self.set_state("explore")
            return

        # 5f) wall avoidance during path to blue
        if self.state == "path_to_blue_avoid":
            turn_dir = 1.0 if self.left_dist > self.right_dist else -1.0
            if self.front_dist > FRONT_CLEAR_DIST:
                self.set_state("search_blue")
            else:
                self.publish_cmd(0.0, turn_dir * TURN_SPEED)
            return

        # 6) final blue approach
        if self.state == "approach_blue":
            if not (self.blue_visible and self.blue_reachable):
                self.set_state("explore")
                return

            if self.front_dist <= TARGET_STOP_DIST:
                self.target_reached = True
                self.stop_robot()
                self.get_logger().info("BLUE BOX REACHED")
                return

            if self.front_dist < FRONT_STOP_DIST:
                self.set_state("explore_turn")
                return

            # Steer toward blue, but bias right when left wall is close
            error_x = self.blue_cx - (self.image_width / 2.0)
            norm_error = error_x / (self.image_width / 2.0)
            angular = -0.8 * norm_error
            if self.left_dist < SIDE_WALL_DIST:
                angular -= 0.3  # nudge right, away from left wall
            angular = max(min(angular, 0.8), -0.8)
            self.publish_cmd(0.15, angular)
            return

        # 7) general explore
        if self.state == "explore":
            if self.red_done and self.blue_visible and self.blue_reachable:
                self.set_state("approach_blue")
                return

            if not self.green_done and self.green_visible and self.green_reachable:
                self.set_state("approach_green")
                return

            if self.green_done and not self.red_done and self.red_visible and self.red_reachable:
                self.set_state("approach_red")
                return

            if self.front_dist < FRONT_AVOID_DIST:
                self.set_state("explore_turn")
                return

            angular = 0.0
            if self.left_dist < SIDE_WALL_DIST:
                angular = -SOFT_TURN_SPEED
            elif self.right_dist < SIDE_WALL_DIST:
                angular = SOFT_TURN_SPEED

            self.publish_cmd(FORWARD_SPEED, angular)
            return

        # 7) turn away from walls
        if self.state == "explore_turn":
            turn_dir = 1.0 if self.left_dist > self.right_dist else -1.0

            if self.front_dist > FRONT_CLEAR_DIST:
                self.set_state("explore")
            else:
                self.publish_cmd(0.0, turn_dir * TURN_SPEED)
            return

    def destroy_node(self):
        self.stop_robot()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ColourProject()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()