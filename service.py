#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import subprocess

class TrainingService:
    def __init__(self):
        """Initialize the training service node and subscribers."""
        rospy.init_node('training_service', anonymous=True)
        self.train_subscriber = rospy.Subscriber("/train/now", String, self.train_callback)
        self.odom_subscriber = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.loginfo("Training service initialized and listening to topics.")

        self.current_position = None

    def train_callback(self, data):
        """
        Handle the training command.
        If the message data is 'train', execute the training script.
        """
        if data.data == "train":
            rospy.loginfo("Received training command, executing q_learning.py")
            try:
                result = subprocess.run(
                    ["python3", "/home/src/local_path_planning/q_learning.py", f"{float(self.current_position[0])},{float(self.current_position[1])}", "arg2"], 
                    check=True,
                    capture_output=True,
                    text=True
                )
                rospy.loginfo("q_learning.py executed successfully.")
                rospy.loginfo(f"Output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                rospy.logerr(f"q_learning.py failed with error: {e}")
            except Exception as e:
                rospy.logerr(f"An unexpected error occurred: {e}")

    def odom_callback(self, data):
        """Handle incoming odometry messages."""
        rospy.loginfo("Received odometry data")
        self.current_position = data.pose.pose.position.x, data.pose.pose.position.y
        print(self.current_position)

    def start(self):
        """Start the ROS spinning loop."""
        rospy.spin()

if __name__ == '__main__':
    service = TrainingService()
    service.start()
