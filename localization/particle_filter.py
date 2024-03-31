from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan

from rclpy.node import Node
import rclpy

from tf_transformations import euler_from_quaternion, quaternion_from_euler

assert rclpy

import numpy as np


class ParticleFilter(Node):
    
    N_PARTICLES = 100

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.declare_parameter('num_beams_per_particle', "default")
        
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value
        self.num_beams_per_particle = self.get_parameter('num_beams_per_particle').get_parameter_value().integer_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # TODO: Publish a transformation frame between the map
        # and the particle_filter_frame.
                
        self.particles = None
        self.particle_probabilities = None
        #self.particles = np.empty((self.N_PARTICLES, 3))
        #self.particle_probabities = np.empty(self.N_PARTICLES)
        
        self.last_time = None
        
    
    def laser_callback(self, scan):
        """
        Update particle probabilities and resamples based on odometry data
        
        args
            odom: sensor_msgs/LaserScan message
        """
        if self.particles is None:
            return
        
        # downsample lidar to correct number of beams, evenly spaced 
        observation = scan.ranges
        mask = np.round(np.linspace(0, len(observation)-1, self.num_beams_per_particle))
        observation_downsampled = observation[mask]
        
        # recalculate probabilities using sensor model
        self.particle_probabilities = self.sensor_model.evaluate(self.particles, observation_downsampled)
        
        # resample particles based on new probabilities
        res = np.random.choice(self.N_PARTICLES, self.N_PARTICLES, True, self.particle_probabilities)
        self.particles = self.particles[res]
        self.particle_probabilities = self.particle_probabilities[res]
        self.particle_probabilities /= np.sum(self.particle_probabilities) # TODO: Is normalization needed?
        
        
        self.publish_average_pose()
        
        
    
    def odom_callback(self, odom):
        """
        Update particles based on odometry data
        
        args
            odom: nav_msgs/Odometry message
        """
        if self.particles is None:
            return
        
        time = odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9
        
        if self.last_time is None:
            self.last_time = time # only if previous odometry data exists
            return
            
        dt = time = self.last_time
        dx = odom.twist.twist.linear.x * dt
        dy = odom.twist.twist.linear.y * dt
        dtheta = odom.twist.twist.angular.z * dt # theta is rotation around z
        self.particles = self.motion_model.evaluate(self.particles, [dx, dy, dtheta])
        
        self.publish_average_pose()
        self.last_time = time
        
    def pose_callback(self, init_pose):
        """
        Initializes particles based on pose guess
        
        args
            init_pose: geometry_msgs/PoseWithCovarianceStamped message
        """
        x = init_pose.pose.pose.position.x
        y = init_pose.pose.pose.position.y
        o = init_pose.pose.pose.orientation
        theta = euler_from_quaternion(o.x, o.y, o.z, o.w)[2]
        
        pose_guess = np.array([[x, y, theta]])
        self.particles = np.repeat(pose_guess, self.N_PARTICLES, axis=0)
        self.particles += self.motion_model.gen_noise(self.N_PARTICLES)
        #TODO: change noise model to use covariances from pose message?
        
        self.particle_probabilities = np.ones(self.N_PARTICLES) / self.N_PARTICLES
        
        
    def publish_average_pose(self):
        """
        Publish particle filter 'average' guess as nav_msgs/Odometry message
        """
        avg_pose = self.particles[np.argmax(self.particle_probabilities)]
        #TODO: do weighted average with outlier detection
        
        msg = Odometry()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = avg_pose[0]
        msg.pose.pose.position.y = avg_pose[1]
        
        quaternion = quaternion_from_euler(0, 0, avg_pose[2])
        
        msg.pose.pose.orientation.x = quaternion.x
        msg.pose.pose.orientation.y = quaternion.y
        msg.pose.pose.orientation.z = quaternion.z
        msg.pose.pose.orientation.w = quaternion.w
        
        self.odom_pub.publish(msg)
        
        
        


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
