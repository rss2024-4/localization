from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32

from vs_msgs.msg import ParkingError

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
        # self.declare_parameter('num_beams_per_particle', "default")
        
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value
        # self.num_beams_per_particle = self.get_parameter('num_beams_per_particle').get_parameter_value().integer_value

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
        self.num_beams_per_particle = self.sensor_model.num_beams_per_particle

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


        # lock
        self.lock = False

        # test in sim
        # self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        # def timer_cb():
        #     signal = AckermannDriveStamped()
        #     signal.drive.speed = 1.0
        #     signal.drive.steering_angle = 0.1
        #     self.drive_pub.publish(signal)
        # self.timer = self.create_timer(1., timer_cb)

        # publish viz/test stuff
        self.error_pub = self.create_publisher(ParkingError, "/pf/error", 1)
        # self.th_error_pub = self.create_publisher(Float32, "pf/th_error", 1)
        self.pose_arr_pub = self.create_publisher(PoseArray, "pf/particles", 1)
        self.timer = self.create_timer(0.05, self.timer_cb)
        self.best_guess = [0,0,0]

        
    
    def laser_callback(self, scan):
        """
        Update particle probabilities and resamples based on odometry data
        
        args
            odom: sensor_msgs/LaserScan message
        """
        if self.lock == False:
            if self.particles is None:
                self.get_logger().info("no particles from sensor")
                return
            self.get_logger().info("sensor running")
            # downsample lidar to correct number of beams, evenly spaced 
            observation = np.array(scan.ranges)
            mask = (np.linspace(0, len(observation)-1, self.sensor_model.num_beams_per_particle)).astype(int)
            observation_downsampled = observation[mask]
            
            # recalculate probabilities using sensor model
            self.particle_probabilities = self.sensor_model.evaluate(self.particles, observation_downsampled)
            
            # resample particles based on new probabilities
            res = np.random.choice(self.N_PARTICLES, self.N_PARTICLES, True, self.normalize(self.particle_probabilities))
            self.particles = self.particles[res]
            self.particle_probabilities = self.particle_probabilities[res]
            self.particle_probabilities /= np.sum(self.particle_probabilities) # Is normalization needed? No individual particles have individual probabilities of existing
            
            self.publish_average_pose()

            
            self.lock = True
        
        
    
    def odom_callback(self, odom):
        """
        Update particles based on odometry data
        
        args
            odom: nav_msgs/Odometry message
        """
        self.get_logger().info("outside lock: motion running")
        if self.lock == True:
            if self.particles is None:
                self.get_logger().info("no particles from odom")
                return
            
            self.get_logger().info("motion running")
            time = odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9
            
            if self.last_time is None:
                self.last_time = time # only if previous odometry data exists
                return
                
            dt = time - self.last_time
            dx = odom.twist.twist.linear.x * dt
            dy = odom.twist.twist.linear.y * dt
            dtheta = odom.twist.twist.angular.z * dt # theta is rotation around z
            self.particles = self.motion_model.evaluate(self.particles, [dx, dy, dtheta])
            
            self.publish_average_pose()
            self.last_time = time

            self.lock = False

        # publish errors
        gt = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            euler_from_quaternion([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w])[2],
        ])
        # xy_msg = Float32()
        # xy_msg.data = np.linalg.norm(gt[:2] - self.best_guess[:2])
        # self.xy_error_pub.publish(xy_msg)
        # th_msg = Float32()
        # th_msg.data = np.sqrt((gt[2]%(2*np.pi) - self.best_guess[2]%(2*np.pi))**2)
        # self.th_error_pub.publish(th_msg)
        
        error_msg = ParkingError()

        error_msg.x_error = np.linalg.norm(gt[:2] - self.best_guess[:2])
        error_msg.y_error = np.sqrt((gt[2]%(2*np.pi) - self.best_guess[2]%(2*np.pi))**2)
        self.error_pub.publish(error_msg)

        
    def pose_callback(self, init_pose):
        """
        Initializes particles based on pose guess
        
        args
            init_pose: geometry_msgs/PoseWithCovarianceStamped message
        """
        self.get_logger().info("Initialized")
        x = init_pose.pose.pose.position.x
        y = init_pose.pose.pose.position.y
        o = init_pose.pose.pose.orientation
        theta = euler_from_quaternion([o.x, o.y, o.z, o.w])[2]
        
        pose_guess = np.array([[x, y, theta]])
        self.particles = np.repeat(pose_guess, self.N_PARTICLES, axis=0)
        
        self.particle_probabilities = np.ones(self.N_PARTICLES) / self.N_PARTICLES
        
        
    def publish_average_pose(self):
        """
        Publish particle filter 'average' guess as nav_msgs/Odometry message
        """
        # avg_pose = self.particles[np.argmax(self.particle_probabilities)]
        avg_pose = np.average(self.particles, weights=self.particle_probabilities, axis=0)
        self.best_guess = avg_pose
        
        msg = Odometry()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = avg_pose[0]
        msg.pose.pose.position.y = avg_pose[1]
        
        quaternion = quaternion_from_euler(0, 0, avg_pose[2])
        
        msg.pose.pose.orientation.x = quaternion[0]
        msg.pose.pose.orientation.y = quaternion[1]
        msg.pose.pose.orientation.z = quaternion[2]
        msg.pose.pose.orientation.w = quaternion[3]
        
        self.odom_pub.publish(msg)


    def timer_cb(self):
        # particle poses
        msg = PoseArray()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        arr = []
        if self.particles is not None:
            for i in self.particles:
                arr.append(self.to_pose_msg(i))
            msg.poses = arr
            self.pose_arr_pub.publish(msg)


    def to_pose_msg(self, pose_vec):
        # pose_vec is [x,y,th]
        msg = Pose()
        msg.position.x = pose_vec[0]
        msg.position.y = pose_vec[1]
        quaternion = quaternion_from_euler(0, 0, pose_vec[2])
        msg.orientation.x = quaternion[0]
        msg.orientation.y = quaternion[1]
        msg.orientation.z = quaternion[2]
        msg.orientation.w = quaternion[3]
        return msg

    def normalize(self, arr):
        return arr / np.sum(arr)
        

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
