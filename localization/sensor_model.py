import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

from tf_transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid

import sys

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', "default")
        node.declare_parameter('scan_theta_discretization', "default")
        node.declare_parameter('scan_field_of_view', "default")
        node.declare_parameter('lidar_scale_to_map_scale', 1)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        self.z_max = 200
        self.eta = 1

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        node.get_logger().info("map topic: %s" % self.map_topic)
        node.get_logger().info("beams per particle: %s" % self.num_beams_per_particle)
        node.get_logger().info("th discretization: %s" % self.scan_theta_discretization)
        node.get_logger().info("fov: %s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        # compute table
        for z in range(self.table_width):
            for d in range(self.table_width):
                self.sensor_model_table[z, d] = self.probability(z, d, self.eta, self.z_max)
        # normalize columns
        for col in self.sensor_model_table.T:
            col /= np.sum(col)

        # with open('src/localization/localization/result.yml', 'r') as f:
        #     arr = yaml.load(f, Loader=yaml.SafeLoader)
        # self.sensor_model_table = np.array(arr)
        


    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """
        if not self.map_set:
            return
        positions = self.scan_sim.scan(particles)
        observation = np.clip(observation / (self.resolution*self.lidar_scale_to_map_scale), 0, self.z_max)
        observation = np.round(observation).astype(int)
        probs = []
        for scan in positions:
            scan = np.round(np.clip(scan / (self.resolution*self.lidar_scale_to_map_scale), 0, self.z_max)).astype(int)
            prob = np.prod([self.sensor_model_table[z, d] for z, d in zip(observation, scan)])
            probs.append(prob)
        return np.array(probs)
        
    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")

    def probability(self, z, d, eta, z_max):
        s_2 = self.sigma_hit**2
        p_hit = eta * (1/np.sqrt(2*np.pi*s_2)) * np.exp(-((z-d)**2) / (2*s_2)) if 0<=z<=z_max else 0
        p_short = (2/d) * (1 - (z/d)) if 0<=z<=d and d!=0 else 0
        p_max = 1 if z==z_max else 0
        p_rand = 1/z_max if 0<=z<=z_max else 0
        return self.alpha_hit*p_hit + self.alpha_short*p_short + self.alpha_max*p_max + self.alpha_rand*p_rand
