import numpy as np

class MotionModel:

    def __init__(self, node):
        self.rng = np.random.default_rng()
        x_max_deviation = 0.1
        y_max_deviation = 0.1
        th_max_deviation = np.pi/30
        self.deviation = np.array([x_max_deviation, y_max_deviation, th_max_deviation])

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size as an nparray
        """
        particles_T = list(map(self.to_T, particles))
        odom_T = self.to_T(odometry)
        new_particles = []
        for particle in particles_T:
            noise = self.normalize(self.rng.standard_normal(3))
            new_particles.append(self.from_T(particle@odom_T) + noise)
        return np.array(new_particles)

    def to_T(self, pose):
        '''
        args:
            pose: [x, y, th]
        returns:
            T: Transform rep of the pose
        '''
        x, y, th = pose[0], pose[1], pose[2]
        return np.array([
            [np.cos(th), -np.sin(th), x],
            [np.sin(th),  np.cos(th), y],
            [         0,           0, 1],
        ])
    
    def from_T(self, T):
        '''
        args:
            T: Transform rep of the pose as 3x3 np array
            noise: [x, y, th]
        returns:
            pose: [x, y, th]
        '''
        return np.array([
            T[0, 2],
            T[1, 2],
            np.arctan2(T[1, 0], T[0, 0]),
        ])

    def normalize(self, pose):
        '''
        max is assumed to be 4 std dev away from mean of 0
        args:
            pose: [x, y, th] as np array
        returns:
            clipped pose normalized between max values defined above
        '''
        return pose * self.deviation / 4
    
    def gen_noise(self, N):
        '''
        returns:
            clipped normal pose noise (N x 3)
        '''
        return self.normalize(self.rng.standard_normal((N, 3)))
    

if __name__ == '__main__':
    mm = MotionModel(None)
    particles = [[1,2,0],[1,1,0]]
    odom = [1,0,0]
    particles_tplus1 = mm.evaluate(particles, odom)
    print(particles_tplus1)
