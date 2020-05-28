import numpy as np
from matplotlib import pyplot as plt
from math import atan2
from scipy.stats import norm
from Ploter import config_plot


class Robot:
    """
    the robot class, we will use this to describe a robot
    """
    def __init__(self, world_size=100):
        """
        creating a robot object
        :param world_size: the world size in pixels
        """
        self._world_size = world_size
        # pose declaration
        self.x = np.random.rand() * self._world_size
        self.y = np.random.rand() * self._world_size
        self.theta = np.random.rand() * 2 * np.pi
        # noise declaration
        self.forward_noise = 0
        self.turn_noise = 0
        self.sense_distance_noise = 0
        self.sense_noise_bearing = 0

    def set(self, new_x, new_y, new_orientation):
        """
        setting the configuration of the robot
        :param new_x: the new x coordinate
        :param new_y: the new y coordinate
        :param new_orientation: the new orientation
        """
        if new_x < 0 or new_x >= self._world_size:
            raise Exception('X coordinate out of bound')

        if new_y < 0 or new_y >= self._world_size:
            raise Exception('Y coordinate out of bound')

        if new_orientation < 0.0 or new_orientation >= 2 * np.pi:
            Exception('Orientation must be in [0,2pi]')

        self.x = new_x
        self.y = new_y
        self.theta = new_orientation

    def print(self):
        """"
        printing the pose
        """
        print('[x= {} y={} heading={}]'.format(self.x, self.y, self.theta))

    def plot(self, mycolor="b", style="robot", show=True, markersize=1):
        """
        plotting the pose of the robot in the world
        :param mycolor: the color of the robot
        :param style: the style to plot with
        :param show: if to show or not show - used to create a new figure or not
        """
        if style == "robot":
            phi = np.linspace(0, 2 * np.pi, 101)
            r = 1
            # plot robot body
            plt.plot(self.x + r * np.cos(phi), self.y + r * np.sin(phi), color=mycolor)
            # plot heading direction
            plt.plot([self.x, self.x + r * np.cos(self.theta)], [self.y, self.y + r * np.sin(self.theta)], color=mycolor)

        elif style == "particle":
            plt.plot(self.x, self.y, '.', color=mycolor, markersize=markersize)
        else:
            print("unknown style")

        if show:
            plt.show()

    def set_noise(self, new_forward_noise, new_turn_noise, new_sense_noise_range, new_sense_noise_bearing):
        """
        setting the noise if pose of the robot
        :param new_forward_noise: the noise for moving forward
        :param new_turn_noise: the noise in the turn of the robot
        :param new_sense_noise_range: the noise in range measurement
        :param new_sense_noise_bearing: the noise in bearing measurement
        """
        self.forward_noise = new_forward_noise
        self.turn_noise = new_turn_noise
        self.sense_distance_noise = new_sense_noise_range
        self.sense_noise_bearing = new_sense_noise_bearing

    def get_pose(self):
        """
        returning the pose vector
        :return: (x, y, theta) the pose vector
        """
        return self.x, self.y, self.theta

    def move(self, turn_movement_command, forward_movement_command, add_noise=True):
        """
        takes as input the two motor commands and outputs the new pose
        :param add_noise: boolean - to define if the true path is returned or the path defined by the model
        :param turn_movement_command: a turn movement command u1,(u1 ∈ [0, 2π))
        :param forward_movement_command: a forward movement command u2 (u2 > 0)
        :return: new pose- x, y, orientation
        """
        assert 0 <= turn_movement_command < 2 * np.pi, 'turn_movement_command should be in [0, 2π)'
        assert forward_movement_command > 0, 'forward_movement_command should be a positive number'

        if add_noise:
            # add noise to the controls if needed.
            turn_movement_command = turn_movement_command + np.random.normal(0, self.turn_noise)
            forward_movement_command = forward_movement_command + np.random.normal(0, self.forward_noise)

        # update the new pose based on the physical model.
        # keep x,y in range according to assumption that the world is cyclic.
        theta_prime = (self.theta + turn_movement_command) % (2 * np.pi)
        x_prime = (self.x + forward_movement_command * np.cos(theta_prime)) % self._world_size
        y_prime = (self.y + forward_movement_command * np.sin(theta_prime)) % self._world_size

        self.set(x_prime, y_prime, theta_prime)
        return self.get_pose()

    def sense(self, world):
        """
        sense the location of the landmarks
        :param world: the world
        :return: list of measurements
        """
        return [tuple(self._calc_distances_bearing(landmark)) for landmark in world.get_landmarks()]

    def _calc_distances_bearing(self, landmark, true_distance=False):
        """
        calculate the distances and bearing of the robot to a given landmark
        :param landmark: the landmark the is measured
        :param true_distance: boolean - calculate the real distance or the measured one (with noise)
        :return: distance, bearing
        """
        distance = np.linalg.norm(np.asarray(landmark) - np.array([self.x, self.y]))
        bearing = atan2(landmark[1] - self.y, landmark[0] - self.x) - self.theta
        if not true_distance:
            distance = distance + np.random.normal(0, self.sense_distance_noise)
            bearing = bearing + np.random.normal(0, self.sense_noise_bearing)
        return distance % self._world_size, bearing % (2 * np.pi)

    def measurement_probability(self, measurements, world):
        """
        calculate the measurement probability
        :param measurements: list of measurements
        :param world: the world
        :return: measurement probability
        """
        landmarks = world.get_landmarks()
        q = 1  # initialize
        for i in range(len(landmarks)):
            true_distance, true_bearing = self._calc_distances_bearing(landmark=landmarks[i], true_distance=True)
            q = q * norm.pdf(measurements[i][0], true_distance, self.sense_distance_noise)
            q = q * norm.pdf(measurements[i][1], true_bearing, self.sense_noise_bearing)
        return q


