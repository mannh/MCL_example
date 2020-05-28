
from World import World
from Robot import Robot
import numpy as np
import random
import matplotlib.pyplot as plt

params = {
    'M': 1000,
    'noise_params': {
        'new_forward_noise': 6,
        'new_turn_noise': 0.1,
        'new_sense_noise_range': 5,
        'new_sense_noise_bearing': 0.3
    },
    'init_location': (10, 15, 0),
    'seed': 255,
    'commands': [(0, 60), (np.pi/3, 30), (np.pi/4, 30), (np.pi/4, 20), (np.pi/4, 40)]
}


# section a
fig = plt.figure()
my_world = World()
my_world.plot(show=False)
my_robot = Robot()
for pose in [(45, 45, 0),  (50, 60, 0.5 * np.pi), (70, 30, 0.75 * np.pi)]:
    my_robot.set(*pose)
    my_robot.print()
    my_robot.plot(show=False)
plt.savefig('robot_world_a.png')
plt.show()


def execute_commands(robot, commands_list, add_noise, world, is_sense=False):
    """
    make the robot execute commands in order
    :param robot: the robot
    :param commands_list: list of commands to execute
    :param add_noise: boolean - used to get the true and desired path
    :param world: World
    :param is_sense: boolean - if sensory data is needed
    :return: list - robot locations after commands ; list (if is_sense = True) of sensory data
    """
    robot_commands_locations = [robot.get_pose()]
    sense_list = [None]
    for command in commands_list:
        robot_commands_locations.append(robot.move(*command, add_noise=add_noise))
        if is_sense:
            sense_list.append(robot.sense(world))
        robot.plot(show=False)
    if is_sense:
        return robot_commands_locations, sense_list
    else:
        return robot_commands_locations


def resampling(X, W):
    """
    Resample particles as part of MCL using numpy random library
    :param X: list of particles
    :param W: list of weights
    :return: list of resampled particles
    """
    # convert weights to probabilities
    probas = np.array(W)
    probas /= sum(probas)
    return list([X[i] for i in np.random.choice(a=np.arange(0, len(X)), p=probas, size=params['M'])])


def mcl(x_pre, u, z, m, sampled_particles = None):
    """
    MCL algorithm
    :param x_pre: tuple, current location of the robot
    :param u: tuple, (turn_movement_command forward_movement_command)
    :param z: tuple, current robot measurement
    :param m: World
    :return:
    """
    def get_xt_wt(_particle, _u):
        location = _particle.move(*_u)
        _particle.plot(mycolor="black", style="particle", show=False, markersize=1)
        proba = _particle.measurement_probability(measurements=z, world=m)
        return location, proba

    # Initialize particles
    particles = [Robot() for _ in range(params['M'])]
    if sampled_particles is None:
        for particle in particles:
            particle.set(*x_pre)
            particle.set_noise(**params['noise_params'])
    else:
        for index, particle in enumerate(particles):
            particle.set(*sampled_particles[index])
            particle.set_noise(**params['noise_params'])

    x = []
    w = []

    for particle in particles:
        _x, _w = get_xt_wt(particle, u)
        x.append(_x)
        w.append(_w)

    resampling_list = resampling(x, w)
    for sample in resampling_list:
        robot = Robot()
        robot.set_noise(**params['noise_params'])
        robot.set(*sample)
        robot.plot(mycolor="gray", style="particle", show=False, markersize=1)

    return resampling_list

# sections e,f,g
random.seed(params['seed'])
np.random.seed(params['seed'])
fig = plt.figure()
my_world = World()
my_world.plot(show=False)
my_robot = Robot()
my_robot.set_noise(**params['noise_params'])

my_robot.set(*params['init_location'])
my_robot.plot(show=False)
locations = execute_commands(robot=my_robot, commands_list=params['commands'], add_noise=False, world=my_world)
plt.plot([loc[0] for loc in locations], [loc[1] for loc in locations], 'r--')
my_robot.set(*params['init_location'])
locations, measurements = execute_commands(robot=my_robot, commands_list=params['commands'], add_noise=True,
                                           world=my_world, is_sense=True)
plt.plot([loc[0] for loc in locations], [loc[1] for loc in locations], 'g-')
plt.savefig('pre_particles.png')

x_mean_list = [params['init_location'][0]]
y_mean_list = [params['init_location'][1]]
mean_robot = Robot()
particles_locations = None
for index, measurement in enumerate(measurements):
    if measurement is not None:
        particles_locations = mcl(locations[index-1], params['commands'][index-1], measurement, my_world,
                                  particles_locations)
        x_mean = np.mean([loc[0] for loc in particles_locations])
        y_mean = np.mean([loc[1] for loc in particles_locations])
        thetha_mean = np.mean([loc[2] for loc in particles_locations])
        x_mean_list.append(x_mean)
        y_mean_list.append(y_mean)
        mean_robot.set(x_mean, y_mean, thetha_mean)
        mean_robot.plot(show=False, mycolor='m', markersize=2)

plt.plot(x_mean_list, y_mean_list, 'm-.')
plt.savefig('robot_world.png')
plt.show()




