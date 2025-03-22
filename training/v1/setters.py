from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils import RewardFunction
from random import choices
from typing import Sequence, Union, Tuple
from rlgym_sim.utils.math import rand_vec3
from rlgym_sim.utils.common_values import CAR_MAX_SPEED, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z, BALL_RADIUS, CAR_MAX_ANG_VEL, \
    BALL_MAX_SPEED
import random
import numpy as np
from numpy import random as rand
SPAWN_BLUE_POS = [[-2048, -2560, 17], [2048, -2560, 17],
                      [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]]
SPAWN_BLUE_YAW = [0.25 * np.pi, 0.75 * np.pi,
                  0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
SPAWN_ORANGE_POS = [[2048, 2560, 17], [-2048, 2560, 17],
                    [256, 3840, 17], [-256, 3840, 17], [0, 4608, 17]]
SPAWN_ORANGE_YAW = [-0.75 * np.pi, -0.25 *
                    np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]

X_MAX = 3000
Y_MAX = 4500
Z_MAX_BALL = 1200
Z_MAX_CAR = 1900
PITCH_MAX = np.pi / 2
YAW_MAX = np.pi
ROLL_MAX = np.pi

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Z = CEILING_Z - BALL_RADIUS

PITCH_LIM = np.pi / 2
YAW_LIM = np.pi
ROLL_LIM = np.pi

GOAL_X_MAX = 800.0
GOAL_X_MIN = -800.0

PLACEMENT_BOX_X = 5000
PLACEMENT_BOX_Y = 2000
PLACEMENT_BOX_Y_OFFSET = 3000

GOAL_LINE = 5100


class AerialSetup(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies state_wrapper values to simulate an aerial setup opportunity
        """
        ball_x = random.randrange(-X_MAX, X_MAX)#rand.random() * self.X_MAX - self.X_MAX / 2
        ball_y = random.randrange(-Y_MAX, Y_MAX)#rand.random() * self.Y_MAX - self.Y_MAX / 2
        state_wrapper.ball.set_pos(ball_x, ball_y, 140)
        state_wrapper.ball.set_lin_vel(x=random.randrange(-75, 75), y=random.randrange(-75, 75), z=1400)

        for car in state_wrapper.cars:
            ground_spawn = random.randint(0, 6) > 4
            c_z = 17
            if ground_spawn:
                c_z = 35
            c_x = random.randrange(ball_x-500, ball_x+500)
            if car.team_num == 0:
                # select a unique spawn state from pre-determined values
                c_y = random.randrange(ball_y-500, ball_y)
                yaw = 0.5 * np.pi

            else:
                # select a unique spawn state from pre-determined values
                c_y = random.randrange(ball_y, ball_y + 500)
                yaw = -0.5 * np.pi

            pos = [c_x, c_y, c_z]

            car.set_pos(*pos)
            if not ground_spawn:
                car.set_lin_vel(*[0, 0, random.randint(100, 1000)])
            car.set_rot(yaw=yaw)
            car.boost = random.random()

class BetterRandom(StateSetter):  # Random state with some triangular distributions
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.ball.set_pos(
            x=np.random.uniform(-LIM_X, LIM_X),
            y=np.random.uniform(-LIM_Y, LIM_Y),
            z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z),
        )

        # 99.9% chance of below ball max speed
        ball_speed = np.random.exponential(-BALL_MAX_SPEED / np.log(1 - 0.999))
        vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED))
        state_wrapper.ball.set_lin_vel(*vel)

        ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
        state_wrapper.ball.set_ang_vel(*ang_vel)

        for car in state_wrapper.cars:
            # On average 1 second at max speed away from ball
            ball_dist = np.random.exponential(BALL_MAX_SPEED)
            ball_car = rand_vec3(ball_dist)
            car_pos = state_wrapper.ball.position + ball_car
            if abs(car_pos[0]) < LIM_X \
                    and abs(car_pos[1]) < LIM_Y \
                    and 0 < car_pos[2] < LIM_Z:
                car.set_pos(*car_pos)
            else:  # Fallback on fully random
                car.set_pos(
                    x=np.random.uniform(-LIM_X, LIM_X),
                    y=np.random.uniform(-LIM_Y, LIM_Y),
                    z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z),
                )

            vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
            car.set_lin_vel(*vel)

            car.set_rot(
                pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
            )

            ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
            car.set_ang_vel(*ang_vel)
            car.boost = np.random.uniform(0, 1)

from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper
from random import choices
from typing import Sequence, Union, Tuple


class WeightedSampleSetter(StateSetter):
    """
    Samples StateSetters randomly according to their weights.
    :param state_setters: 1-D array-like of state-setters to be sampled from
    :param weights: 1-D array-like of the weights associated with each entry in state_setters
    """

    def __init__(self, state_setters: Sequence[StateSetter], weights: Sequence[float]):
        super().__init__()
        self.state_setters = state_setters
        self.weights = weights
        assert len(state_setters) == len(weights), \
            f"Length of state_setters should match the length of weights, " \
            f"instead lengths {len(state_setters)} and {len(weights)} were given respectively."

    @classmethod
    def from_zipped(
            cls,
            *setters_and_weights: Union[StateSetter, Tuple[RewardFunction, float]]
    ) -> "WeightedSampleSetter":
        """
        Alternate constructor which takes any number of either rewards, or (reward, weight) tuples.
        :param setters_and_weights: a sequence of RewardFunction or (RewardFunction, weight) tuples
        """
        rewards = []
        weights = []
        for value in setters_and_weights:
            if isinstance(value, tuple):
                r, w = value
            else:
                r, w = value, 1.
            rewards.append(r)
            weights.append(w)
        return cls(tuple(rewards), tuple(weights))

    def reset(self, state_wrapper: StateWrapper):
        """
        Executes the reset of randomly sampled state-setter
        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        choices(self.state_setters, weights=self.weights)[0].reset(state_wrapper)