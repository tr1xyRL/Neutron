from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.common_values import BLUE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM, BALL_MAX_SPEED, \
    CAR_MAX_SPEED
import numpy as np
import math

def distance(x: np.array, y: np.array) -> float:
    return np.linalg.norm(x - y)

def clamp(max_range: float, min_range: float, number: float) -> float:
    return max((min_range, min((number, max_range))))

class RetreatSpeedReward(RewardFunction):
    """
    Reward that encourages going fast towards the ball when retreating
    """
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.car_data.position[2] > state.ball.position[2]:
            return np.linalg.norm(player.car_data.linear_velocity) / 500
        return 0

class AerialTraining(RewardFunction):
    def __init__(self, ball_height_min=400, player_min_height=300) -> None:
        super().__init__()
        self.vel_reward = VelocityPlayerToBallReward()
        self.ball_height_min = ball_height_min
        self.player_min_height = player_min_height

    def reset(self, initial_state: GameState) -> None:
        self.vel_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
                not player.on_ground
                and state.ball.position[2] > self.ball_height_min
                and self.player_min_height < player.car_data.position[2] < state.ball.position[2]
        ):
            divisor = max(1, distance(player.car_data.position, state.ball.position)/1000)
            return self.vel_reward.get_reward(player, state, previous_action)/divisor

        return 0

class PositiveWrapperReward(RewardFunction):
    """A simple wrapper to ensure a reward only returns positive values"""
    def __init__(self, base_reward):
        super().__init__()
        #pass in instantiated reward object
        self.rew = base_reward

    def reset(self, initial_state: GameState):
        self.rew.reset(initial_state)

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        rew = self.rew.get_reward(player, state, previous_action)
        return 0 if rew < 0 else rew

class JumpTouchReward(RewardFunction):
    def __init__(self, min_height=132):
        self.min_height = min_height
        self.max_height = 2044-132
        self.range = self.max_height - self.min_height

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            return (state.ball.position[2] - self.min_height) / self.range

        return 0

class TouchVelChange(RewardFunction):
    def __init__(self):
        self.last_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.last_vel = np.zeros(3)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.ball_touched:
            vel_difference = abs(np.linalg.norm(self.last_vel - state.ball.linear_velocity))
            reward = vel_difference / 4600.0

        self.last_vel = state.ball.linear_velocity

        return reward

class WallTouchReward(RewardFunction):
    def __init__(self, min_height=350, exp=0.2):
        self.min_height = min_height
        self.exp = exp
        self.max = math.inf

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and player.on_ground and state.ball.position[2] >= self.min_height:
            return (clamp(self.max, 0.0001, state.ball.position[2] - 92) ** self.exp)-1

        return 0

class OmniBoostDiscipline(RewardFunction):
    def __init__(self, aerial_forgiveness=False):
        super().__init__()
        self.values = [0 for _ in range(64)]
        self.af = aerial_forgiveness

    def reset(self, initial_state: GameState):
        self.values = [0 for _ in range(64)]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        old, self.values[player.car_id] = self.values[player.car_id], player.boost_amount
        if player.on_ground or not self.af:
            return -int(self.values[player.car_id] < old)
        return 0

class InAirReward(RewardFunction): # We extend the class "RewardFunction"
    # Empty default constructor (required)
    def __init__(self):
        super().__init__()

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        
        # "player" is the current player we are getting the reward of
        # "state" is the current state of the game (ball, all players, etc.)
        # "previous_action" is the previous inputs of the player (throttle, steer, jump, boost, etc.) as an array
        
        if not player.on_ground:
            # We are in the air! Return full reward
            return 1
        else:
            # We are on ground, don't give any reward
            return 0