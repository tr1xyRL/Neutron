import math
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward, FaceBallReward
import numpy as np
from rlgym.utils import math as rl_math
from fridge_math import normalize, distance, distance2D, clamp

SIDE_WALL_X = 4096  # +/-
BACK_WALL_Y = 5120  # +/-
CEILING_Z = 2044
BACK_NET_Y = 6000  # +/-

GOAL_HEIGHT = 642.775

ORANGE_GOAL_CENTER = (0, BACK_WALL_Y, GOAL_HEIGHT / 2)
BLUE_GOAL_CENTER = (0, -BACK_WALL_Y, GOAL_HEIGHT / 2)

# Often more useful than center
ORANGE_GOAL_BACK = (0, BACK_NET_Y, GOAL_HEIGHT / 2)
BLUE_GOAL_BACK = (0, -BACK_NET_Y, GOAL_HEIGHT / 2)

BALL_RADIUS = 92.75

BALL_MAX_SPEED = 6000

BLUE_TEAM = 0
ORANGE_TEAM = 1

CAR_MAX_SPEED = 2300


class RetreatReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.defense_target = np.array(BLUE_GOAL_BACK)

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == BLUE_TEAM:
            ball = state.ball.position
            pos = player.car_data.position
            vel = player.car_data.linear_velocity
        else:
            ball = state.inverted_ball.position
            pos = player.inverted_car_data.position
            vel = player.inverted_car_data.linear_velocity

        reward = 0.0
        if ball[1]+200 < pos[1]:
            pos_diff = self.defense_target - pos
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / CAR_MAX_SPEED
            reward = np.dot(norm_pos_diff, norm_vel)
        return reward


class AerialNavigation(RewardFunction):
    def __init__(
        self, ball_height_min=400, player_height_min=200, beginner=True
    ) -> None:
        super().__init__()
        self.ball_height_min = ball_height_min
        self.player_height_min = player_height_min
        self.face_reward = FaceBallReward()
        self.beginner = beginner

    def reset(self, initial_state: GameState) -> None:
        self.face_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if (
            not player.on_ground
            and state.ball.position[2]
            > self.ball_height_min
            > player.car_data.position[2]
            and player.car_data.linear_velocity[2] > 0
            and distance2D(player.car_data.position, state.ball.position)
            < state.ball.position[2] * 3
        ):
            # vel check
            ball_direction = normalize(state.ball.position - player.car_data.position)
            alignment = ball_direction.dot(normalize(player.car_data.linear_velocity))
            if self.beginner:
                reward += max(
                    0, alignment * 0.5
                )  # * (np.linalg.norm(player.car_data.linear_velocity)/2300)
                # old
                # #face check
                # face_reward = self.face_reward.get_reward(player, state, previous_action)
                # if face_reward > 0:
                #     reward += face_reward * 0.25
                # #boost check
                #     if previous_action[6] == 1 and player.boost_amount > 0:
                #         reward += face_reward

            reward += alignment * (
                np.linalg.norm(player.car_data.linear_velocity) / 2300.0
            )

        return max(reward, 0)


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


class WallTouchReward(RewardFunction):
    def __init__(self, min_height=92, exp=0.2):
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

        
class TeamSpacingReward(RewardFunction):
    def __init__(self, min_spacing: float = 1000) -> None:
        super().__init__()
        self.min_spacing = clamp(math.inf, 0.0000001, min_spacing)

    def reset(self, initial_state: GameState):
        pass

    def spacing_reward(self, player: PlayerData, state: GameState) -> float:
        reward = 0
        for p in state.players:
            if p.team_num == player.team_num and p.car_id != player.car_id and not player.is_demoed and not p.is_demoed:
                separation = distance(player.car_data.position, p.car_data.position)
                if separation < self.min_spacing:
                    reward -= 1-(separation / self.min_spacing)

        return reward

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return self.spacing_reward(player, state)


class VelocityBallDefense(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
            player.team_num == BLUE_TEAM
            and state.ball.position[1] < 0
            or player.team_num == ORANGE_TEAM
            and state.ball.position[1] > 0
        ):
            if player.team_num == BLUE_TEAM:
                defense_objective = np.array(BLUE_GOAL_BACK)
            else:
                defense_objective = np.array(ORANGE_GOAL_BACK)

            vel = state.ball.linear_velocity
            pos_diff = state.ball.position - defense_objective
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            vel /= BALL_MAX_SPEED
            return float(np.dot(norm_pos_diff, vel))
        return 0


class CradleReward(RewardFunction):
    def __init__(self, minimum_barrier: float = 200):
        super().__init__()
        self.min_distance = minimum_barrier

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
            player.car_data.position[2] < state.ball.position[2]
            and (BALL_RADIUS + 20 < state.ball.position[2] < BALL_RADIUS + 200)
            and distance2D(player.car_data.position, state.ball.position) <= 170
        ):
            if (
                abs(state.ball.position[0]) < 3946
                and abs(state.ball.position[1]) < 4970
            ):  # side and back wall values - 150
                if self.min_distance > 0:
                    for _player in state.players:
                        if (
                            _player.team_num != player.team_num
                            and distance(_player.car_data.position, state.ball.position)
                            < self.min_distance
                        ):
                            return 0

                return 1

        return 0


class CradleFlickReward(RewardFunction):
    def __init__(
        self,
        minimum_barrier: float = 400,
        max_vel_diff: float = 400,
        training: bool = True,
    ):
        super().__init__()
        self.min_distance = minimum_barrier
        self.max_vel_diff = max_vel_diff
        self.training = training
        self.cradle_reward = CradleReward(minimum_barrier=0)

    def reset(self, initial_state: GameState):
        self.cradle_reward.reset(initial_state)

    def stable_carry(self, player: PlayerData, state: GameState) -> bool:
        if BALL_RADIUS + 20 < state.ball.position[2] < BALL_RADIUS + 80:
            if (
                abs(
                    np.linalg.norm(
                        player.car_data.linear_velocity - state.ball.linear_velocity
                    )
                )
                <= self.max_vel_diff
            ):
                return True
        return False

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = self.cradle_reward.get_reward(player, state, previous_action) * 0.5
        if reward > 0:
            if not self.training:
                reward = 0
            stable = self.stable_carry(player, state)
            challenged = False
            for _player in state.players:
                if (
                    _player.team_num != player.team_num
                    and distance(_player.car_data.position, state.ball.position)
                    < self.min_distance
                ):
                    challenged = True
                    break
            if challenged:
                if stable:
                    if player.on_ground:
                        return reward - 0.5
                    else:
                        if player.has_flip:
                            # small reward for jumping
                            return reward + 2
                        else:
                            # print("PLAYER FLICKED!!!")
                            # big reward for flicking
                            return reward + 5
            else:
                if stable:
                    return reward + 1

        return reward


class JumpTouchReward(RewardFunction):
    def __init__(self, min_height=92.75):
        self.min_height = min_height
        self.max_height = 2044-92.75
        self.range = self.max_height - self.min_height

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            return (state.ball.position[2] - self.min_height) / self.range

        return 0


class ForwardBiasReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return player.car_data.forward().dot(normalize(player.car_data.linear_velocity))
        

class NaiveSpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        linear_velocity = player.car_data.linear_velocity
        return rl_math.vecmag(linear_velocity) / 2300


class PositioningReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == BLUE_TEAM:
            ball = state.ball.position
            pos = player.car_data.position
        else:
            ball = state.inverted_ball.position
            pos = player.inverted_car_data.position

        reward = 0.0
        if ball[1] < pos[1]:
            diff = abs(ball[1] - pos[1])
            reward = -(diff / 12000)
        return reward


class GroundedReward(RewardFunction):
    def __init__(
        self,
    ):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return player.on_ground is True


class KickoffReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.vel_dir_reward = VelocityPlayerToBallReward()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if state.ball.position[0] == 0 and state.ball.position[1] == 0:
            return self.vel_dir_reward.get_reward(player, state, previous_action)
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


class PowerShotReward(RewardFunction):
    def __init__(self, min_change: float = 300):
        super().__init__()
        self.min_change = min_change
        self.last_velocity = np.array([0, 0])

    def reset(self, initial_state: GameState):
        self.last_velocity = np.array([0, 0])

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        cur_vel = np.array(
            [state.ball.linear_velocity[0], state.ball.linear_velocity[1]]
        )
        if player.ball_touched:
            vel_change = rl_math.vecmag(self.last_velocity - cur_vel)
            if vel_change > self.min_change:
                reward = vel_change / (2300*2)

        self.last_velocity = cur_vel
        return reward
