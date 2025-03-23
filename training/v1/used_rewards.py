from rlgym_sim.utils.reward_functions import RewardFunction, CombinedReward
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward, TouchBallReward, FaceBallReward
from rlgym_sim.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym_sim.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym_sim.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rewards2 import PowerShotReward, CradleReward, CradleFlickReward, KickoffReward, JumpTouchReward, TouchVelChange, WallTouchReward, VelocityBallDefense, AerialNavigation, AerialTraining
from rlgym_sim.utils.reward_functions.common_rewards.misc_rewards import SaveBoostReward
from rewards3 import SpeedTowardBallReward, InAirReward

class NeutronRewards(RewardFunction):
    def __init__(self):
        super().__init__()
        self.reward = CombinedReward(
            (
            TouchBallReward(1),
            SpeedTowardBallReward(),
            InAirReward(),
            FaceBallReward()
            
            ),
            (50, 5, 0.15, 1)
        )

    def reset(self, initial_state: GameState):
        self.reward.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        return self.reward.get_reward(player, state, previous_action)
