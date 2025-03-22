from rlgym_sim.utils.reward_functions import RewardFunction, CombinedReward
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward, TouchBallReward
from rlgym_sim.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym_sim.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym_sim.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rewards2 import PowerShotReward, CradleReward, CradleFlickReward, KickoffReward, JumpTouchReward, TouchVelChange, WallTouchReward, VelocityBallDefense, AerialNavigation, AerialTraining
from rlgym_sim.utils.reward_functions.common_rewards.misc_rewards import SaveBoostReward


class NeutronRewards(RewardFunction):
    def __init__(self):
        super().__init__()
        self.reward = CombinedReward(
            (
            CradleReward(), #1
            CradleFlickReward(),  #2
            SaveBoostReward(),  #3
            VelocityBallDefense(),    #4
            WallTouchReward(),   #5
            TouchBallReward(2.4),   #6
            KickoffReward(),    #7
            JumpTouchReward(93),   #8
            TouchVelChange(),   #9
            #PowerShotReward(550),    #10
            VelocityPlayerToBallReward(),   #11
            VelocityBallToGoalReward(),    #12
            AerialNavigation(), #13
            AerialTraining(), #14
            EventReward(        #15
                goal=100.0,
                team_goal=100.0,
                concede=-100.0,
                shot=15.0,
                save=30.5,
                demo=5.0,
                boost_pickup=1.75
            ),
            ),
            (0.95, 1.35, 0.50, 1.0, 1.0, 0.3, 1.0, 2.75, 1.6, 0.1, 1.15, 1.15, 1.0, 1.0)
        )

    def reset(self, initial_state: GameState):
        self.reward.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        return self.reward.get_reward(player, state, previous_action)