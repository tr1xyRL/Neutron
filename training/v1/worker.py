import numpy as np
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger



avg_goals_last = 0  # Initialize it outside the class scope (global)

class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [
                game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score,  # Orange team's score
                game_state.players[0].boost_amount,  # Player's boost amount (fixed reference)
                game_state.players[0].on_ground,  # Whether the player is on the ground
                game_state.players[0].match_goals,  # Goals scored by the player
                game_state.players[0].match_demolishes,  # Demolitions caused by the player
                game_state.players[0].ball_touched, # Touched the ball
                game_state.ball.linear_velocity,  # Ball's linear velocity
                game_state.players[0].match_saves # Saves
               ]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        global avg_goals_last  # Referencing the global variable

        avg_car_vel = 0
        avg_ball_vel = 0
        avg_goals = 0
        avg_goals_increase = 0
        avg_boost = 0
        avg_airtime = 0
        avg_demos = 0
        avg_touches = 0
        avg_saves = 0

        for metric_array in collected_metrics:
            car_velocity = metric_array[0]
            car_vel_magnitude = np.linalg.norm(car_velocity)
            avg_car_vel += car_vel_magnitude

            avg_boost += metric_array[3]
            avg_airtime += 1 - metric_array[4]
            avg_goals += metric_array[5]
            avg_demos += metric_array[6]
            avg_touches += metric_array[7]

            ball_velocity = metric_array[8]
            ball_vel_magnitude = np.linalg.norm(ball_velocity)
            avg_ball_vel += ball_vel_magnitude

            avg_saves += metric_array[9]

        avg_car_vel /= len(collected_metrics)
        avg_ball_vel /= len(collected_metrics)
        avg_boost /= len(collected_metrics)
        avg_airtime /= len(collected_metrics)
        avg_goals /= len(collected_metrics)
        avg_demos /= len(collected_metrics)
        avg_touches /= len(collected_metrics)
        avg_saves /= len(collected_metrics)

        # Ensure avg_goals_last is initialized before using it
        avg_goals_increase = avg_goals - avg_goals_last
        avg_goals_last = avg_goals  # Update avg_goals_last for future use

        avg_boost_big = avg_boost * 1000

        report = {
            "average player speed": avg_car_vel,
            "average ball speed": avg_ball_vel,
            "average boost": avg_boost,
            "average boost 0-1000": avg_boost_big,
            "average airtime": avg_airtime,
            "average goals": avg_goals,
            "average demos": avg_demos,
            "ball touch ratio": avg_touches,
            "Cumulative Timesteps": cumulative_timesteps,
            "average goals increase": avg_goals_increase,
            "average saves": avg_saves
        }

        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, \
        EventReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.action_parsers import ContinuousAction
    from rlgym_sim.utils.obs_builders import AdvancedObs
    from extra_state_setters.goalie_state import GoaliePracticeState
    from extra_state_setters.wall_state import WallPracticeState
    from extra_state_setters.replay_setter import ReplaySetter
    #from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward
    from extra_state_setters.wall_state import WallPracticeState
    from extra_state_setters.goalie_state import GoaliePracticeState
    from kbb_states import GroundAirDribble, WallDribble
    from obsaction.lookup_act import LookupAction
    from used_rewards import NeutronRewards
    from setters import WeightedSampleSetter
    from rlgym_sim.utils.state_setters import DefaultState, RandomState

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = LookupAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    #rewards_to_combine = (VelocityPlayerToBallReward(),
                          #VelocityBallToGoalReward(),
                          #EventReward(team_goal=1, concede=-1, demo=0.1))
    #reward_weights = (0.01, 0.1, 10.0)

    reward_fn = NeutronRewards()

    obs_builder = AdvancedObs()

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         state_setter=WeightedSampleSetter([RandomState(ball_rand_speed=True,cars_rand_speed=True,cars_on_ground=False),RandomState(ball_rand_speed=True,cars_rand_speed=True,cars_on_ground=True),ReplaySetter("ssl_1v1.npy"),], [5,5,2]),
                         action_parser=action_parser)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()

    # 32 processes
    n_proc = 12

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      ppo_batch_size=50000,
                      ts_per_iteration=50000,
                      exp_buffer_size=150000,
                      ppo_minibatch_size=50000,
                      policy_lr=5e-5,
                      critic_lr=5e-5,
                      policy_layer_sizes=[2048, 2048, 1024, 1024],
                      critic_layer_sizes=[2048, 2048, 1024, 1024],
                      ppo_ent_coef=0.001,
                      ppo_epochs=1,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=1_000_000_000_000_000,
                      log_to_wandb=True)
    learner.learn()
