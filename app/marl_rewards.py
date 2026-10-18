"""
Perspecta - MARL Reward Shaping Engine (Objective 6)
Synthetic 5x5 Gridworld with 2 Coordinated Agents.
Compares Baseline Tabular Q-Learning (Sparse Goal Reward) vs Dense LLM-Heuristic Reward Shaping.
Measures real episodes-to-convergence and calculates exact percentage sample efficiency improvement.
"""

import asyncio
import random
import time
from typing import Dict, Any, List, Tuple
import numpy as np

from app.providers import MockProvider


class GridWorldEnv:
    """
    Synthetic 5x5 Gridworld Environment with 2 Agents and 1 Shared Goal at (4,4).
    Actions: 0=Up, 1=Down, 2=Left, 3=Right.
    """
    def __init__(self, grid_size: int = 5):
        self.grid_size = grid_size
        self.goal_pos = (4, 4)
        self.reset()

    def reset(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        self.agent1_pos = (0, 0)
        self.agent2_pos = (0, 1)
        self.steps = 0
        return self.agent1_pos, self.agent2_pos

    def step(self, a1_action: int, a2_action: int) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], float, bool]:
        self.steps += 1
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Move Agent 1
        d1 = moves[a1_action]
        new_a1 = (
            max(0, min(self.grid_size - 1, self.agent1_pos[0] + d1[0])),
            max(0, min(self.grid_size - 1, self.agent1_pos[1] + d1[1]))
        )
        self.agent1_pos = new_a1

        # Move Agent 2
        d2 = moves[a2_action]
        new_a2 = (
            max(0, min(self.grid_size - 1, self.agent2_pos[0] + d2[0])),
            max(0, min(self.grid_size - 1, self.agent2_pos[1] + d2[1]))
        )
        self.agent2_pos = new_a2

        # Check goal condition (both agents reach or close to goal)
        at_goal = (self.agent1_pos == self.goal_pos or self.agent2_pos == self.goal_pos)
        done = at_goal or (self.steps >= 50)
        
        # Sparse reward
        reward = 10.0 if at_goal else -0.1
        return (self.agent1_pos, self.agent2_pos), reward, done


def run_tabular_q_learning(use_shaping: bool, max_episodes: int = 200) -> Tuple[int, List[float]]:
    """
    Runs multi-agent tabular Q-learning on 5x5 gridworld.
    Returns (episodes_to_convergence, reward_history).
    """
    env = GridWorldEnv(grid_size=5)
    # Q-table dimensions: (grid_x, grid_y, actions=4)
    q_table_1 = np.zeros((5, 5, 4))
    q_table_2 = np.zeros((5, 5, 4))
    
    alpha = 0.2
    gamma = 0.9
    epsilon = 0.3
    
    episode_lengths: List[int] = []
    rewards_history: List[float] = []
    
    episodes_to_converge = max_episodes

    for ep in range(1, max_episodes + 1):
        s1, s2 = env.reset()
        done = False
        total_ep_reward = 0.0

        while not done:
            # Epsilon-greedy action selection
            a1 = random.randint(0, 3) if random.random() < epsilon else int(np.argmax(q_table_1[s1[0], s1[1]]))
            a2 = random.randint(0, 3) if random.random() < epsilon else int(np.argmax(q_table_2[s2[0], s2[1]]))

            (next_s1, next_s2), reward, done = env.step(a1, a2)

            # Base sparse reward
            r1 = reward
            r2 = reward

            # LLM-Generated Dense Reward Shaping Signal
            if use_shaping:
                # Potential-based shaping heuristic F(s, s') = gamma * phi(s') - phi(s)
                phi_s1 = MockProvider.compute_dense_reward_heuristic(s1, env.goal_pos)
                phi_next1 = MockProvider.compute_dense_reward_heuristic(next_s1, env.goal_pos)
                shaping_signal_1 = gamma * phi_next1 - phi_s1

                phi_s2 = MockProvider.compute_dense_reward_heuristic(s2, env.goal_pos)
                phi_next2 = MockProvider.compute_dense_reward_heuristic(next_s2, env.goal_pos)
                shaping_signal_2 = gamma * phi_next2 - phi_s2

                r1 += shaping_signal_1
                r2 += shaping_signal_2

            # Q-learning update
            best_next_q1 = np.max(q_table_1[next_s1[0], next_s1[1]])
            q_table_1[s1[0], s1[1], a1] += alpha * (r1 + gamma * best_next_q1 - q_table_1[s1[0], s1[1], a1])

            best_next_q2 = np.max(q_table_2[next_s2[0], next_s2[1]])
            q_table_2[s2[0], s2[1], a2] += alpha * (r2 + gamma * best_next_q2 - q_table_2[s2[0], s2[1], a2])

            s1, s2 = next_s1, next_s2
            total_ep_reward += (r1 + r2) / 2.0

        episode_lengths.append(env.steps)
        rewards_history.append(round(total_ep_reward, 2))

        # Check convergence: average episode length over last 10 episodes <= 10 steps
        if len(episode_lengths) >= 10:
            avg_steps = sum(episode_lengths[-10:]) / 10.0
            if avg_steps <= 9.0 and episodes_to_converge == max_episodes:
                episodes_to_converge = ep
                break

        # Decay epsilon slightly
        epsilon = max(0.05, epsilon * 0.98)

    return episodes_to_converge, rewards_history


async def run_marl_scenario(run_id: str, event_queue: Any) -> Dict[str, Any]:
    """
    Executes Objective 6: MARL Reward Shaping evaluation.
    Runs sparse vs shaped Q-learning, measures actual episodes-to-convergence,
    computes real % improvement, and streams live telemetry events.
    """
    await event_queue.emit(
        "agent_message",
        {"agent": "MARLSupervisor", "content": "Starting MARL evaluation: 5x5 Gridworld with 2 agents."},
        simulated=False
    )

    # Run 1: Baseline Sparse Rewards
    await event_queue.emit(
        "agent_message",
        {"agent": "MARLSupervisor", "content": "Run 1: Training baseline tabular Q-learning with sparse goal reward (+10)..."},
        simulated=False
    )
    sparse_episodes, sparse_rewards = run_tabular_q_learning(use_shaping=False, max_episodes=150)

    # Emit telemetry step event for sparse run
    await event_queue.emit(
        "reward_step",
        {
            "mode": "sparse",
            "episodes_to_convergence": sparse_episodes,
            "sample_reward_curve": sparse_rewards[:15],
            "simulated": False
        },
        simulated=False
    )

    await asyncio.sleep(0.3)

    # Run 2: Dense LLM Reward Shaping
    await event_queue.emit(
        "agent_message",
        {
            "agent": "MARLSupervisor",
            "content": "Run 2: Training tabular Q-learning with LLM dense potential-based reward shaping...",
            "shaping_heuristic": "phi(s) = -manhattan_distance(agent, goal)",
            "disclaimer": "Simulated dense reward heuristic generated by MockProvider"
        },
        simulated=True
    )
    shaped_episodes, shaped_rewards = run_tabular_q_learning(use_shaping=True, max_episodes=150)

    # Emit telemetry step event for shaped run
    await event_queue.emit(
        "reward_step",
        {
            "mode": "shaped",
            "episodes_to_convergence": shaped_episodes,
            "sample_reward_curve": shaped_rewards[:15],
            "simulated": True
        },
        simulated=True
    )

    # Compute actual improvement percentage from measured numbers
    if sparse_episodes > 0:
        improvement_pct = round(((sparse_episodes - shaped_episodes) / float(sparse_episodes)) * 100.0, 2)
    else:
        improvement_pct = 0.0

    disclaimer_msg = (
        "Illustrative toy result on a synthetic gridworld — not the benchmark referenced in the literature review."
    )

    await event_queue.emit(
        "agent_message",
        {
            "agent": "MARLSupervisor",
            "content": f"MARL Evaluation Complete! Sparse: {sparse_episodes} eps | Dense Shaped: {shaped_episodes} eps | Measured Improvement: {improvement_pct}%.",
            "sparse_episodes": sparse_episodes,
            "shaped_episodes": shaped_episodes,
            "improvement_pct": improvement_pct,
            "disclaimer": disclaimer_msg
        },
        simulated=False
    )

    return {
        "run_id": run_id,
        "sparse_episodes": sparse_episodes,
        "shaped_episodes": shaped_episodes,
        "improvement_pct": improvement_pct,
        "disclaimer": disclaimer_msg,
        "success": True
    }
