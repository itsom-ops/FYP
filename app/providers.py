"""
Perspecta - Mock Provider & LLM Abstractions
Simulates LLM responses, confidence scores, consensus positions, and dense reward heuristics.
Strictly adheres to Honesty Rule: all outputs carry simulated=True.
"""

import time
import random
from typing import Dict, Any, Tuple


class MockProvider:
    """
    Mock LLM provider generating deterministic simulated outputs for multi-agent workflows.
    Explicitly flags all responses with simulated=True.
    """
    
    CONFIDENCE_DISCLAIMER = (
        "Simulated confidence signal — hosted LLM APIs (Anthropic/OpenAI) do not expose logits; "
        "a logit-derived version would require a self-hosted open-weight backend (e.g. vLLM)."
    )
    
    @staticmethod
    def generate_code_draft(attempt: int) -> str:
        """
        Generates Python code for the collab-engineering scenario.
        Attempt 1 intentional syntax/logic bug to trigger sandbox failure & iteration loop.
        Attempt 2+ produces verified working Python code.
        """
        if attempt == 1:
            # Buggy python code: syntax error or failing assertion
            return (
                "def calculate_sum(a, b):\n"
                "    # Intentional bug in iteration 1 for verification loop\n"
                "    return a + b + 'invalid'\n"
            )
        else:
            # Fixed code
            return (
                "def calculate_sum(a, b):\n"
                "    # Fixed code passing unit tests\n"
                "    return a + b\n\n"
                "assert calculate_sum(10, 20) == 30\n"
                "assert calculate_sum(-5, 5) == 0\n"
                "print('All tests passed successfully!')\n"
            )

    @staticmethod
    def generate_reviewer_feedback(code: str, sandbox_success: bool, exit_code: int, error_log: str) -> Dict[str, Any]:
        """
        Simulates code reviewer critique based on sandbox execution result.
        """
        if not sandbox_success:
            return {
                "simulated": True,
                "persona": "Reviewer (Mock-Llama-3)",
                "verdict": "REJECT",
                "feedback": f"Sandbox execution failed with exit code {exit_code}. Error log: {error_log.strip()[:150]}. Please fix type error.",
                "confidence": 0.94
            }
        else:
            return {
                "simulated": True,
                "persona": "Reviewer (Mock-Llama-3)",
                "verdict": "APPROVE",
                "feedback": "Code executed cleanly in isolated sandbox and all assertions passed.",
                "confidence": 0.98
            }

    @staticmethod
    def generate_belief_state(agent_name: str, node_state: str, history_len: int, extra_context: str = "") -> str:
        """
        Theory of Mind / belief state lines built from real graph state filled into a template.
        Labeled as templated language built from real graph state, not learned inference.
        """
        template = (
            f"[{agent_name} Belief State | Graph Node: {node_state}]: "
            f"Observed {history_len} prior message(s). "
            f"Current belief: Task progress requires step '{node_state}'. "
            f"Confidence in peer output: High. {extra_context}"
        )
        return template

    @staticmethod
    def generate_debate_turn(persona: str, round_num: int, stance: str) -> Dict[str, Any]:
        """
        Simulates multi-round consensus debate turn between heterogeneous personas ("Llama-3-70B" vs "ERNIE-Bot-4").
        Emits stance, simulated confidence, and derived weight.
        """
        confidences = {
            "Llama-3-70B": [0.72, 0.85, 0.93],
            "ERNIE-Bot-4": [0.68, 0.79, 0.88]
        }
        conf = confidences.get(persona, [0.70, 0.80, 0.90])[min(round_num - 1, 2)]
        
        # Stance scoring scale: -1.0 (strongly oppose) to +1.0 (strongly support)
        if persona == "Llama-3-70B":
            stance_val = 0.5 + (round_num * 0.15)
            text = f"Round {round_num}: Proposing architectural partitioning based on benchmark data split."
        else:
            stance_val = 0.4 + (round_num * 0.18)
            text = f"Round {round_num}: Recommending unified memory pool with cross-validation guardrails."
            
        stance_val = min(1.0, stance_val)
        
        # Weighting formula: weight = stance * confidence
        derived_weight = round(stance_val * conf, 4)

        return {
            "simulated": True,
            "persona": persona,
            "round": round_num,
            "stance": stance,
            "stance_score": round(stance_val, 2),
            "confidence": conf,
            "derived_weight": derived_weight,
            "text": text,
            "disclaimer": MockProvider.CONFIDENCE_DISCLAIMER
        }

    @staticmethod
    def compute_dense_reward_heuristic(agent_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> float:
        """
        Simulates an LLM-generated dense reward shaping signal (potential-based Manhattan distance heuristic).
        Formula: phi(s) = -manhattan_distance(agent, goal)
        """
        dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        # Dense shaping potential phi(s)
        return float(-dist * 0.5)
