from typing import Literal, List, Dict, Any, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State

class ModerationAction(Action):
    decision: Literal["ALLOW", "WARN", "SOFT_BLOCK", "HARD_BLOCK", "ESCALATE"]
    policy_tags: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None

class ModerationObservation(Observation):
    content: str
    metadata: Dict[str, Any]
    thread_context: List[str]
    task_id: Literal["easy", "medium", "hard"]
    step_index: int
    done: bool

class ModerationState(State):
    task_id: Literal["easy", "medium", "hard"] = "easy"
    total_reward: float = 0.0
    catastrophic_error: bool = False
