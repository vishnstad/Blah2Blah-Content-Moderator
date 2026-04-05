"""
Pydantic models for the Content Moderation Environment.
These are standalone and do not depend on openenv imports.
"""
from typing import Literal, List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ModerationAction(BaseModel):
    """Action the agent takes to moderate content."""
    decision: Literal["ALLOW", "WARN", "SOFT_BLOCK", "HARD_BLOCK", "ESCALATE"]
    policy_tags: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None

class ModerationObservation(BaseModel):
    """Observation returned by the environment at each step."""
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    thread_context: List[str] = Field(default_factory=list)
    task_id: Literal["easy", "medium", "hard"] = "easy"
    step_index: int = 0
    done: bool = False
    reward: float = 0.0

class ModerationState(BaseModel):
    """Internal state of the environment."""
    episode_id: str = ""
    step_count: int = 0
    task_id: Literal["easy", "medium", "hard"] = "easy"
    total_reward: float = 0.0
    catastrophic_error: bool = False
