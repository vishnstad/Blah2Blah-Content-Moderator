from typing import Any, Optional
import uuid

from openenv.core.env_server.environment import Environment
from ..models import ModerationAction, ModerationObservation, ModerationState
from ..graders import grade_easy, grade_medium, grade_hard
from data.datasets import get_dataset

class ModerationEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._state = ModerationState(episode_id=str(uuid.uuid4()))
        self.dataset = []
        self._current_task = "easy"
        
    def reset(self, seed: Optional[int] = None, task: Optional[str] = None, episode_id: Optional[str] = None, **kwargs: Any) -> ModerationObservation:
        self._current_task = task or "easy"
        self.dataset = get_dataset(self._current_task)
        
        self._state = ModerationState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=self._current_task,
            total_reward=0.0,
            catastrophic_error=False
        )
        
        return self._get_observation(done=False, reward=0.0)
        
    def step(self, action: ModerationAction, timeout_s: Optional[float] = None, **kwargs: Any) -> ModerationObservation:
        if len(self.dataset) == 0 or self._state.step_count >= len(self.dataset):
            return self._get_observation(done=True, reward=0.0)
            
        current_data = self.dataset[self._state.step_count]
        
        reward = 0.0
        if self._current_task == "easy":
            reward = grade_easy(action.decision, current_data["true_decision"])
        elif self._current_task == "medium":
            reward = grade_medium(action.decision, action.policy_tags, current_data["true_decision"], current_data["true_tags"])
        elif self._current_task == "hard":
            reward = grade_hard(action.decision, action.policy_tags, current_data["true_decision"], current_data["true_tags"], current_data["severity"])

        self._state.total_reward += reward
        self._state.step_count += 1
        
        done = self._state.step_count >= len(self.dataset)
        
        return self._get_observation(done=done, reward=reward)

    def _get_observation(self, done: bool, reward: float) -> ModerationObservation:
        step_idx = min(self._state.step_count, len(self.dataset) - 1)
        if step_idx < 0:
            return ModerationObservation(
                content="NO CONTENT",
                metadata={},
                thread_context=[],
                task_id=self._current_task,
                step_index=0,
                done=True,
                reward=reward
            )
            
        data = self.dataset[step_idx]
        return ModerationObservation(
            content=data["content"],
            metadata=data["metadata"],
            thread_context=data.get("thread_context", []),
            task_id=self._current_task,
            step_index=self._state.step_count,
            done=done,
            reward=reward
        )
        
    @property
    def state(self) -> ModerationState:
        return self._state
