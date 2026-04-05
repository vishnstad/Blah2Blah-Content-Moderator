# Client implementation
from openenv.core.client import EnvClient
from .models import ModerationAction, ModerationObservation, ModerationState

class ModerationEnv(EnvClient[ModerationAction, ModerationObservation, ModerationState]):
    pass
