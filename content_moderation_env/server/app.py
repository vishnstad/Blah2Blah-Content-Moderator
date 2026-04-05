try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    pass # Provide logic later if needed
from .moderation_environment import ModerationEnvironment
from ..models import ModerationAction, ModerationObservation

app = create_app(
    ModerationEnvironment, 
    ModerationAction, 
    ModerationObservation, 
    env_name="content_moderation_env"
)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
