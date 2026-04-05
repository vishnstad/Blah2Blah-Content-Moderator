# inference.py
import os
import json
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN")

def build_prompt_from_observation(obs) -> str:
    prompt = f"""
You are a content moderation AI. Your task is to moderate the following content.
Task ID: {obs.task_id}
Step Index: {obs.step_index}

Content to moderate:
"{obs.content}"

Metadata:
{json.dumps(obs.metadata, indent=2)}

Thread Context (if any):
{json.dumps(obs.thread_context, indent=2)}

You must respond with a JSON object containing the following fields:
- decision: Must be one of "ALLOW", "WARN", "SOFT_BLOCK", "HARD_BLOCK", "ESCALATE".
- policy_tags: A list of tags applicable to this content (e.g. ["spam", "harassment"]).
- explanation: A short explanation of your decision.

Respond ONLY with valid JSON.
"""
    return prompt

def parse_action_from_completion(completion, action_class) -> dict:
    try:
        content = completion.choices[0].message.content
        # Basic JSON extraction in case there's markdown formatting
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        data = json.loads(content)
        
        return action_class(
            decision=data.get("decision", "ALLOW"),
            policy_tags=data.get("policy_tags", []),
            explanation=data.get("explanation", None)
        )
    except Exception as e:
        # Fallback action on error
        return action_class(decision="ALLOW", policy_tags=[], explanation=str(e))

def main():
    if not HF_TOKEN:
        print("Warning: HF_TOKEN environment variable not set. OpenAI validation may fail.")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy-token")
    
    # Import locally
    from content_moderation_env.server.moderation_environment import ModerationEnvironment
    from content_moderation_env.models import ModerationAction
    
    env = ModerationEnvironment()
    
    tasks = ["easy", "medium", "hard"]
    
    for task_name in tasks:
        # 1. Print START line
        print(f"[START] task={task_name} env=content-moderation-env model={MODEL_NAME}")
        
        # 2. Run the episode
        obs = env.reset(task=task_name)
        rewards = []
        step = 1
        
        while True:
            # Check if environment is already done
            if getattr(obs, "done", False):
                break
                
            prompt = build_prompt_from_observation(obs)
            last_error = None
            action = None
            reward = 0.0
            done = False
            info = {}
            
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                action = parse_action_from_completion(completion, ModerationAction)
                
                obs = env.step(action)
                # In standard environment, obs might contain these, but our step returns ModerationObservation
                reward = obs.reward
                done = obs.done
                
            except Exception as e:
                reward = 0.0
                done = True
                last_error = str(e)
                action = str(action) if action else "error"
                
            rewards.append(reward)
            
            action_str = f"decision={getattr(action, 'decision', 'unknown')}" if not isinstance(action, str) else action
            
            # Print EXACT step log
            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={last_error or 'null'}"
            )
            
            step += 1
            if done:
                break
                
        # 3. Print END line
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        success = (len(rewards) > 0) and (sum(rewards) / len(rewards) >= 0.3)
        
        print(
            f"[END] success={str(success).lower()} "
            f"steps={len(rewards)} rewards={rewards_str}"
        )

if __name__ == "__main__":
    main()
