# Meta PyTorch OpenEnv Hackathon – Content Moderation Environment Plan

> Working title: **Content Moderation RL Environment (OpenEnv)**  
> Target: **Round 1 – Meta PyTorch OpenEnv Hackathon x Scaler School of Technology**

---

## 1. Hackathon overview and time reality

- **Hackathon**: Meta PyTorch OpenEnv Hackathon x Scaler School of Technology, in partnership with Meta, Hugging Face, and PyTorch.[web:2][web:42]  
- **Format**:
  - **Round 1 (Online)** – build and submit an OpenEnv environment from home.  
  - **Finale (On‑site, Bangalore)** – 48‑hour hackathon for top teams with Meta/HF engineers.[web:2][web:5]
- **Round‑1 submission**: a **Hugging Face Space URL** that exposes your OpenEnv environment and runs your `inference.py` end‑to‑end.[web:8]
- **Evaluation pipeline (high‑level)**:[web:8]
  1. Ping HF Space → must be live and respond.  
  2. Call `reset()` and `step()` via OpenEnv to validate the interface.  
  3. Run `inference.py` inside a Docker container with constrained resources (2 vCPU, 8 GB RAM).  
  4. Parse your `[START] / [STEP] / [END]` logs and compute scores per task.  
  5. Check that all tasks have graders and scores in `[0.0, 1.0]`.
- **Your time window** (from 5 April ~1 PM IST to 8 April 11:59 PM IST): roughly **3 days** of work time. Use it as a focused, structured sprint.

This means the priority is **correctness + compliance + clear design**, not exotic research; the judges and infrastructure will reward environments that “just work” and are easy to understand.[web:2][web:8]

---

## 2. Evaluation criteria and what they imply

From the Scaler dashboard and official hackathon pages, the Round‑1 evaluation focuses on:[web:2][web:8]

1. **Runtime correctness**  
   - Environment and `inference.py` run without errors.  
   - Handles edge cases and invalid actions gracefully (no crashes, no `None` rewards).

2. **Interface compliance (OpenEnv spec)**  
   - Implements `step()`, `reset()`, `state()` exactly as OpenEnv expects.[web:10][web:41]  
   - Uses **typed Pydantic models** for Observation, Action, and State / Reward schemas.  
   - Has a valid `openenv.yaml` that passes `openenv validate`.

3. **Task design**  
   - At least **three tasks** (`easy`, `medium`, `hard`).  
   - Realistic, testable objectives, not toy games.  
   - Clear difficulty progression and meaningful differences between tasks.

4. **Grading logic**  
   - Each task has a **programmatic grader** that returns scores in `[0.0, 1.0]`.[web:8]  
   - Grading must be deterministic and reproducible; given the same actions, you always get the same scores.  
   - Reward system should “make sense” from a human perspective (partial credit, penalties, etc.).

5. **LLM and logging compliance**  
   - All LLM calls must use the **OpenAI Client**, reading `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from environment variables with defaults where required.[web:8]  
   - `inference.py` must be in the **project root** and emit logs in exactly this format:  
     - `[START] task=<task_name> env=<benchmark> model=<model_name>`  
     - `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`  
     - `[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>`  
   - Any deviation in fields, order, or formatting can break automated evaluation.[web:8]

6. **Deployment & resource fit**  
   - A live **Hugging Face Space** with `openenv` tag, fully built and in `Running` state at submission time.[web:5][web:8]  
   - A working `Dockerfile` that builds and runs within **2 vCPU / 8 GB RAM** (keep dependencies lean, no heavy models inside the container).

Interpretation: you should treat the **OpenEnv spec, logging format, and Docker/HF constraints as strict contracts**.

---

## 3. High‑level project concept: Content Moderation Environment

### 3.1 Real‑world problem

Design an OpenEnv environment that simulates a **content moderation console** for a synthetic social platform (posts / comments / messages).  
The agent acts as a **moderation engine** that must decide how to handle user‑generated content while following safety policies.

This matches the requirement for “real‑world tasks humans actually perform” (content moderation, email triage, customer support etc.).[web:2][web:42]

### 3.2 Safety and data

- All content is **synthetic** (hand‑crafted or templated), with no scraped real data.  
- For highly toxic content, use symbolic tokens like `<HATE_SPEECH>`, `<SELF_HARM_REF>`, `<NSFW>` instead of detailed text.  
- Each sample is labelled with:  
  - A ground‑truth moderation decision.  
  - One or more policy tags (e.g., “spam”, “harassment”, “hate”, “self‑harm”, “benign”).

### 3.3 Episode structure

- An **episode** = moderating a small batch or thread of posts.  
- At each `step`, the environment presents one post plus its context; the agent responds with an action.  
- The environment computes a reward at **every step** and moves to the next post until the episode ends.

This gives dense reward signals and matches OpenEnv’s production‑style RL usage.[web:18][web:38]

---

## 4. Environment design

### 4.1 Pydantic models (sketch)

```python
from pydantic import BaseModel
from typing import Literal, List, Dict, Any, Optional

class ModerationAction(BaseModel):
    decision: Literal["ALLOW", "WARN", "SOFT_BLOCK", "HARD_BLOCK", "ESCALATE"]
    policy_tags: List[str] = []
    explanation: Optional[str] = None

class ModerationObservation(BaseModel):
    content: str
    metadata: Dict[str, Any]
    thread_context: List[str]
    task_id: Literal["easy", "medium", "hard"]
    step_index: int
    done: bool

class ModerationState(BaseModel):
    episode_id: str
    task_id: Literal["easy", "medium", "hard"]
    step_index: int
    total_reward: float
    catastrophic_error: bool
```

You will implement your environment class with:

- `reset(task: str | None = None) -> ModerationObservation`  
- `step(action: ModerationAction) -> tuple[ModerationObservation, float, bool, dict]`  
- `state() -> ModerationState`

### 4.2 Observation and action spaces (for README)

You can paste something like this into your README:

**Observation space**

| Field           | Type                         | Description                                      |
|----------------|------------------------------|--------------------------------------------------|
| `content`       | `str`                        | Current post/message text (synthetic).           |
| `metadata`      | `Dict[str, Any]`             | User reputation, previous strikes, platform info |
| `thread_context`| `List[str]`                  | Previous posts in the thread (for context).      |
| `task_id`       | `Literal["easy","medium","hard"]` | Current task difficulty level.                   |
| `step_index`    | `int`                        | Current step number within the episode.          |
| `done`          | `bool`                       | Whether the episode has terminated.              |

**Action space**

| Field        | Type                                    | Description                                         |
|-------------|-----------------------------------------|-----------------------------------------------------|
| `decision`  | one of `ALLOW`, `WARN`, `SOFT_BLOCK`, `HARD_BLOCK`, `ESCALATE` | Moderation action to apply.         |
| `policy_tags`| `List[str]`                            | Policy categories applicable to this content.       |
| `explanation`| `Optional[str]`                        | Optional natural‑language justification (for RL/LLM)|

---

## 5. Tasks and graders (easy → medium → hard)

### 5.1 Task 1 – Obvious binary moderation (Easy)

**Goal**: Correctly classify clearly benign vs clearly unsafe content.

- Episodes are short (e.g., 3–5 posts).  
- Each post has a **single obvious label**, like spam vs safe vs explicit abuse.  
- The agent outputs only the `decision` (tags/explanation can be ignored or optional here).

**Grader (deterministic)**

- Ground truth contains `true_decision` for each post.  
- Per‑step reward:
  - `1.0` if `decision == true_decision`.  
  - `0.5` if the decision is conservative but safe (e.g., `SOFT_BLOCK` where gold is `WARN`).  
  - `0.0` if the decision is under‑moderating (e.g., `ALLOW` where gold is `HARD_BLOCK`).
- Episode reward = average of per‑step rewards.

This task sanity‑checks your environment and makes it easy for the baseline model to get non‑trivial reward.

### 5.2 Task 2 – Decision + policy tagging (Medium)

**Goal**: Classify posts and also identify applicable policy categories.

- Posts are more nuanced (borderline spam, mild harassment, self‑promotion).  
- Agent outputs `decision` plus `policy_tags`.

**Grader**

- Ground truth has `true_decision` and `true_tags` (a set of tags).  
- Compute:
  - `decision_score`:
    - `0.7` for correct decision.  
    - `0.4` for conservative but safe decision.  
    - `0.0` for under‑moderation.  
  - `tag_score` = F1‑score between `policy_tags` and `true_tags`.  
- Per‑step reward:

  ```text
  reward = clip(0.6 * decision_score + 0.4 * tag_score, 0.0, 1.0)
  ```

- Episode reward = average per‑step reward.

This task shows richer, multi‑dimensional grading while staying deterministic.

### 5.3 Task 3 – Context‑aware thread moderation (Hard)

**Goal**: Moderate a full conversation thread where meaning is only clear from context.

- An episode is a **thread** (e.g., 5–10 messages) revealed one message at a time.  
- `thread_context` in the observation accumulates past messages.  
- Agent outputs `decision` and optionally `policy_tags` for each message.

**Grader**

- For each message `i`, your dataset stores `ideal_decision[i]` and `ideal_tags[i]`.  
- You also mark message severity (e.g., `none`, `low`, `high`).  
- For each step:
  - Start from medium task reward.  
  - Apply penalties:  
    - Severe under‑moderation on high‑severity content: large penalty (e.g., −0.7 capped).  
    - Over‑moderation on benign content: small penalty (e.g., −0.2).  
- Aggregate per‑step scores and normalise to `[0.0, 1.0]` for the episode.

This task demonstrates that your environment captures **contextual reasoning**, which is a strong differentiator.

---

## 6. Reward shaping and safety

To satisfy the “meaningful reward function” requirement:

- Give **per‑step rewards** as described above; do not only reward at episode end.[web:18]  
- Penalise degenerate behaviours:  
  - If the agent uses the same decision for all posts in an episode, subtract a small episode‑level penalty (encourages nuanced decisions).  
  - Limit max steps per episode to the number of posts (no infinite loops are possible by design).  
  - If you expose a `NO_ACTION` or `SKIP` decision, treat it as low reward except in explicitly allowed cases.

This gives dense, informative RL signals and makes your environment suitable for real training/evaluation workflows as described in the OpenEnv/TRL docs.[web:18][web:38]

---

## 7. OpenEnv spec and openenv.yaml

Based on the OpenEnv README and examples, your repo should follow a structure like:[web:10][web:41]

```text
src/moderation_env/
├── __init__.py            # Export ModerationEnv, ModerationAction, ModerationObservation
├── models.py              # Pydantic models: Action, Observation, State
├── env.py                 # Implement ModerationEnv(Environment)
├── openenv.yaml           # Environment metadata and spec
└── server/
    ├── app.py             # (optional) FastAPI/Gradio wrapper for HF Space
    └── Dockerfile         # Container definition
```

Key `openenv.yaml` contents:

- Environment metadata: name, description, authors, tags (`openenv`, `moderation`).  
- Python entrypoint: module and class paths for Environment and models.  
- Task definitions: `easy`, `medium`, `hard` with descriptions.  
- Validation: run `openenv validate` locally and in Docker.

The **`openenv-course`** repo from Hugging Face (5 modules, ~45–60 min each) shows canonical patterns for environment layout, Pydantic model usage, and validation; you can mirror its structure and gradually replace the example env with your moderation env.[web:37]

---

## 8. Baseline inference.py (root of repo)

### 8.1 Environment variables and OpenAI Client

`inference.py` in the repo root must:

- Import `OpenAI` client from `openai`.  
- Read **exactly** these env vars:
  - `API_BASE_URL` – default to `"https://api.openai.com/v1"`.  
  - `MODEL_NAME` – default to `"gpt-4.1-mini"` or similar.  
  - `HF_TOKEN` – required, no default, used as API key.  
- Initialises the client as:

```python
from openai import OpenAI
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
```

### 8.2 Inference loop + logging format

For each task (`easy`, `medium`, `hard`):

1. Print the start line:

```python
print(f"[START] task={task_name} env=content-moderation-env model={MODEL_NAME}")
```

2. Run the episode:

```python
obs = env.reset(task=task_name)
rewards = []
last_error = None
step = 1

while True:
    prompt = build_prompt_from_observation(obs)  # your function
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    action = parse_action_from_completion(completion)

    try:
        obs, reward, done, info = env.step(action)
        last_error = info.get("error") if info else None
    except Exception as e:
        reward, done = 0.0, True
        last_error = str(e)

    rewards.append(reward)

    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={last_error or 'null'}"
    )

    step += 1
    if done:
        break
```

3. Print end line and success flag:

```python
rewards_str = ",".join(f"{r:.2f}" for r in rewards)
success = (len(rewards) > 0) and (sum(rewards) / len(rewards) >= SUCCESS_THRESHOLD)

print(
    f"[END] success={str(success).lower()} "
    f"steps={len(rewards)} rewards={rewards_str}"
)
```

This satisfies the logging and environment‑variable rules described in the hackathon guide.[web:8]

---

## 9. HF Space + Docker + constraints

### 9.1 Hugging Face Space

- Build a **Gradio** or **Streamlit** UI that wraps your environment:  
  - Select task (easy/medium/hard).  
  - Show current `content` and `thread_context`.  
  - Buttons / dropdowns for `decision` and text area for tags / explanation.  
  - Display per‑step rewards and final episode reward.
- Add the `openenv` tag and any relevant topics (`moderation`, `rl-env`).[web:38]
- Turn off unnecessary Spaces to keep the build queue short and ensure your primary Space is in `Running` state before submission.[web:5][web:8]

### 9.2 Dockerfile and resources

- Use a slim Python base image (e.g., `python:3.11-slim`) to stay within **2 vCPU / 8 GB RAM**.[web:5]
- Install only what you need: `pydantic`, `openai`, `fastapi`/`gradio`, `uvicorn`, and your package.  
- Test locally:

```bash
docker build -t moderation-env .
docker run -p 7860:7860 moderation-env
```

- Run `openenv validate` inside the container to confirm spec compliance.

---

## 10. Day‑by‑day plan with time left

Assuming it is 5 April afternoon and the deadline is 8 April 23:59 IST, a realistic schedule:

### Day 1 (remainder of today)

- Finalise **problem scope**: content moderation environment + 3 tasks as described.  
- Sketch your dataset format and create a small synthetic dataset (even 10–20 examples per task).  
- Set up repo structure and Pydantic models (`models.py`).  
- Implement the environment skeleton (`env.py`) and `reset()/step()/state()` with simple dummy logic.

### Day 2

- Implement full **Task 1** end‑to‑end:  
  - Load real synthetic examples.  
  - Implement deterministic grader and reward formula.  
  - Add unit‑style tests that run multiple episodes.  
- Extend to **Task 2**: add policy tagging and corresponding grader.  
- Validate `openenv.yaml` with `openenv validate`.

### Day 3

- Implement **Task 3** (context‑aware threads) and its grader.  
- Implement `inference.py` with OpenAI Client, env vars, and logging format.  
- Create HF Space UI and wrap environment for manual testing.  
- Write README sections for overview, spaces, tasks, observation/action spaces.

### Day 4 (deadline day)

- Run everything inside Docker; fix any dependency or import issues.  
- Verify HF Space is in `Running` state and responds.  
- Re‑run `inference.py` and ensure logs strictly match the [START]/[STEP]/[END] specification.  
- Tighten README, add baseline scores, and remove dead code.  
- Submit HF Space URL via Scaler dashboard with a buffer of a few hours.

Because the hackathon allows resubmissions with no penalty, you can **submit an early working version**, then iterate and re‑submit as you polish, as long as the Space is always live at submission time.[web:5][web:8]

---

## 11. How to use the openenv-course repo effectively

The `openenv-course` repository by Hugging Face is a structured course (≈5 modules, 45–60 minutes each) that teaches how to use and build OpenEnv environments.[web:37]

Given your time, use it selectively:

- **Module 1 – Concepts**: understand OpenEnv’s goals, Gym‑style API, and spec.  
- **Module 2 – Using existing envs**: skim to see how clients interact with `reset()` / `step()`.  
- **Module 3/4 – Building and packaging envs**: mirror their directory layout, use their `openenv.yaml` and Docker examples as templates.  
- Ignore long training loops; focus on **environment structure + validation**.

Your goal is not to complete the whole course, but to use it as a **reference implementation** for:

- Pydantic model definitions.  
- Environment class layout.  
- `openenv.yaml` patterns.  
- Docker patterns and HF Space packaging.

With this document as your working spec and checklist, you can now start implementing the environment directly in your IDE, using `openenv-course` and the official OpenEnv repo as concrete references for structure and validation.
