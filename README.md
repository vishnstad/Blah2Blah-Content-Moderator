---
title: Blah2Blah Content Moderator
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - moderation
  - rl-env
---

# Content Moderation Core Environment

A custom Content Moderation RL Environment for the Meta PyTorch OpenEnv Hackathon.

## Authors
**Team Blah2Blah**
- Asmi K (asmi114sonu@gmail.com)
- Vishal S (gemsid2023@gmail.com)

## Overview
This OpenEnv environment simulates a content moderation console for a synthetic social platform. The agent acts as a moderation engine that decides how to handle user-generated content (posts, comments, messages) while adhering to given safety policies.

Each episode is a batch of content or a single thread of posts. At each step, the model observes the content, its metadata, and thread context (if applicable), acting by choosing a moderation decision and tagging violated policy tags.

## Observation Space

| Field           | Type                         | Description                                      |
|----------------|------------------------------|--------------------------------------------------|
| `content`       | `str`                        | Current post/message text (synthetic).           |
| `metadata`      | `Dict[str, Any]`             | User reputation, previous strikes, platform info |
| `thread_context`| `List[str]`                  | Previous posts in the thread (for context).      |
| `task_id`       | `Literal["easy","medium","hard"]` | Current task difficulty level.                   |
| `step_index`    | `int`                        | Current step number within the episode.          |
| `done`          | `bool`                       | Whether the episode has terminated.              |

## Action Space

| Field        | Type                                    | Description                                         |
|-------------|-----------------------------------------|-----------------------------------------------------|
| `decision`  | `ALLOW`, `WARN`, `SOFT_BLOCK`, `HARD_BLOCK`, `ESCALATE` | Moderation action to apply. |
| `policy_tags`| `List[str]`                            | Policy categories applicable to this content.       |
| `explanation`| `Optional[str]`                        | Optional justification (for RL/LLM)                 |

## Tasks

### 1. Easy Task 
**Goal:** Correctly classify clearly benign vs clearly unsafe content. Single obvious labels like spam vs explicitly safe.
- **Grader:** Deterministic. 1.0 for match, 0.5 for conservative-but-safe.

### 2. Medium Task
**Goal:** Classify more nuanced posts (borderline spam, hidden self-promotion) and accurately tag applicable policies.
- **Grader:** 60% based on decision correctness, 40% based on F1-score of the policy tags matching true tags.

### 3. Hard Task
**Goal:** Context-aware Moderation. Evaluate escalating conversation threads where previous messages change meaning.
- **Grader:** Builds on Medium rules, but applies severe penalties (-0.7) for under-moderating high-severity content. Over-moderation incurs slight structural penalties.

## Setup & Testing Locally

You can spin up the environment with Docker and access the Gradio Interface interactively:

```bash
docker build -t moderation-env .
docker run -p 7860:7860 moderation-env
```

Navigate to `http://localhost:7860` to access the environment UI.

## Inference Validation
To validate run:
```bash
export HF_TOKEN=your_hugging_face_token_here
python inference.py
```
This script will loop over all tasks and use standard `[START]`, `[STEP]`, and `[END]` logging syntax required by the Hackathon.
