"""
Gradio-based interactive UI for the Content Moderation Environment.
This is the main entry point for the Hugging Face Space (Docker).
"""
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import json
from content_moderation_env.server.moderation_environment import ModerationEnvironment
from content_moderation_env.models import ModerationAction

env = ModerationEnvironment()
current_obs = None


def reset_env(task):
    global current_obs
    current_obs = env.reset(task=task)
    return update_ui()


def step_env(decision, tags, explanation):
    global current_obs
    if current_obs and current_obs.done:
        return update_ui() + ("⚠️ Episode is done. Click Reset to start a new one.",)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    action = ModerationAction(
        decision=decision,
        policy_tags=tag_list,
        explanation=explanation or None,
    )
    current_obs = env.step(action)
    status = ""
    if current_obs.done:
        avg = env.state.total_reward / max(env.state.step_count, 1)
        status = f"✅ Episode finished! Average reward: {avg:.2f}"
    return update_ui() + (status,)


def update_ui():
    if current_obs is None:
        return "Click **Reset / Start Episode** to begin.", "{}", "—", 0.0, False

    content_display = f"### 📝 Content to Moderate\n\n> {current_obs.content}"
    if current_obs.thread_context:
        content_display += "\n\n**Thread Context:**\n"
        for msg in current_obs.thread_context:
            content_display += f"- _{msg}_\n"

    metadata_display = json.dumps(current_obs.metadata, indent=2)
    step_info = f"**Task:** `{current_obs.task_id}` · **Step:** {current_obs.step_index} / {len(env.dataset)}"

    return content_display, metadata_display, step_info, current_obs.reward, current_obs.done


with gr.Blocks(
    title="🛡️ Content Moderation Environment – Team Blah2Blah",
    theme=gr.themes.Soft(),
) as interface:
    gr.Markdown(
        "# 🛡️ Content Moderation Agentic Environment\n"
        "**Team Blah2Blah** · Meta PyTorch OpenEnv Hackathon x Scaler"
    )

    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=["easy", "medium", "hard"],
            value="easy",
            label="Task Difficulty",
        )
        reset_btn = gr.Button("🔄 Reset / Start Episode", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            content_box = gr.Markdown("Click **Reset / Start Episode** to begin.")
        with gr.Column(scale=1):
            metadata_box = gr.Code(label="User Metadata (JSON)", language="json")
            step_box = gr.Markdown("—")

    with gr.Row():
        with gr.Column():
            decision_radio = gr.Radio(
                ["ALLOW", "WARN", "SOFT_BLOCK", "HARD_BLOCK", "ESCALATE"],
                label="Decision",
                value="ALLOW",
            )
            tags_input = gr.Textbox(
                label="Policy Tags (comma separated)",
                placeholder="e.g. spam, harassment",
            )
            explanation_input = gr.Textbox(
                label="Explanation (optional)",
                placeholder="Short justification…",
            )
            submit_btn = gr.Button("✅ Submit Action", variant="secondary")

        with gr.Column():
            reward_slider = gr.Slider(
                minimum=0.0, maximum=1.0, label="Step Reward", interactive=False
            )
            done_checkbox = gr.Checkbox(label="Episode Done", interactive=False)
            status_box = gr.Markdown("")

    reset_btn.click(
        reset_env,
        inputs=[task_dropdown],
        outputs=[content_box, metadata_box, step_box, reward_slider, done_checkbox],
    )

    submit_btn.click(
        step_env,
        inputs=[decision_radio, tags_input, explanation_input],
        outputs=[content_box, metadata_box, step_box, reward_slider, done_checkbox, status_box],
    )


if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
