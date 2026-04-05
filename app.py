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
    action = ModerationAction(
        decision=decision,
        policy_tags=[t.strip() for t in tags.split(",") if t.strip()],
        explanation=explanation
    )
    current_obs = env.step(action)
    return update_ui()

def update_ui():
    if current_obs is None:
        return "No content", "", "", 0.0, False
        
    content_display = f"**Content:**\n\n{current_obs.content}"
    if current_obs.thread_context:
        content_display += f"\n\n**Thread Context:**\n"
        for idx, msg in enumerate(current_obs.thread_context):
             content_display += f"> {msg}\n"
             
    metadata_display = json.dumps(current_obs.metadata, indent=2)
    step_info = f"Task: {current_obs.task_id} | Step: {current_obs.step_index}"
    
    return content_display, metadata_display, step_info, current_obs.reward, current_obs.done

with gr.Blocks(title="Content Moderation OpenEnv") as interface:
    gr.Markdown("# Content Moderation Agentic Environment (OpenEnv)")
    
    with gr.Row():
        task_dropdown = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Task Difficulty")
        reset_btn = gr.Button("Reset / Start Episode")
        
    with gr.Row():
        with gr.Column(scale=2):
            content_box = gr.Markdown("Click Reset to start.")
        with gr.Column(scale=1):
            metadata_box = gr.Code(label="Metadata", language="json")
            step_box = gr.Markdown("Task Summary")
            
    with gr.Row():
        with gr.Column():
            decision_radio = gr.Radio(["ALLOW", "WARN", "SOFT_BLOCK", "HARD_BLOCK", "ESCALATE"], label="Decision", value="ALLOW")
            tags_input = gr.Textbox(label="Policy Tags (comma separated)")
            explanation_input = gr.Textbox(label="Explanation (optional)")
            submit_btn = gr.Button("Submit Action", variant="primary")
            
        with gr.Column():
            reward_slider = gr.Slider(minimum=0.0, maximum=1.0, label="Step Reward", interactive=False)
            done_checkbox = gr.Checkbox(label="Episode Done", interactive=False)
            
    reset_btn.click(
        reset_env, 
        inputs=[task_dropdown], 
        outputs=[content_box, metadata_box, step_box, reward_slider, done_checkbox]
    )
    
    submit_btn.click(
        step_env,
        inputs=[decision_radio, tags_input, explanation_input],
        outputs=[content_box, metadata_box, step_box, reward_slider, done_checkbox]
    )

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
