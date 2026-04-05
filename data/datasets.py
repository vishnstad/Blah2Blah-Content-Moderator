"""
Synthetic datasets for the Content Moderation Environment.
"""
from typing import Dict, Any, List

EASY_DATASET = [
    {
        "content": "Check out my new blog post about learning Python!",
        "metadata": {"user_age_days": 150, "reputation": 5},
        "thread_context": [],
        "true_decision": "ALLOW",
        "true_tags": [],
        "severity": "none"
    },
    {
        "content": "Click here to win a free iPhone! http://spam-link.com",
        "metadata": {"user_age_days": 0, "reputation": 0},
        "thread_context": [],
        "true_decision": "HARD_BLOCK",
        "true_tags": ["spam", "scam"],
        "severity": "high"
    },
    {
        "content": "You are stupid and I hate you.",
        "metadata": {"user_age_days": 20, "reputation": 1},
        "thread_context": [],
        "true_decision": "HARD_BLOCK",
        "true_tags": ["harassment", "abuse"],
        "severity": "high"
    },
    {
        "content": "Does anyone know what time the library opens?",
        "metadata": {"user_age_days": 300, "reputation": 10},
        "thread_context": [],
        "true_decision": "ALLOW",
        "true_tags": [],
        "severity": "none"
    },
    {
        "content": "Hey guys, buy my crypto course for 50% off!",
        "metadata": {"user_age_days": 5, "reputation": -1},
        "thread_context": [],
        "true_decision": "SOFT_BLOCK",
        "true_tags": ["spam", "self_promotion"],
        "severity": "low"
    }
]

MEDIUM_DATASET = [
    {
         "content": "I sell herbal supplements that cure everything. DM for prices.",
         "metadata": {"user_age_days": 10, "reputation": 0},
         "thread_context": [],
         "true_decision": "SOFT_BLOCK",
         "true_tags": ["spam", "medical_misinfo"],
         "severity": "low"
    },
    {
         "content": "I'm so frustrated with the government, I wish we could just overthrow them.",
         "metadata": {"user_age_days": 500, "reputation": 20},
         "thread_context": [],
         "true_decision": "WARN",
         "true_tags": ["political_discussion", "borderline_violence"],
         "severity": "medium"
    },
    {
         "content": "Here is a recipe for chocolate chip cookies.",
         "metadata": {"user_age_days": 100, "reputation": 5},
         "thread_context": [],
         "true_decision": "ALLOW",
         "true_tags": ["benign"],
         "severity": "none"
    },
    {
         "content": "I'm thinking about ending it all tonight.",
         "metadata": {"user_age_days": 2, "reputation": 0},
         "thread_context": [],
         "true_decision": "ESCALATE",
         "true_tags": ["self_harm"],
         "severity": "high"
    },
    {
         "content": "My neighbor is so annoying. I should go over there and punch him.",
         "metadata": {"user_age_days": 50, "reputation": 2},
         "thread_context": [],
         "true_decision": "WARN",
         "true_tags": ["harassment", "violence"],
         "severity": "low"
    }
]

HARD_DATASET = [
    # Thread 1: Escalating argument
    {
        "content": "Your code is terrible, you should read the docs.",
        "metadata": {"user_age_days": 100, "reputation": 10},
        "thread_context": [],
        "true_decision": "WARN",
        "true_tags": ["unprofessional", "minor_harassment"],
        "severity": "low"
    },
    {
        "content": "Excuse me? I've been coding for 10 years.",
        "metadata": {"user_age_days": 50, "reputation": 5},
        "thread_context": ["Your code is terrible, you should read the docs."],
        "true_decision": "ALLOW",
        "true_tags": ["defensive"],
        "severity": "none"
    },
    {
        "content": "Well 10 years implies you're just old and dumb.",
        "metadata": {"user_age_days": 100, "reputation": 10},
        "thread_context": ["Your code is terrible, you should read the docs.", "Excuse me? I've been coding for 10 years."],
        "true_decision": "SOFT_BLOCK",
        "true_tags": ["harassment", "insult"],
        "severity": "medium"
    },
    {
        "content": "I know where you live, watch your back.",
        "metadata": {"user_age_days": 100, "reputation": 10},
         "thread_context": ["Your code is terrible, you should read the docs.", "Excuse me? I've been coding for 10 years.", "Well 10 years implies you're just old and dumb."],
        "true_decision": "HARD_BLOCK",
        "true_tags": ["threat", "violence"],
        "severity": "high"
    }
]

def get_dataset(task_id: str) -> List[Dict[str, Any]]:
    if task_id == "easy":
        return EASY_DATASET
    elif task_id == "medium":
        return MEDIUM_DATASET
    elif task_id == "hard":
        return HARD_DATASET
    else:
        raise ValueError(f"Unknown task: {task_id}")
