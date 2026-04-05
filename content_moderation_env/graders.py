from typing import Set, List
import math

def grade_easy(action_decision: str, true_decision: str) -> float:
    """Deterministic grader for Easy task."""
    if action_decision == true_decision:
        return 1.0
    
    # Conservative but safe
    if true_decision == "WARN" and action_decision in ["SOFT_BLOCK", "HARD_BLOCK", "ESCALATE"]:
        return 0.5
    if true_decision == "SOFT_BLOCK" and action_decision in ["HARD_BLOCK", "ESCALATE"]:
        return 0.5
        
    return 0.0

def grade_medium(action_decision: str, action_tags: List[str], true_decision: str, true_tags: List[str]) -> float:
    """Deterministic grader for Medium task."""
    # Decision Score
    decision_score = 0.0
    if action_decision == true_decision:
        decision_score = 0.7
    elif true_decision in ["WARN", "SOFT_BLOCK"] and action_decision in ["HARD_BLOCK", "ESCALATE", "SOFT_BLOCK"]:
         decision_score = 0.4
    else:
        decision_score = 0.0
        
    # Tag Score (F1 approximation)
    pred_set = set(action_tags)
    true_set = set(true_tags)
    
    if not true_set and not pred_set:
        tag_score = 1.0
    elif not true_set or not pred_set:
        tag_score = 0.0
    else:
        intersection = len(pred_set.intersection(true_set))
        precision = intersection / len(pred_set)
        recall = intersection / len(true_set)
        if precision + recall > 0:
            tag_score = 2 * (precision * recall) / (precision + recall)
        else:
            tag_score = 0.0
            
    reward = (0.6 * decision_score) + (0.4 * tag_score)
    return min(max(reward, 0.0), 1.0)

def grade_hard(action_decision: str, action_tags: List[str], true_decision: str, true_tags: List[str], severity: str) -> float:
    """Deterministic grader for Hard task."""
    base_score = grade_medium(action_decision, action_tags, true_decision, true_tags)
    
    penalty = 0.0
    # Over-moderation on benign content
    if true_decision == "ALLOW" and action_decision != "ALLOW":
        penalty += 0.2
        
    # Severe under-moderation on high severity
    under_mod = ["ALLOW", "WARN"]
    if severity == "high" and action_decision in under_mod and true_decision not in under_mod:
        penalty += 0.7
        
    final_score = base_score - penalty
    return min(max(final_score, 0.0), 1.0)
