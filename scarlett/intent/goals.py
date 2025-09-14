"""Goals management placeholder.

Purpose:
- Represent hierarchical goals (long_term, mid_term, immediate) each with vector form + priority scalar.
- Translate active goals into intent drafts before desire weighting.
- Maintain a goal stack / queue and basic arbitration (e.g., priority decay, satisfaction progress).

Planned interfaces (conceptual):
class Goal: id, level(str), embedding(ndarray), priority(float), progress(float)
def select(active_goals:list[Goal]) -> list[Goal]  # choose subset for current tick
def integrate(goals, fused_emb) -> intent_drafts (list / aggregated vector)

No implementation yet."""
