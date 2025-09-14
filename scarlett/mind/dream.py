"""Dream (offline generative rehearsal) placeholder.

Purpose:
- Produce synthetic scenario rollouts (episodes) for internal rehearsal without external data.
- Feed generated trajectories back into outcome / relation / emotion modules for adaptation or analysis.

Conceptual operations:
def generate(seed_state, n_episodes:int, horizon:int) -> list[Episode]
def score(episode) -> dict[str,float]  # engagement proxy, diversity, surprise

Episode sketch (future):
{
  'states': ndarray[horizon, emb_dim],
  'actions': ndarray[horizon, action_dim],
  'emotion_logits': ndarray[horizon, E],
  'meta': {...}
}

Initial sandbox may use random walks in latent space constrained by simple norms.
No implementation yet."""
