Scarlett Prototype Scaffold (Sandbox)

Scope: Minimal architectural skeleton to exercise internal cognitive/expert routing ideas. Explicitly excludes external safety / moral policy layers (to be integrated later per whitepaper validation). Filenames kept short, no underscores.

Directories:
- core: Shared primitive types & config placeholders
- route: Global routing & gating surfaces
- intent: Intent formation & desire/agency modeling stubs + goals registry
- sim: Predictive simulation (theory of mind, outcomes, emotions, relations)
- perception: Multimodal feature providers (audio, vision, fusion) – placeholder abstractions
- language: Language model integration surface & semantic utilities (LLM deferred)
- decision: Action synthesis & action space definition
- mind: Meta/self model (metmind) + offline dream/simulation substrate
- diag: Internal diagnostics / drift / generation of internal tests (not external safety)

Deferred (future): safety, moral, policy, filter, audit, persistence, memory, tooling, plugin api.

New conceptual modules:
- mind.metmind: global self-representation, meta-cognitive signals (confidence, coherence, novelty)
- mind.dream: offline / background generative scenario rehearsal feeding synthetic trajectories back into sim & intent
- intent.goals: hierarchical goal objects (long, mid, immediate) -> translation to intent drafts -> reconciled with desire weights

Guiding Constraints (current sandbox):
1. Keep parameter counts tiny (debug scale) until interfaces stabilize.
2. No coupling between perception, simulation, and decision modules except via typed contracts in core.types.
3. All modules side-effect free (pure forward-style) in this phase.
4. No external network calls; all placeholders local.
5. Moral / safety constructs intentionally omitted.

Next Potential Steps (after interface review):
- Define dataclasses in core.types (Embeddings, IntentDraft, Goal, GoalStack, SimulationBundle, DecisionLogits, Rationale).
- Implement lightweight in-memory router in route.router using async dispatch.
- Add seed control & determinism harness in diag.
- Introduce minimal training loop harness (optional) outside of decision layer.
- Specify metmind signal schema (e.g., coherence_score, novelty_score, stability_index) feeding gating.
- Define dream batch interface (generate_scenarios -> list[SimulatedEpisode]).

Rename Note: Existing legacy file 'global_router' (with underscore) retained for now; can be migrated into route/router.py after confirmation.

This file structure is a proposal; adjust before adding real code.
