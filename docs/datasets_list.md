# Dataset Plan for multimind (image gen + MetaMind + tokenizer + evaluation)

## Policy
- NSFW content: allowed by default for model training and generation.
- Minors/young users: strictly prohibited in sexual contexts. All datasets and ingest pipelines MUST attempt to detect and exclude images that depict minors or that are tagged/likely to represent underage characters.
- Reporting/remote hooks: opt-in only; no implicit networking.

## Image generation (G1/G2)
Recommended (anime / “moe” focus):
- Danbooru (Danbooru2018/2020/2021)
  - Use tag metadata for conditioning; includes NSFW and many tags for filtering.
  - Filter: exclude tags indicating minors (see exclude list), and optionally exclude explicit adult-only artists if you want.
- Gelbooru / Safebooru (for varied booru sources)
  - Safebooru tends to be SFW; Gelbooru contains NSFW.
- animeface / animeface200k
  - Smaller, face-focused — fast prototyping.
- Hentai collections / hentai booru (if you intend explicit NSFW)
  - Only use if you accept legal and ethical risk; must filter for minors and keep provenance.
- LAION subsets filtered for anime / adult tags
  - Good for large-scale multimodal pretraining (with careful filtering).

Real-person baselines / auxiliary:
- FFHQ, CelebA / CelebA-HQ — for face realism and discriminator pretraining (verify license).
- AVA, LSUN, AFHQ — backgrounds, animals, and scene variety.

## Discriminators / Perceptual / Coherence (A1/A2)
- CLIP (pretrained or fine-tuned on Danbooru tag captions)
- LPIPS (perceptual loss) pretrained networks
- AVA/Aesthetic datasets for quality scoring
- VGGFace2, CelebA for identity/face-consistency discriminators

## Captions / Tokenizer / Conditioning (text)
- Danbooru tags -> textual prompts (normalize and translate where needed)
- COCO captions, Conceptual Captions, CC3M/CC12M for general caption conditioning
- The Pile / OpenWebText / CC-100 for tokenizer pretraining (if MetaMind needs language capabilities)
- RedPajama / HuggingFace corpora for robust tokenizer training
- Domain-specific captions (Danbooru tags → pseudo-captions) — Transform Danbooru tags into textual prompts (e.g., “1girl, blue hair, school uniform”) to create a tagged dataset for text→image conditioning.

## MetaMind controller training data
There is no standard public “adversary-control” dataset. MetaMind benefits from training on many recorded GAN/triad runs. Build your own MetaMind dataset by logging training runs:
- What to record:
  - Per-step or per-iteration: adversary losses (A1/A2), generator losses, gradients norms, learning rates, weights, model checkpoints, metrics (FID, IS, LPIPS, CLIP-score), generated samples (or embeddings), timestamps.
  - Episode labels: stable / diverged / mode-collapsed / converged (human-labeled or heuristics).
- Sources:
  - Run many training experiments across datasets (Danbooru, FFHQ, LSUN, CelebA) and hyperparameters to build varied examples.
  - Augment with publicly available GAN training logs from GitHub repos/papers (if available).
- Format:
  - JSONL time-series entries per run, with pointers to image artifacts (local paths or hashed storages).
- Use for: supervised MetaMind (map stats→optimal adversary weights / lr multipliers), offline RL or imitation learning to replicate good control policy.
- Synthetic option:
  - Simulate “adversary behaviors” by varying noise and recording resulting losses; use curriculum to generate edge cases.

## Evaluation & metrics datasets
- Use held-out splits of training datasets (train/val) for FID/IS computation.
- Inception / ImageNet subset: often used to compute Inception Score and FID embeddings.
- Human eval: Amazon MTurk / Labelbox to collect human preference labels, realism/coherence judgements.
- Benchmark datasets:
  - FFHQ / CelebA-HQ stats for face models
  - Danbooru sampled holdout for anime style

## Safety / NSFW / filtering datasets and tools
- NSFW filtering:
  - Use tag-based filters from Danbooru and LAION metadata; train a small classifier if needed.
  - Pretrained detectors: Yahoo Open NSFW, NudeNet, or moderation models (Hugging Face moderation).
- Toxicity/safety for text:
  - RealToxicityPrompts, Jigsaw toxicity data for safety fine-tuning.
- Copyright filtering:
  - Keep provenance manifests and, if required, implement “excluded artists” lists.
- Recommendation: maintain a dataset manifest CSV/JSON with fields: source_url, license, tags, nsfw_flag, curated_flag.

## Deployment / inference auxiliary datasets
- For downstream adapters (scarlett), small curated set of sample prompts + seeds for demo images.
- Dataset of high-quality prompts and outputs for regression/regeneration tests.

## Data acquisition & tooling (practical)
- Download tools:
  - Kaggle CLI for Kaggle-hosted datasets.
  - huggingface datasets (pip install datasets) for many corpora (C4, CC-100, COCO).
  - LAION tools (laion-dl, laion-py) and S3 mirrors where available.
  - booru API clients / official Danbooru dumps.
  - gdown for Google Drive shared files.
- Storage:
  - Use hashed filenames; store original URLs/IDs in manifest.
  - Consider storing metadata (tags, captions) in a separate sqlite/JSONL manifest and images as files.
- Preprocessing & pipelines:
  - Provide reproducible preprocessing script (you have the preprocess_images.py).
  - Keep transforms/augmentations consistent between train and evaluation.

## Prioritized shortlist (fast path to prototype)
If you want to get a working “moe” generator quickly:
- Small / fast prototyping:
  1. Anime Face dataset (curated, small) — get model producing faces quickly.
  2. CelebA (if you want real-face baselines).
  3. COCO / Conceptual Caption (for conditioning experiments).
- Mid-term (multi-condition training):
  1. Danbooru subset filtered for SFW + top tags (use tags as captions/prompts).
  2. LAION subset filtered for anime-like images (if available).
- MetaMind / controller:
  - Start building your own MetaMind dataset by logging a few dozen training runs with varied hyperparams on the above image datasets. Save JSONL records per step and label outcomes.

## Recommended dataset manifest fields (store with all datasets)
- id / filename
- source_url / original_id
- dataset_name
- license
- tags / captions
- nsfw_flag
- date_downloaded
- preprocessing_version
- sha256 or hash (for provenance)
- split (train/val/test)

## Privacy / legal / ethical checklist
- Verify dataset licenses before redistribution or publishing models trained on them.
- Filter and label NSFW content if you plan public release.
- Keep auditable manifest of sources so you can respond to takedown requests or provenance queries.

## Next steps I can take for you (choose one)
- Add small scripts and manifests to the repo:
  - tools/download_examples.sh (cli examples for Kaggle/LAION/COCO),
  - tools/preprocess_images.py (you already have),
  - tools/make_meta_mind_logger.py (small logger that writes JSONL per training run).
- Produce a concrete Danbooru tag-filtering recipe (list of safe/unsafe tags and example booru queries).
- Draft a minimal MetaMind dataset schema (JSONL example) and a small generator that synthesizes training examples from existing runs.
- Provide concrete Hugging Face datasets/dl code snippets for COCO, Conceptual Captions, LAION subsets.

Which of the above would you like me to do next? (I can create the scripts/manifest files in the repo if you say “add scripts” and I will commit them to main).