# ReLook-KGC paper experiment protocol v1 (2026-09-07)

Single training seed: **41504**, as requested. This is not a multi-seed study.
Original source and old results remain in CSProm-KG-main-ReLook-filtered-utility.

## Architecture and claims

1. Progressive Structural Reasoning: structural conditional prompts at every
   frozen BERT layer, followed by a ConvE decoder.
2. Directional Reasoning Experts: train the shared upper-layer expert, then
   head/tail residual experts, retaining the learned shared path.
3. Selective Visual Adapter: train visual compatibility, relation-role prior,
   bounded query utility, visual utility prior, and joint evidence routing.
   Calibrate invocation and positive residual trust bounds on validation only.

The all-layer structural prompt mechanism already exists in CSProm-KG. This
suite explicitly tests that implementation against the official forward and
does **not** interpret renaming it as an independently new architecture.
The progressive multimodal staging and the later modules require their own
empirical comparisons.

## Data, initialization and objective

- DB15K, YAGO15K, FB15k-237 use the existing complete splits and text/images.
- Fresh 256-d ComplEx structural initialization is trained for 50 epochs using
  training triples only (reciprocal relations included); all suite variants for
  a dataset use these exact same exported embeddings. No historical checkpoint
  is used as a warm start. SHA-256 hashes record all inputs.
- Main ReLook objective: mean negative log-probability of all known training
  answers, label smoothing 0.1; uniform head/tail weight 1.0, both directions.
- Utility labels: train-filtered RR only; evaluation: all-known filtered RR.
- Text encoder: local bert-base-uncased (12 layers); prompt length 10 per
  source/relation, bottleneck 128, embeddings 256, ConvE grid 16 x 16.
- Visual correction scope: **global**, consistently on all three datasets,
  matching the complete DB15K joint-router implementation. Top-128 is used for
  the auxiliary visual objective and diagnostics, not a claim that global
  scoring only computes 128 candidates.

## Frozen optimization budget

- Backbone controls: AdamW lr 5e-4; max 40 epochs DB15K/YAGO15K and 60 FB15k-237;
  min 10 epochs; early stop after 6 validation checks without >1e-4 MRR gain.
- Expert stages: AdamW lr 1e-4, max 6 epochs, patience 3. Shared-extension control
  receives the same two-stage training allowance as the hierarchical variant.
- Visual compatibility: lr 1e-4, max 6 epochs, patience 3; role path fixed full.
- Relation router, bounded utility, visual prior and joint router: each lr
  1e-4, max 3 epochs, patience 2. Query residual scale 0.25.
- Frozen backbone and experts remain in eval mode during subsequent training.
- Sampling is reseeded after each model constructor. The official recipe
  control copies canonical random common weights and uses the same frozen-BERT
  dropout regime; its substantive training difference is CE+LAR versus set loss.
- Backbone/experts use AMP after finite-gradient smoke checks; visual and
  routing stages use FP32. All use the project venv; Conda is prohibited.
- Train batch 512, eval batch 128, 4 workers. Scheduler permits one task per
  assigned GPU and checks available memory; smoke profiles precede full runs.

## Core controls

- All three datasets: All-12 and the official CSProm recipe control.
- Additional DB15K structural controls: Mixed, Early-6, Late-6.
- All-12 is also the matched CSProm structural forward. The separate official
  control uses its original CE+LAR objective (alpha .1, 8 LARs), with matched
  encoder, embedding dimension, train-only initialization and convergence
  budget. It is labeled as an objective/recipe comparison, not a new prompting
  mechanism or an exact reproduction of the paper's tuned numerical score.
- DB15K experts: Shared, Role-only, Hierarchical, Shared extended training.
- All three datasets: train and evaluate the complete architecture and the
  cumulative backbone/shared/hierarchical/visual/routing stages.

## Selection and evaluation

- Every checkpoint chosen on full validation MRR. No test metrics are used in
  learning, stopping, threshold calibration, or stage selection.
- Final calibration grid: visual blend {0.1, 0.25, 0.5, 1.0}; threshold
  {0, .1, .25, .5, .75, 1.01}; tail/head trust cap {-1 (unbounded), 0, .1, .25}.
- Fixed full architecture reported even if a cumulative module fails to improve.
  Any validation-selected no-vision deployment is explicitly identified.
- Store config, convergence, gradient checks, selected checkpoint, per-query
  ranks, aggregate valid/test metrics, source and input hashes.
- Training checkpoint selection uses the recorded training precision. The
  final held-out tables and visual calibration consistently use FP32. They are
  not copied from rounded console tables or old mixed-precision run histories.

## Selection strategy and actual efficiency (after main results)

- Compare no-vision, always-on, random, entropy, inverse margin, and learned
  selection at identical invocation budgets (.25, .5, .75), plus the selected
  deployment setting. Random selection seed fixed independently of targets.
- Equal budgets are enforced per evaluation batch of 128 (round-to-even quota),
  among image-eligible queries, with stable tie ordering. Random priorities are
  deterministic source/relation hashes. Actual invocation rates are reported.
- A final-score replay check compares calibration caches against the actual
  selected forward on the first tail/head batches before reporting timings.
- Replay exact held-out ranks for risk/accuracy; do not call invocation count
  a latency measurement. Thresholds selected on validation, test used once.
- Actual inference benchmark: same dedicated GPU, same batch and inputs,
  warmup, synchronized CUDA timing, repetitions; report ms/query, throughput,
  peak memory, trainable and effective parameter counts. Model-only timings
  are labeled separately from tokenization/data-loading timings.
- Original WN18 experiments remain exploratory and are not in this main suite.
