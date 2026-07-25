# Preregistration round 2: learned representations and within-session signal

Committed before any new footage is downloaded, any embedding is
computed, or any new-session model output is seen. Two experiments.

## Experiment A: pretrained embeddings replace hand-crafted features

Hypothesis under test: the behavioral nulls of the current paper are an
artifact of hand-crafted summary features, and a learned representation
recovers out-of-session signal.

Fixed plan:
1. Representation: DINOv2 ViT-S/14, frozen, applied to the
   identity-gated face crop and the pose crop of each accepted frame in
   each covered decision window of the existing four sessions; per-window
   pooling is the mean and standard deviation over frames of the CLS
   embedding, PCA-reduced to 32 components fit on training folds only.
2. Identity gating, coverage threshold (0.5), covered-row set, targets,
   LOSO evaluation, logistic probe, and hand-grouped bootstrap CIs are
   identical to the current paper. No fine-tuning of any weights.
3. Primary test: delta AUC of betting + embedding-probe over the betting
   baseline for is_weak on the pooled four sessions. Secondary: delta
   Spearman rho for continuous equity; within-session grouped CV AUC of
   the embedding probe alone.
4. Interpretation rule: a positive delta with CI excluding zero
   contradicts the paper's feature-artifact reading and will be reported
   as such; a null or negative delta extends the paper's conclusion from
   hand-crafted features to a standard learned representation. Either
   way the result is added to the paper.
5. Permitted engineering validation before results: running the
   extractor on a handful of windows to verify shapes, gating, and
   pooling mechanics, without computing any evaluation metric.

## Experiment B: within-session signal on new sessions

The current paper's one unresolved estimate is within-session
behavior-only AUC 0.574, CI [0.492, 0.652]. This experiment adds
sessions to resolve it.

Fixed plan:
1. Selection: full Hustler Casino Live VODs in the frozen 2026 graphics
   layout (probe frames may be inspected for layout and roster only),
   prioritized by presence of the primary player, then by expected
   camera coverage. Target: at least four new sessions.
2. Each new session runs the frozen pipeline at the current commit, with
   the seat-map discipline of round 1: commit-moment identification,
   deterministic enrollment, chip-level visual verification, and a
   distractor audit sweep, all before any model output.
3. Primary test: pooled within-session grouped 5-fold (by hand)
   behavior-only AUC for is_weak across ALL sessions old and new, with
   hand-grouped bootstrap CI. Resolution criterion: the CI excludes 0.5
   in either direction, or the total covered n exceeds 600 (at which
   point the interval width is expected near 0.10 and a null is
   reported as a bounded null).
4. Secondary, unchanged from the paper: the cross-session transfer and
   session-locality analyses rerun on the enlarged pool.
5. The face-and-pose out-of-session delta is NOT re-litigated: round 1's
   preregistered failure stands regardless of these outcomes.

## Not allowed

Threshold tuning, coverage-cutoff changes, model swaps, or dropping
sessions after seeing results. Pipeline bug fixes forced by new footage
are committed, documented, and trigger a full rerun, as in round 1.


## Results (recorded 2026-07-25, after both tests ran as specified)

Experiment A: NULL. Pooled LOSO delta of the DINOv2 probe over the
betting baseline on the four original sessions: +0.012, hand-grouped
bootstrap 95% CI [-0.079, +0.103], n=257 embedding-covered decisions.
Per interpretation rule 4 this extends the feature-artifact conclusion
to a standard learned representation. Nuance recorded: the embedding
probe alone transfers at 0.545 (not inverted, unlike hand-crafted
features), and within sessions it scores 0.398, CI [0.320, 0.476],
below chance, consistent with high-dimensional instability at small n.

Experiment B: RESOLVED POSITIVE. Pooled within-session grouped 5-fold
behavior-only AUC across all eight sessions: 0.587, hand-grouped
bootstrap 95% CI [0.515, 0.655], n=494. The CI excludes 0.5, meeting
resolution criterion 3. Per-session AUCs range 0.26 to 0.76. The
secondary cross-session arm on the enlarged pool remains at or below
chance (behavior-only LOSO 0.430, CI [0.358, 0.502]; face and pose
delta over betting -0.034, CI [-0.080, +0.012]).
