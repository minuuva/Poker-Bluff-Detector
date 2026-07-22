# Preregistration: iteration 4 replication test

Written and committed before any iteration 4 footage was downloaded or
processed. The point of this document is that the hypothesis and analysis
were fixed in advance, so the coming result, whichever way it lands,
cannot be quietly reshaped by looking at the data first.

## Hypothesis

On sessions unseen by every previous analysis, adding face and pose
features to the betting-only baseline improves out-of-session AUC for the
is_weak target. The registered point estimate from iterations 1 to 3 is
delta AUC +0.096, 95% CI [+0.005, +0.181], obtained after cross-player
attribution was cleaned with distractor references.

## Fixed analysis plan

1. New sessions: full Hustler Casino Live VODs featuring Nik Airball,
   2026 broadcast layout, not previously downloaded or inspected beyond
   confirming the roster and layout.
2. Pipeline settings frozen at commit f54b780: same ROIs, template
   library, assembler, labels, coverage threshold 0.5, per-player
   z-scoring, logistic model, leave-one-session-out evaluation.
3. Seat maps for new sessions are built from commit-moment evidence
   before any model output is examined, and every lookalike neighbor
   found inside a search region is registered as a distractor at seat-map
   time, not after results are seen.
4. The test: pooled leave-one-session-out delta AUC (face and pose
   features over betting baseline) for is_weak on behavior-covered
   decisions of all sessions combined, with the hand-grouped bootstrap
   95% CI. Replication supports the hypothesis if the pooled delta stays
   positive with a CI excluding zero; a CI that includes zero is reported
   as a failed replication, full stop.
5. Event-detector features remain excluded from the primary test (they
   were null in iteration 3); they may be reported as a secondary
   exploratory arm, labeled as such.

## Session substitution log

2026-07-22, before any extraction: ZeCmz2DA6Zg (May 30) was downloaded
and found to use a different broadcast graphics package (bottom-left
panel block, relocated pot box); pot and stakes do not parse with the
frozen ROI layout, so it fails selection criterion 1 and is excluded on
technical grounds. Replaced by CyQRb173NfA (Apr 25, Jungleman, Nik
Airball, Big Mike), whose probe frames parse with the frozen layout.
The substitution was decided from probe frames only, before any
extraction, labeling, or model output on either session.

## What is not allowed

Re-tuning thresholds, swapping models, moving the coverage cutoff,
dropping windows, or reclassifying labels after seeing new-session
results. Any pipeline bug fix forced by the new footage is documented in
the commit history and the test is rerun from scratch.
