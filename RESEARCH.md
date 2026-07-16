# Research: Replicating the 2026 WSOP "AI Bluff Detector"

*Deep-research pass completed 2026-07-16. 24 sources fetched, 117 claims extracted; the most
load-bearing claims were verified directly against primary sources (noted inline).*

---

## 1. The original system (verified against primary sources)

Built by **Luke Geel**, an AI engineer for the US Air Force (math/econ degrees, Masters in AI
in progress at Johns Hopkins), solo, on a regular Apple MacBook. Used by **Omaha Productions**
(Peyton Manning) in ESPN's 2026 WSOP Main Event coverage. This was ESPN's first WSOP broadcast
since 2021, with 100+ hours of coverage starting July 2 and the final table airing Aug 3-5.

**Architecture, in his own words** ([PokerOrg, April 16, 2026](https://www.poker.org/latest-news/ai-insider-can-your-poker-tells-be-hacked-aWAYz3t1S250), all quotes verified verbatim):

1. **OCR of broadcast graphics**: "getting the program to be able to actually watch the stream
   and read the information like bet sizes, how much is in the pot, what the cards are."
2. **Frame-level behavioral features**: hands activity, mouth shape, smile symmetry ("a
   symmetrical smile is usually more genuine"), gaze direction, blink rate, chip movement speed
   when betting, hand clench/tension, whether he's talking.
3. **ML mapping features to hand-strength classes**: "more than whether he's bluffing or not.
   Like, does he have a really good draw, for example, or a good but not a great hand?"

**Training data**: a single player, **Nik Airball on Hustler Casino Live**, chosen because he
plays many hands, bluffs a lot, and "has voluntarily played in front of cameras, with exposed
hole cards, for many, many hours."

**Timeline nuance (matters for honest framing)**:
- **March 17, 2026** (his first PokerOrg column): frames face/behavior-scanning AI as a
  *hypothetical future* capability.
- **April 16, 2026**: describes the build; the ML component was *still unfinished* and WSOP
  producers had merely "been in touch."
- **July 2026** (Sportico, July 3; corroborated by ZME Science and PokerScout, July 8): deployed
  in coverage after roughly 6 months of development, applied **only to players already
  eliminated** ("for game integrity reasons"). Geel: "It was significantly more difficult than
  I had initially hoped... I can't just, like, upload a YouTube URL and say, 'find their
  tells.'"
- Heart rate and skin flushness (rPPG-style signals) are **not** in the deployed system; they
  are named only as future possibilities.

**Geel's stated ethical constraints**: only analyzes players who willingly appeared on stream
with exposed hole cards; no real-time table use ("When you get to the table, though, you should
be on your own"); "There's preparation, and there's cheating."

**Prior art he didn't mention**: a peer-reviewed poker bluff detection dataset already existed.
[Feinland et al., ICIAP 2022 (Binghamton University)](https://link.springer.com/chapter/10.1007/978-3-031-06433-3_34)
built a facial-analysis dataset from broadcast tournament footage and trained an InceptionV3
regression, reporting MSE 0.0288 without a chance or action-only baseline, which makes the
number uninterpretable. A useful example of what *not* to do in the eval.

## 2. Scientific grounding

### Slepian et al. 2013, *Psychological Science* (verified against the paper PDF)

Observers judged 20-22 silent ~1.6 s clips of players betting (2009 WSOP), against the
broadcast's shown win-probability as ground truth:

| Condition | Mean accuracy r [95% CI] |
|---|---|
| Face only | **-.07** [-.15, .01], "nearly worse than chance" (b = -0.74, p = .07, marginal) |
| Upper body (face + arms) | **.02** [-.06, .09], at chance |
| Arms only (Study 1 / Study 2) | **.07** [.01, .14] / **.15** [.11, .19] |
| Rated player confidence (Study 3) | .15 [.07, .24] |
| **Rated movement smoothness (Study 3)** | **.29** [.22, .36], the strongest cue tested |

Three design implications:
- **Chip-motion smoothness is the highest-value behavioral feature**, not the face.
- **Fusing face and body naively can cancel the signal** (upper-body was at chance while
  arms-only beat chance; the deceptive face signal drowned the diagnostic arm signal).
- Caveat: "smoothness" was a *subjective rating*, never kinematically defined. Quantifying it
  (jerk and spectral arc length on wrist trajectories) is a genuine small contribution.

**Criticisms** ([Zachary Elwood](https://www.readingpokertells.com/2021/02/criticisms-of-michael-slepians-stanford-study-on-poker-tells-and-hand-movements-published-2015/)):
labels were broadcast equity-vs-opponent (a strong hand codes "weak" if the opponent holds
better); clips may leak bet size (chip volume visible); "professional players" is overstated
(roughly 20% of Main Event fields are pros); smoothness is confounded with speed and
positioning. Elwood accepts the correlation likely exists but calls it *very weak* and only
usable **with a player-specific behavioral baseline**, which supports per-player z-scoring as
a core design decision.

### The deception-detection literature: realistic ceilings

- Human lie detection: **54%** average accuracy across 25,000+ judgments (Bond and DePaulo
  2006 meta-analysis).
- All nonverbal deception cues are "faint and unreliable" (DePaulo et al. 2003, 116 studies;
  Vrij et al. 2019). Ekman-style facial micro-expression lie detection **has no scientific
  support**. One prominent review ([Denault et al. 2020, Frontiers in Psychology](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2020.613410/full))
  recommends abandoning nonverbal deception research entirely; cite this as the strong null
  hypothesis the project tests against.
- **DOLOS** ([ICCV 2023](https://arxiv.org/abs/2303.12745)), the largest gameshow deception
  dataset (1,675 clips, 213 subjects): best method 66.8% acc / 73.4 F1. Visual-only 61.4%,
  audio-only 59.2%, fusion 64.8%, so fusion adds a few points. **Facial landmarks
  underperformed raw RGB face frames** (in-domain).
- **Cross-domain benchmark** ([arXiv 2405.06995](https://arxiv.org/abs/2405.06995), 6 datasets):
  best fusion accuracy 56.8-59.0%; landmarks consistently weak; DINOv2 raw-frame features are
  the best visual input (54.5%); intra-domain reaches ~75.8%, so cross-domain drops 15-20
  points. Training on one player/show and testing on another is where models die.
- **SVC 2025 challenge** ([arXiv 2508.04129](https://arxiv.org/html/2508.04129), 21 teams):
  winner 62.44% accuracy (F1 43.89) cross-domain. The baseline fused ResNet18 face frames,
  OpenFace AUs/gaze, and Mel spectrograms.
- **The leakage cautionary tale** ([LieWaves, Feb 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12899830/)):
  subject-dependent overlapping-window evaluation gave 99.94% accuracy (AUC ~1.0); honest
  subject-independent session-level evaluation on the same data gave 66.7% acc and **AUC
  0.58**. Assign whole sessions to splits; evaluate at decision level, never at
  overlapping-window level.

**Bottom line**: a credible behavioral delta over a betting-only baseline is small; think
+0.02 to +0.05 AUC, concentrated in high-tell recreational players. Anything like "90% bluff
detection accuracy" signals leakage, not skill. The honest number *is* the portfolio piece.

## 3. Open-source building blocks (state as of mid-2026)

| Component | Pick | Notes |
|---|---|---|
| HUD OCR | **PaddleOCR PP-OCRv6** (released 2026-06-11) | Tiny/small/medium tiers (1.5M-34.5M params), Apache 2.0, claims 6.1x speedup on Apple M4. Use fixed-ROI crops plus template matching for card/suit glyphs, not full-frame OCR. |
| Face | **MediaPipe Face Landmarker** (52 blendshapes: blink, brow, smile asym) first; **OpenFace 3.0** (CMU, FG 2025: RetinaFace + STAR landmarks, AUs, yaw/pitch gaze) as the research-grade alternative | OpenFace 3.0 is image-oriented (needs its own frame loop) with modest adoption (~190 stars). |
| Pose/hands | **MediaPipe Pose** (33 kpts, permissive, fast on CPU) or **RTMPose-m** (75.8 AP at 90+ FPS on CPU, Apache 2.0); RTMW for 133-kpt whole-body | Avoid OpenPose (non-commercial license, slow). **Always smooth keypoints (One-Euro filter) before computing motion features.** |
| Equity/labels | **treys** (pure Python, ~3.2M hands/sec) or **eval7** (Cython, range-vs-range equity; wheels only up to Python 3.12; unweighted ranges only) | eval7 powers the betting-baseline features; last release Dec 2023 (stable, not evolving). |
| rPPG (stretch) | **rPPG-Toolbox** (NeurIPS 2023) | Validated only on lab-style recordings; no claims of surviving broadcast compression. Expect (and publish) a clean negative result. |
| Transcription (stretch) | Whisper + openSMILE prosody | Table-talk features; audio did well in deception benchmarks. |

**No open-source clone exists** (verified by direct repo inspection): everything on GitHub is
card/chip recognition or bot helpers (MemExplorer/poker-vision et al.). One repo named
`poker-lie-detector` was created 2026-07-10 in reaction to the WSOP publicity and is
**completely empty**. The behavioral-tells plus broadcast-OCR fusion is genuinely unbuilt in
open source.

## 4. Data

- **Hustler Casino Live** (YouTube) is the primary source: consistent HUD, exposed hole cards,
  consenting players, huge per-player volume for regulars. Triton/Lodge/PokerStars Live are
  alternates with different HUD formats; pick ONE format for v1.
- Scale check: Hand2Note commercially sells **1,006,639 hands from 37,589 players across 19
  live-stream channels** ($499), each hand linked to its source video timestamp. Two
  implications: (a) roughly 1M hole-card-labeled hands exist in public broadcast footage, so
  the data strategy is sound; (b) Hand2Note collects them with **human transcribers,
  dual-reviewed (99.9% claimed accuracy)**, so even a commercial vendor hasn't fully automated
  HUD reading. The OCR pipeline is a real contribution, and its accuracy must be validated
  against hand-transcribed ground truth.
- **Legal/ethical footing**: training on copyrighted broadcast footage implicates reproduction
  rights; fair-use factors favor a non-commercial research/portfolio project whose output
  (features and predictions) doesn't substitute for the footage (cf. intermediate-copying
  doctrine, *Sega v. Accolade*; note the 2025 ML-training fair-use rulings, *Thomson Reuters
  v. Ross*, *Bartz v. Anthropic*, and *Kadrey v. Meta*, make this fact-specific, not
  automatic). Practical policy: **don't redistribute footage or clips; publish code, derived
  features, and timestamps only; analyze only streams where players consented to hole-card
  broadcast; post-hoc analysis only, never real-time.** yt-dlp use technically violates
  YouTube ToS; keep downloads local and unpublished.

## 5. Evaluation design (the actual portfolio piece)

- **Question**: does nonverbal behavior add predictive power over betting action alone?
- **Baseline model**: betting features only: position, street, bet size / pot ratio, pot odds,
  action history, stack depth (SPR), equity vs. range where applicable.
- **Treatment model**: baseline plus behavioral features (per-player z-scored).
- **Feature window**: action-on-player to action-committed (the decision window).
- **Splits**: session-held-out AND player-held-out. Whole sessions assigned to one split.
  Evaluate per decision, never per overlapping frame window.
- **Leakage traps**: bet size correlates with hand strength (never let "behavioral" features
  encode it, e.g. via chip-stack-size visual features); autocorrelated windows; same-hand
  frames in train and test.
- **Report**: delta AUC, delta log-loss, calibration curves, per-player breakdown, confidence
  intervals (bootstrap over hands). A small positive delta on recreational players plus a null
  on pros is the *expected, literature-consistent* result.

## 6. Verification status

| Claim | Status |
|---|---|
| Geel architecture / MacBook / single-player Nik Airball training | Verified verbatim against PokerOrg (2026-04-16) |
| Post-elimination-only broadcast use | Corroborated by 3 independent outlets (Sportico 7/3, ZME 7/8, PokerScout 7/8) |
| Slepian statistics (face -.07, arms .07/.15, smoothness .29, upper-body .02) | Verified against the paper PDF |
| No OSS clone exists | Verified by direct inspection of candidate repos |
| Landmarks underperform raw frames (deception benchmarks) | Supported in-domain (DOLOS) and cross-domain (2405.06995) |
| DOLOS / cross-domain / SVC-2025 ceilings | Quoted accurately from arXiv abstracts/pages; not independently recomputed |
| PP-OCRv6 release and Apple-silicon speedups | Vendor-reported (PaddleOCR changelog) |
