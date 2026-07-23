# Paper

Nonverbal Behavior Does Not Add Out-of-Session Predictive Power for Hand
Strength in Broadcast Poker: A Preregistered Replication Failure and a
Case Study in Identity-Contamination Artifacts.

## Build

    pdflatex main && bibtex main && pdflatex main && pdflatex main

Figures regenerate with `uv run python figures/make_figures.py` from the
repo root; every number in the figure script cites its source commit.

## arXiv submission

Run `./make_arxiv.sh` and upload `dist/arxiv.tar.gz`. Suggested
categories: cs.CV primary (video understanding), cross-list cs.HC and
stat.AP. Submission needs an arXiv account with endorsement for cs.CV;
a first-time submitter with a .edu address usually clears this quickly.
Before submitting: flip the GitHub repository public (the paper links
it), and consider an OSF or Zenodo deposit of the preregistration file
for an independent timestamp (the git history already provides one).

## Venue notes

The paper is a preregistered negative result with a methods
contribution. Friendly venues: ICMI, ACII, or FG (main or workshop
tracks value behavioral-signal rigor), or a CHI/CSCW workshop on
behavioral sensing. arXiv first is recommended regardless for the
timestamp.
