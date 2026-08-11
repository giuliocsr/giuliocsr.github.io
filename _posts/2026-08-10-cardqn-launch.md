---
layout: default
title: "CARDQN"
description: "A context-aware robust trading agent that more than doubles the baseline wealth at better risk-adjusted return."
date: 2026-08-10 10:00:00 +0000
---
<p align="center"><img src="/assets/blog/cardqn-logo.png" alt="CARDQN"></p>

*Context-Aware Distributionally Robust Deep Q-Learning — a robust trading agent that hedges hard when uncertain and commits when confident.*

Robust trading agents hedge against the worst plausible market move — safe, but on
the S&P 500 they finish at ~2–3× while buy-and-hold earns **9.5×**. **CARDQN** makes
that hedge *context-aware*: it labels the market regime, scores how reliably its
model predicts it, and tightens the safety margin only where that score is high.
The result: **3.40× terminal wealth** — more than double the robust baseline — at
roughly **2× the Sharpe and Sortino**.

<figure><img src="/assets/blog/terminal-wealth.png" alt="Terminal wealth: CARDQN 3.40× vs RDQN 1.59× vs S&P 9.52×">
<figcaption>End of training, 5 seeds: CARDQN <b>3.40×</b> vs RDQN <b>1.59×</b> vs S&P 500 buy & hold <b>9.52×</b>.</figcaption></figure>

<figure><img src="/assets/blog/risk-metrics.png" alt="Risk metrics: Sharpe, Sortino, volatility, max drawdown">
<figcaption>CARDQN roughly doubles RDQN's Sharpe and Sortino, at comparable volatility and drawdown.</figcaption></figure>

# How it works

Three pieces, all computed from past data only: a **regime tag** τ labels the market
(trend, volatility, position — 27 regimes); a **fidelity score** φ measures
out-of-sample predictability per regime; an **adaptive radius** ε̃ and **reference**
P̃ tighten the ambiguity ball where φ is high, relax it where it's low.

<figure><img src="/assets/blog/pipeline.png" alt="CARDQN pipeline">
<figcaption>State → regime tag → fidelity → adaptive radius & reference → robust Bellman update.</figcaption></figure>

## Links

📄 **[Paper (PDF)](https://giuliocsr.github.io/papers/cardqn.pdf)** · 💻 **[Code (GitHub)](https://github.com/giuliocsr/CARDQN)** · ✍️ **[Explainer](https://giuliocsr.github.io/blog/cardqn-explained)**

# FAQ

### Does it beat buy & hold?
No — CARDQN reaches 3.4× vs buy-and-hold's 9.5×. Every agent here trains on a simulator and then evaluates on the real market; the synthetic data under-represents the S&P's long-term drift, creating a sim-to-real ceiling that caps all agents. CARDQN's claim is over the robust baseline (RDQN), not over the market — it delivers more wealth and better risk-adjusted return than the original robust method, while staying robust.

### What does "distributionally robust" mean?
Instead of trusting one fitted model of the market, you consider every model close to yours (inside an "ambiguity ball" of radius ε) and plan for the worst one. You trade as if the nastiest plausible market were the true one — so you're never caught off-guard by model error. The radius ε is your distrust budget: bigger means more robust but more pessimistic.

### What is the context tag?
A past-only label of the current market: trend (up/down/flat), volatility (low/mid/high), and position (near a recent high/low/interior) — 27 regimes in total. It's computed entirely from the recent return window, with no look-ahead at future prices. The volatility thresholds are frozen once on the full S&P history so a day's label doesn't drift as training proceeds.

### What is the fidelity score?
An out-of-sample measure of how reliably the model predicts each regime — high only when the regime is both well-sampled and genuinely predictable. It's scored on held-out transitions (not the data the model was fit on), so it measures true predictive accuracy, not in-sample overfit. This prevents the system from over-trusting a model that has merely memorized noise.

### What is the proposed Bellman-target blend?
A complementary rule that de-hedges toward the risk-neutral objective in clearly favorable, well-identified regimes — by a confidence-gated, capped weight that provably never becomes optimistic. It's sketched in the paper but not yet validated empirically; the results on this page are CARDQN without it.

### How is the system trained?
On a signature-MMD market simulator — synthetic return paths fitted to S&P 500 data — then evaluated out-of-sample on the real S&P 500 (1995–2024, proportional transaction costs). Each configuration runs across 5 random seeds to average out the per-seed variance inherent in deep RL training.

### How long does training take?
~10–14 minutes per episode on an RTX 2080 Ti GPU; a full 10-episode run takes about 2 hours. A 5-seed campaign across two configurations (off vs full) runs overnight on a university HPC cluster under SLURM.

### What markets does CARDQN trade?
Currently the S&P 500 (1995–2024) as a single-asset portfolio: the agent chooses a weight from all-cash to all-in (and everything in between), rebalanced daily with 0.05% proportional transaction costs. The framework generalizes to any liquid asset with sufficient price history.

### Can CARDQN be applied to other domains?
Yes — the context-aware ambiguity idea isn't finance-specific. Anywhere you do robust sequential decision-making under a model you only partially trust (robotics, control, supply-chain optimization), the principle of "hedge hard when uncertain, commit when confident" applies directly. The regime tag, fidelity score, and adaptive radius are domain-agnostic machinery.

### Where is the full math?
In the [paper](https://giuliocsr.github.io/papers/cardqn.pdf) and the [explainer post](https://giuliocsr.github.io/blog/cardqn-explained).

### What license?
CC BY-NC-SA 4.0.

## Cite

```bibtex
@misc{golinelli2026cardqn,
  author = {Giulio Golinelli},
  title  = {Context-Aware Distributionally Robust Deep {Q}-Learning ({CARDQN})},
  year   = {2026},
  url    = {https://giuliocsr.github.io/papers/cardqn.pdf},
  note   = {Code: https://github.com/giuliocsr/CARDQN}
}
```
