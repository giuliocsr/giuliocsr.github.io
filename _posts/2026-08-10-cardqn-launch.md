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
No — 3.4× vs 9.5×. All agents are trained on a simulator and hit a sim-to-real ceiling.

### What does "distributionally robust" mean?
Don't trust one model — consider all nearby models and plan for the worst.

### What is the context tag?
A past-only label with three components (trend, volatility, position), giving 27 regimes total. No look-ahead.

### What is the fidelity score?
An out-of-sample measure of how reliably the model predicts each regime.

### What is the proposed Bellman-target blend?
A rule that de-hedges in favorable regimes. Proposed in the paper, not yet validated.

### How is the system trained?
On a signature-MMD market simulator, then evaluated out-of-sample on the real S&P 500 (1995–2024). 5 random seeds per configuration.

### How long does training take?
~10–14 minutes per episode on an RTX 2080 Ti. A full 10-episode run takes about 2 hours.

### What markets does CARDQN trade?
The S&P 500 (1995–2024), single-asset, daily rebalancing with 0.05% transaction costs.

### Can CARDQN be applied to other domains?
Yes — the context-aware ambiguity idea is domain-agnostic and applies to robotics, control, and operations.

### Where is the full math?
In the [paper](https://giuliocsr.github.io/papers/cardqn.pdf) and the [explainer](https://giuliocsr.github.io/blog/cardqn-explained).

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
