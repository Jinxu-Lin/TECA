# TECA — Geometric Incommensurability of Knowledge Operations

**Target**: NeurIPS 2026

## Research Question

Model editing (ROME/MEMIT) and training data attribution (TDA) both operate on model parameters — do their parameter update directions share geometric structure?

**Answer**: No. TECS (TDA-Editing Consistency Score) reveals **structured geometric incommensurability**: editing directions span ~40D (distributed), while attribution directions collapse to ~1D. This is not random misalignment — it reflects fundamentally different optimization geometries.

## Key Findings (Pilot: GPT-2-XL, 100 CounterFact facts)

| Metric | Value |
|--------|-------|
| TECS Cohen's d vs null | 0.050 (indistinguishable from chance) |
| Editing effective dim | 40.8 |
| Attribution effective dim | 1.2 |
| Whitening (C^{-1}) explains gap? | No (H6 rejected) |
| Structured incommensurability? | Yes (H7 confirmed) |

## Project Structure

```
TECA/
├── research/          # Problem statement, method design, experiment design
├── Codes/             # Experiment code and results
│   ├── experiments/   # Pilot and full-scale experiment scripts
│   ├── _Results/      # Probe results + raw JSON data
│   └── core/          # Shared utilities
├── Papers/            # NeurIPS 2026 LaTeX draft
│   ├── sec/           # Section .tex files
│   └── sty/           # neurips_2026.sty
├── Reviews/           # Formalize and design review records
├── Docs/              # Module status files
├── project.md         # Project overview
└── CLAUDE.md          # AI agent instructions
```

## Status

- **Research Module**: `implement` (blueprint complete, full-scale experiments pending)
- **Paper Module**: P7 complete (LaTeX draft with PENDING placeholders)
- Managed by Noesis v3
