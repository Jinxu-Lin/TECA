# TECA — TDA-Editing Consistency Analysis

> Claude: read this file first, then consult specific documents as needed.

## Project Overview

- **Topic**: Geometric relationship between model editing directions and TDA attribution gradients in transformer MLP parameter space
- **Problem**: No prior work connects editing and attribution at parameter-space geometry level. The Hase et al. (2023) localization-editing disconnect lacks a parameter-level explanation.
- **Approach**: TECS metric (cosine similarity) + six-component geometric characterization framework. Dual-outcome design: positive TECS reveals geometric consistency; null TECS (CONFIRMED) reveals structured incommensurability.
- **Core Finding**: TECS ~ 0 (Cohen's d = 0.050). Editing eff-dim = 40.8, attribution eff-dim = 1.2. Asymmetric cross-projection. MEMIT partially bridges gap (d ~ 0.63).
- **Target**: NeurIPS 2026

## Resource Constraints

- **GPU**: 1x NVIDIA RTX 4090 (24GB VRAM), remote server xuchang3
- **Remote Server**: xuchang3 (SSH MCP), path: /home/jinxulin/sibyl_system
- **Timeline**: NeurIPS 2026 submission

> All analysis, design, and experiment planning must fit within these constraints.

## Current Status

- **Module**: research
- **Phase**: implement (pilot experiments complete, full-scale pending)
- **Next step**: Run full-scale experiments (200 facts + ablation + positive control)

> Authoritative state in `Docs/*-module-status.json`. This section is a quick reference only.

## Key Documents

| Document | Description | Stage |
|----------|-------------|-------|
| `project.md` | Project overview (idea, problem, method, hypotheses, probe design, review) | Init |
| `Codes/_Results/probe_result.md` | Pilot experiment results (COMPLETE) | Init / probe_impl |
| `research/problem-statement.md` | Formal Gap + RQ + attack angle | Research / formalize |
| `research/method-design.md` | TECS + six-component framework | Research / design |
| `research/experiment-design.md` | 5 phases + negative/positive paths + ablation | Research / design |
| `Codes/_Results/experiment_result.md` | Full experiment results (PENDING) | Research / implement |
| `research/contribution.md` | 5 contributions tracked | Cross-phase |
| `iteration-log.md` | Project iteration history | Cross-phase |
| `Papers/` | NeurIPS LaTeX template with section drafts | Paper module |

## Legacy Data

| Path | Description |
|------|-------------|
| `iter_001/exp/` | All Sibyl-era experiment code and results |
| `iter_001/exp/results/` | JSON results for all phases |
| `iter_001/exp/results/rome_deltas/` | 100 saved delta tensor files (.pt) |
| `iter_001/idea/` | Sibyl-era proposal, hypotheses, perspectives |
| `legacy/teca-noesis/` | TECA_old (Noesis V1) materials: debate, startup, contribution |

## Noesis System

- **Path**: `~/Research/Noesis`
- **State files**: `Docs/init-module-status.json`, `Docs/research-module-status.json`, `Papers/paper-status.json`
- **CLI reference**: `~/Research/Noesis/Praxis/CLAUDE.md`

## Code Constraints

- Experiment code in `iter_001/exp/` (Sibyl legacy structure)
- `Codes/_Data/` for generated data (gitignore) | `Codes/_Results/` for experiment results (md, git tracked)
- Remote execution via SSH MCP to xuchang3
- **Commit + push after each modification**
