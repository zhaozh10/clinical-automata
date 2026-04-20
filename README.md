# [arXiv'26] From Clinical Intent to Clinical Model: An Autonomous Coding-Agent Framework for Clinician-driven AI Development 🤖
---
Clinical Automata lets a clinician describe a problem in plain language — *"help me diagnose pneumothorax, but don't let the model cheat by looking at chest drains"* — and returns a trained, task-specific deep-learning model. No data-science middleman required. 

<p align="left">by Zihao Zhao, Frederik Hauke, Juliana De Castilhos, Jakob Nikolas Kather, Sven Nebelung, and Daniel Truhn</p>



<!-- 📄 **Paper:** *From Clinical Intent to Clinical Model: An Autonomous Coding-Agent Framework for Clinician-driven AI Development* (Zhao et al., 2026) -->

<div align="center">
  <img src="asset/teaser.png" style="width: 90%">
</div>

> <p align="justify">
> <strong>Comparison between conventional multi-party workflow and our proposed clinician-driven workflow.</strong> In the conventional paradigm, clinicians rely on discussions with AI experts to translate clinical needs into technical implementation, which may incur coordination costs and introduce misalignment because each side lacks deep knowledge of the other's domain. Our proposed framework replaces this intermediate human bottleneck with an autonomous coding agent. Although not a specialist in any single domain, the agent has sufficiently broad knowledge to bridge medicine and AI, while its strong autonomous coding capability makes direct clinician-driven AI development possible.
</p>

---

## Autonomous development
 
The agent proposes a pipeline, and then improve it iteratively through training/validation.Below is a real refinement trajectory on the mixed-supervision wrist-fracture task, where only 5% of training images had bounding boxes. Starting from a 0.582 mAP@50 baseline, the agent climbed to **0.87** by autonomously discovering three key moves: distributed training with a larger batch size (run 2), curated negatives (run 6), and a teacher–student pseudo-labeling strategy that exploited the 95% image-level-only pool (run 8).
 
<!-- ![Autonomous refinement trajectory on the mixed-supervision wrist-fracture task.](asset/evolution.png) -->
<div align="center">
  <img src="asset/evolution.png" style="width: 95%">
</div>

Notably, most of the gain comes from a handful of successful edits out of 30 attempts. The rest edits marked by gray circle are discarded by the acceptance rule.

---

## How it works

A clinician request flows through three stages:

1. **Semantic Parser** — converts the natural-language request into a structured representation capturing the *clinical objective*, *risk preference*, and *output format*.
2. **Task Initializer** — translates that representation into an executable codebase: model architecture, training recipe, evaluation protocol.
3. **Autonomous Developer** — iteratively edits the codebase, runs experiments, inspects failures, and keeps whatever improves a prespecified validation objective. The clinician can inspect choices and negotiate trade-offs along the way.

Under the hood, the three roles are played by a coding agent (Claude Opus 4.6 in our experiments). Each iteration runs a train/validation cycle inside a fixed time budget. The test set is held out from the start and touched exactly once, at the end.

---

## Project structure

```
  clinical-automata/
  ├── README.md           # This file                                                                       
  ├── asset/              # Figures used in README      
  └── src/                                                                                                  
      ├── Parser.md       # Semantic parser prompt: clinician request → structured task spec                
      ├── program.md      # Task Initializer & Autonomous Developer prompts                                 
      ├── README.md       # In-repo context provided to the coding agent                                    
      ├── prepare.py      # Constants and setup helpers                                                     
      ├── gpu.sh          # GPU request script (SLURM)                                                      
      ├── pyproject.toml  # Project metadata and Python dependencies                                        
      └── uv.lock         # Locked dependency versions (uv)   
```

---

## Quick start


```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download and activate Claude CLI
curl -fsSL https://claude.ai/install.sh | bash

claude --permission-mode bypassPermissions

# 4. Prompt something like
[$your_request$] Please review program.md and start a new experiment. Let us begin with the setup first.
```


---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1, CUDA-capable GPU
- Access to a capable coding agent (Claude Opus 4.6 CLI used in the paper)
- During experiments, the coding agent installs dataset- and task-specific packages on demand, so `src/uv.lock` reflects the accumulated state of past runs rather than a clean baseline. For a minimal working environment, refer to the lockfile in [karpathy/autoresearch](https://github.com/karpathy/autoresearch/blob/master/uv.lock).
---

## Citation

If you use this work, please cite:

```bibtex
@article{zhao2026clinicalautomata,
  title   = {From Clinical Intent to Clinical Model: An Autonomous Coding-Agent
             Framework for Clinician-driven AI Development},
  author  = {Zhao, Zihao and Hauke, Frederik and De Castilhos, Juliana
             and Kather, Jakob Nikolas and Nebelung, Sven and Truhn, Daniel},
  year    = {2026}
}
```

---

## Data

All datasets used are publicly available:

- [ISIC 2019](https://challenge.isic-archive.com/data/) — dermoscopic lesion classification
- [GRAZPEDWRI-DX](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193) — pediatric wrist radiographs
- [SIIM-ACR Pneumothorax](https://www.kaggle.com/datasets/anisayari/siimacrpneumothoraxsegmentationzip-dataset) — chest radiographs
- [NEATX](https://zenodo.org/records/14944064) — chest-drain annotations over a subset of NIH ChestX-ray14

---

## Acknowledgements

We thanks Andrej Karpathy for open-sourcing [autoresearch](https://github.com/karpathy/autoresearch), which inspires this study.

