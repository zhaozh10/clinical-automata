# automata

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously. It modifies the code, trains for TIME_BUDGET_MINUTES minutes, checks if the result improved, keeps or discards, and repeats. The core idea is that you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069) and [this tweet](https://x.com/karpathy/status/2031135152349524125).

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, data paths, experiment config (not modified). Not modified.
- **`data.py`** - data preprocessing, augmentation, splitting (written by agent). **This file is edited and iterated on by the agent**.
- **`train.py`** — model, optimizer, training loop, evaluation (written by agent). Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.
- **`Parser.md`** — Protocols on how to interpret the doctor's request. **This file can't be edited**.

By design, a single training run for a **fixed TIME_BUDGET_MINUTES time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. In each run, trains a model, evaluates on validation set, logs metrics, and saves checkpoints. This will loop MAX_LOOPS times

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** At least a single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and readme.md. I want to have an AI model to help me [...]. Let's kick off a new experiment project! let's do the setup first.    
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
data.py         - data preprocessing, augmentation, splitting (written by agent and can be modified)
train.py        — model, optimizer, training loop (written by agent and can be modified)
program.md      — agent instructions
Parser.md      — instructions on how to interpret human input
pyproject.toml  — dependencies
```

## Design choices

- **File to modify.** During experiment loop, the agent only touches `train.py` and `data.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly TIME_BUDGET_MINUTES minutes, regardless of your specific platform. This means you can expect many optimization attempts while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Useful experience
some practical tricks that are often much more useful than blindly scaling model size.

1. Do not downsample blindly.  In medical imaging, resolution is often part of the signal. For example, in digital mammography, aggressive downsampling can destroy microcalcifications or subtle lesion boundaries. Use modality-aware defaults:
   - mammography: treat `1024x1024` as a minimum unless there is a strong reason otherwise
   - chest X-ray: `512x512` may be acceptable for many tasks
   - pathology / microscopy: often better handled with patching or tiling instead of shrinking the whole image too much

2. Use medically appropriate augmentations. Strong natural-image augmentations can be harmful in medicine. Prefer restrained augmentations such as small rotations, flips only when anatomically valid, mild brightness / contrast jitter, light blur or noise only if it reflects acquisition variation. Avoid overly aggressive augmentation that changes pathology appearance or creates anatomically implausible samples.

4. Handle class imbalance explicitly. Many medical datasets are highly imbalanced. Useful options include weighted loss, focal loss, class-balanced sampling, oversampling rare classes.  
For screening-style tasks, improving minority-class recall is often more important than raw accuracy.

5. In medical tasks, the best checkpoint is often not the one with the lowest validation loss. Depending on the medical task, prioritize metrics such as AUC, sensitivity / recall, specificity, F1, Youden-Index, Dice / IoU, etc. The training loss can stay standard, but model selection should follow the clinically relevant primary metric.

6. Tune the decision threshold, not just the model weights. For many classification tasks, especially imbalanced ones, a large gain can come from selecting a better operating threshold on the validation set instead of using the default `0.5`. This is particularly important when the user wants higher sensitivity, fewer false negatives, or a specific sensitivity/specificity tradeoff.

7. Use pretrained backbones whenever possible.  
   Medical datasets are often small. Starting from pretrained models is usually a stronger baseline than training from scratch. The agent should consider:
   - `timm` backbones
   - Hugging Face vision backbones
   - domain-specific checkpoints if available and compatible  
   In many cases, a good pretrained encoder plus a simple head beats a fancy custom architecture.

8. Freeze more, then unfreeze if needed.  
   On small datasets, full fine-tuning can overfit or become unstable. A good default strategy is: (1) start with a frozen backbone and train the head, (2) then unfreeze later layers if validation saturates.
   This is often more compute-efficient and more stable.

9. Use patient-level splitting whenever possible.  
   Medical datasets often contain multiple studies, views, slices, or tiles from the same patient. Splitting at the image level can leak information and produce overly optimistic results. If patient IDs exist, use patient-level splitting by default. Multi-view and multi-modal input are encouraged.

10. Keep validation representative but cheap.  
    Validation should remain stable across runs, but on small hardware it may be too expensive to run very frequently. A practical compromise is:
    - validate on the same fixed validation set
    - reduce validation frequency if needed
    - keep the evaluation function unchanged within the experiment loop

11. For segmentation and detection, start simple.  
    Before trying a large architecture, first make sure (1) masks / boxes are loaded correctly (2) interpolation is correct (3) target alignment is correct (4) postprocessing is reasonable  
    Many failures in medical segmentation/detection come from pipeline bugs rather than model weakness.

12. Watch calibration and confidence, not just discrimination.
    In clinical settings, a model with decent AUC but badly calibrated confidence can still be difficult to use. If relevant, consider logging:
    - calibration error
    - confidence histograms
    - sensitivity/specificity at chosen thresholds

14. Prefer stable, reproducible wins over fragile complexity.  
    Medical datasets are often small and noisy. Tricks that give tiny gains but add a lot of engineering complexity are often not worth it. Simpler pipelines with robust validation are usually better foundations for autonomous iteration.

15. **Important**.
    When tuning `train.py` and `data.py`, improvement proposals should not remain concentrated on incremental knob-tuning alone once a stable version of method has been established. The agent should increasingly explore changes that are plausibly capable of producing qualitatively different behavior, rather than only small quantitative shifts from routine hyperparameter or recipe variation. Each proposal should be motivated by an explicit hypothesis about what is limiting current performance and how the proposed change could alter that limitation.

In short, for medical tasks the biggest wins often come from:
- preserving clinically relevant image detail
- handling imbalance properly
- choosing the right validation metric
- tuning thresholds
- preventing leakage
- using pretrained backbones and task-appropriate data pipelines

I think these would be the reasonable tuning tricks to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## License

MIT
