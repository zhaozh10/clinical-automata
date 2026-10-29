# Femoral head segmentation (in-house paediatric hip radiographs)

Project `hip_femoral_head_seg`. The code is the agent's version at commit `215c564`.

The test result reported in the paper (Dice 0.928 ± 0.003 on all cases, 0.906 ± 0.004 on difficult cases, NSD 0.854 ± 0.012) is the mean ± standard deviation over three models trained with different seeds: run5 (seed 0), run_final_s1 (seed 1) and run_final_s2 (seed 2). No single checkpoint corresponds to it.

## Files

- [`code/train.py`](code/train.py)
- [`code/data.py`](code/data.py)
- [`results.tsv`](results.tsv)
- [`USAGE_MANUAL.md`](USAGE_MANUAL.md)

## Best checkpoint

Not released: this model was trained on in-house patient data.
