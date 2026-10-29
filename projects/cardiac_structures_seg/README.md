# Cardiac structure segmentation for teaching (PAX-ray++)

Project `cardiac_structures_seg`. The final model is an **ensemble of three models (run13 + run14 + run15)**. It reaches a mean chamber Dice of 0.918 on the test set (`results.tsv`, row "ensemble3"). All three checkpoints are needed.

## Ensemble members

| Member | Encoder | Seed | Code (commit) | Checkpoint |
|---|---|---|---|---|
| run13 | DINOv3 ConvNeXt-Small | 1234 | [`code/run13/`](code/run13/) (`9b1e4a0`) | [run13_best.pth](https://clinical-automata-ckpt.ciaochao.com/cardiac_structures_seg/run13_best.pth) |
| run14 | DINOv3 ConvNeXt-Base | 1234 | [`code/run14/`](code/run14/) (`b0d7518`) | [run14_best.pth](https://clinical-automata-ckpt.ciaochao.com/cardiac_structures_seg/run14_best.pth) |
| run15 | DINOv3 ConvNeXt-Small | 2024 | [`code/run15/`](code/run15/) (`82bc2ef`) | [run15_best.pth](https://clinical-automata-ckpt.ciaochao.com/cardiac_structures_seg/run15_best.pth) |

The three members share the same `data.py`. Their `train.py` files differ only in the `encoder` and `seed` settings.

## Files

- [`code/run13/`](code/run13/), [`code/run14/`](code/run14/), [`code/run15/`](code/run15/): `train.py` and `data.py` for each member
- [`code/predict_test.py`](code/predict_test.py), [`code/visualize.py`](code/visualize.py): inference and teaching figures (from `82bc2ef`). Pass all three checkpoints comma-separated to `--checkpoint` to run the ensemble.
- [`results.tsv`](results.tsv)
- [`USAGE_MANUAL.md`](USAGE_MANUAL.md)
- [`USAGE_MANUAL.txt`](USAGE_MANUAL.txt)
