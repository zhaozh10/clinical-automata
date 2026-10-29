# Apical pneumothorax second reader (SIIM-ACR + PAX-ray++)

Project `apical_pneumothorax_reader`. The best model is from **run29**; the code is the agent's version at commit `b05350b`.

## Files

- [`code/train.py`](code/train.py)
- [`code/data.py`](code/data.py)
- [`code/predict.py`](code/predict.py)
- [`code/setup/analyze.py`](code/setup/analyze.py)
- [`code/setup/build_nih_pool.py`](code/setup/build_nih_pool.py)
- [`code/setup/build_sidecrop_cache.py`](code/setup/build_sidecrop_cache.py)
- [`code/setup/build_siim_cache.py`](code/setup/build_siim_cache.py)
- [`code/setup/finalize_eval.py`](code/setup/finalize_eval.py)
- [`code/setup/hash_overlap.py`](code/setup/hash_overlap.py)
- [`code/setup/hash_overlap2.py`](code/setup/hash_overlap2.py)
- [`code/setup/infer_zones.py`](code/setup/infer_zones.py)
- [`code/setup/make_512_cache.py`](code/setup/make_512_cache.py)
- [`code/setup/paxray_zones.py`](code/setup/paxray_zones.py)
- [`code/setup/train_anatomy.py`](code/setup/train_anatomy.py)
- [`results.tsv`](results.tsv)
- [`USAGE_MANUAL.txt`](USAGE_MANUAL.txt)

## Best checkpoint

- [best.pth](https://clinical-automata-ckpt.ciaochao.com/apical_pneumothorax_reader/best.pth)
