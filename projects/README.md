# Projects

The projects shown in the paper, as the agent left them. Each folder contains:

- `code/`: the programs the agent wrote (training, data pipeline, and inference scripts where they exist)
- `results.tsv`: the log of every run, kept or discarded
- `USAGE_MANUAL`: written during finalisation. Projects that predate the finalisation stage do not have one.
- a link to the best checkpoint

Absolute paths in the code were replaced with the placeholders `/path/to/your/datasets` and `/path/to/your/external_data` (see `src/config.env.example`).

## Straightforward requests

- [Skin lesion diagnosis, 8 classes (ISIC 2019)](dermoscopy_multiclass/): checkpoint [best.pth](https://clinical-automata-ckpt.ciaochao.com/dermoscopy_multiclass/best.pth)
- [Melanoma vs. nevus (ISIC 2019)](dermoscopy_melanoma_nv/): checkpoint [best.pth](https://clinical-automata-ckpt.ciaochao.com/dermoscopy_melanoma_nv/best.pth)
- [Wrist fracture detection (GRAZPEDWRI-DX)](wrist_fracture_supervised_det/): checkpoint [best.pth](https://clinical-automata-ckpt.ciaochao.com/wrist_fracture_supervised_det/best.pth)
- [Wrist fracture detection with mixed supervision, 5% boxes (GRAZPEDWRI-DX)](wrist_fracture_semi_det/): checkpoint [best.pt](https://clinical-automata-ckpt.ciaochao.com/wrist_fracture_semi_det/best.pt)
- [Heart chamber segmentation (PAX-ray++)](heart_chamber_seg/): checkpoint [best.pth](https://clinical-automata-ckpt.ciaochao.com/heart_chamber_seg/best.pth)
- [Pelvic fracture fragment segmentation (PENGWIN 2026)](pelvic_ct_fragment_seg/): checkpoint [best.pth](https://clinical-automata-ckpt.ciaochao.com/pelvic_ct_fragment_seg/best.pth)

## Complex requests

- [Interpretable wrist fracture detection (GRAZPEDWRI-DX)](wrist_fracture_semi_det_explainable/): checkpoint [best.pth](https://clinical-automata-ckpt.ciaochao.com/wrist_fracture_semi_det_explainable/best.pth)
- [Apical pneumothorax second reader (SIIM-ACR + PAX-ray++)](apical_pneumothorax_reader/): checkpoint [best.pth](https://clinical-automata-ckpt.ciaochao.com/apical_pneumothorax_reader/best.pth)
- [Femoral head segmentation (in-house data, checkpoint not released)](hip_femoral_head_seg/): checkpoint not released
- [Lung lobe / zone segmentation for teaching (PAX-ray++)](lobe_zone_seg/): checkpoint [best.pth](https://clinical-automata-ckpt.ciaochao.com/lobe_zone_seg/best.pth)
- [Cardiac structure segmentation for teaching (PAX-ray++), 3-model ensemble](cardiac_structures_seg/): checkpoint [run13_best.pth](https://clinical-automata-ckpt.ciaochao.com/cardiac_structures_seg/run13_best.pth), [run14_best.pth](https://clinical-automata-ckpt.ciaochao.com/cardiac_structures_seg/run14_best.pth), [run15_best.pth](https://clinical-automata-ckpt.ciaochao.com/cardiac_structures_seg/run15_best.pth)

## Clinician-guided debiasing (pneumothorax vs. chest drains)

- [Debiased pneumothorax classifier (NIH ChestX-ray14)](pneumothorax_debiased/): checkpoint [best.pth](https://clinical-automata-ckpt.ciaochao.com/pneumothorax_debiased/best.pth)
- [Auxiliary chest-drain predictor (NEATX)](chest_drain_cls/): checkpoint [best.pth](https://clinical-automata-ckpt.ciaochao.com/chest_drain_cls/best.pth)
