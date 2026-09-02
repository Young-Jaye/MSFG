# Manuscript configurations

These YAML files record the parameter sets selected for the manuscript experiments.

## DLPFC

The twelve `dlpfc_<sample>.yaml` files record the selected configuration for each DLPFC sample.

## Other datasets

- `hbrc.yaml`: best supplied Human Breast Cancer configuration, seed `166081447`.
- `mba.yaml`: best supplied Mouse Brain Anterior configuration, seed `236017509`; recorded ARI `0.508271`.
- `mouse_e1s1.yaml`: best supplied mouse E1S1 configuration, seed `637602023`; the runner derives 12 clusters from `obs['annotation']`.

## Usage

Run from the repository root:

```bash
python train.py --config configs/paper/dlpfc_151507.yaml
python train.py --config configs/paper/hbrc.yaml
python train.py --config configs/paper/mba.yaml
python train.py --config configs/paper/mouse_e1s1.yaml
```

The `provenance` section is documentation only. The configuration loader passes only `dataset`, `preprocessing`, `graph`, `model`, and `training` values to the established runner.
