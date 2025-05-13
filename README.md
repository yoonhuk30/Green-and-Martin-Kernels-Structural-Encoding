# README
This repo is the official implementation of ICML 2025 paper - [**_From Theory to Practice: Rethinking Green and Martin Kernels for Unleashing Graph Transformers_**]

> The implementation is based on [GRIT (Ma et al., ICML 2023)](https://github.com/LiamMa/GRIT).

### Python environment setup
```bash
# Please refer GRIT repo for environment setup
```

### Running GKSE and MKSE
```bash
python main.py --cfg configs/GKSE/ckt-bench101-GKSE.yaml  accelerator "cuda:0" seed 0
python main.py --cfg configs/MKSE/ckt-bench101-MKSE.yaml  accelerator "cuda:1" seed 0
```

### Configurations and Scripts

- Configurations are available under `./configs/GKSE/xxx.yaml`, `./configs/MKSE/xxx.yaml`
- Scripts to execute are available under `./scripts/xxx.sh`
  - will run 4 trials of experiments parallel on `GPU:0,1,2,3`.
