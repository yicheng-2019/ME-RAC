# ME-RAC

> **Multipath 3D-Conv encoder and temporal-sequence decision for repetitive-action counting**
>
> Yicheng Qiu, Li Niu, Feng Sha
>
> *Expert Systems with Applications*, 2024 &nbsp;|&nbsp; [Paper](https://doi.org/10.1016/j.eswa.2024.123760)

## Introduction

ME-RAC is built upon [TransRAC](https://github.com/SvipRepetitionCounting/TransRAC) and introduces three contributions for repetitive-action counting:

- **ME module** — A multipath 3D-Conv encoder with parallel 1x1x1, 3x3x3, and 5x5x5 branches, replacing the single-path 3D convolution to capture richer temporal-spatial features.
- **TSRC** — Temporal-Sequence Random Combination data augmentation that recombines action segments to prevent overfitting.
- **TSD** — A two-stage Temporal-Sequence Decision framework combining YOLOv5 detection with ME-RAC counting for long video scenarios.

## Getting Started

### Requirements

- Python >= 3.7
- PyTorch >= 1.7.0
- CUDA >= 11.0
- [mmaction2](https://github.com/open-mmlab/mmaction2) & [mmcv-full](https://github.com/open-mmlab/mmcv)

```bash
git clone https://github.com/yicheng-2019/ME-RAC.git
cd ME-RAC
pip install -r requirements.txt
```

### Backbone

Download the Video Swin Transformer (Swin-Tiny) pretrained on Kinetics-400 from [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) and place it under `./pretrained/`.

### Data Preparation

We use [RepCount-A](https://svip-lab.github.io/dataset/RepCount.html) and [UCFRep](https://www.crcv.ucf.edu/data/UCF101.php) datasets. Convert videos to NPZ for faster loading:

```bash
python tools/video2npz.py --input_dir ./data/raw_videos --output_dir ./data/npz --num_frames 64
```

## Usage

### Training

```bash
# ME-RAC
python train.py --model me_rac \
    --data_dir ./data/RepCountA/train \
    --label_file ./data/RepCountA/train.csv \
    --val_dir ./data/RepCountA/valid \
    --val_label ./data/RepCountA/valid.csv \
    --epochs 200 --batch_size 16 --lr 8e-7

# ME-RAC + TSRC
python train.py --model me_rac \
    --data_dir ./data/RepCountA/train \
    --label_file ./data/RepCountA/train.csv \
    --val_dir ./data/RepCountA/valid \
    --val_label ./data/RepCountA/valid.csv \
    --use_tsrc --tsrc_prob 0.3

# TransRAC baseline
python train.py --model transrac \
    --data_dir ./data/RepCountA/train \
    --label_file ./data/RepCountA/train.csv \
    --val_dir ./data/RepCountA/valid \
    --val_label ./data/RepCountA/valid.csv
```

### Evaluation

```bash
python test.py --model me_rac \
    --data_dir ./data/RepCountA/test \
    --label_file ./data/RepCountA/test.csv \
    --checkpoint ./checkpoints/best.pt
```

## Project Structure

```
ME-RAC/
├── configs/                    # Video Swin Transformer configs
├── dataset/                    # Data loaders & label generation
├── models/                     # Model components
│   ├── ME_Module.py            #   Multipath 3D-Conv encoder
│   ├── TransRAC.py             #   TransRAC baseline
│   ├── base_modules.py         #   Attention, positional encoding, predictor
│   └── encoder_modules.py      #   Similarity matrix, transformer encoder
├── training/                   # Training loop
├── testing/                    # Evaluation loop
├── tools/                      # Video preprocessing utilities
├── ME_RAC.py                   # ME-RAC model
├── TSRC.py                     # TSRC data augmentation
├── TSD.py                      # TSD framework
├── train.py                    # Training entry point
└── test.py                     # Testing entry point
```

## Citation

```bibtex
@article{QIU2024123760,
    title     = {Multipath 3D-Conv encoder and temporal-sequence decision for repetitive-action counting},
    journal   = {Expert Systems with Applications},
    volume    = {249},
    pages     = {123760},
    year      = {2024},
    issn      = {0957-4174},
    doi       = {https://doi.org/10.1016/j.eswa.2024.123760},
    author    = {Yicheng Qiu and Li Niu and Feng Sha},
}
```

## Acknowledgements

- [TransRAC](https://github.com/SvipRepetitionCounting/TransRAC)
- [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)
- [mmaction2](https://github.com/open-mmlab/mmaction2)
- [YOLOv5](https://github.com/ultralytics/yolov5)

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
