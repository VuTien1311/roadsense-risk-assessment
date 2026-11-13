# Pretrained weights

This repository runs inference only: nothing is trained or fine-tuned, and the three
checkpoints are the **official pretrained files of their authors**. They are not stored in
this repository (size and licenses); a script downloads them from the original releases and
verifies them.

## One command

```powershell
& .\.venv\Scripts\python.exe scripts\download_weights.py
& .\.venv\Scripts\python.exe run_paper.py --check
```

`download_weights.py` reads `weights_manifest.json`, downloads each missing file from its
official URL, and keeps it only if the size and the SHA-256 match the values used for the
paper. A file that is already present and correct is not downloaded again; a file that is
present but wrong is reported and never overwritten. `--list` shows the sources and licenses
without downloading anything.

| File, relative to the repository | Model | Size | Official source | License |
|---|---|---:|---|---|
| `yolo11x-seg.pt` | YOLO11x-seg, COCO-pretrained | 125 MB | [Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0) | AGPL-3.0 |
| `hybrid2/weights/hybridnets.pth` | HybridNets-D3, BDD100K-pretrained | 55 MB | [HybridNets v1.0](https://github.com/datvuthanh/HybridNets/releases/tag/v1.0) | MIT |
| `checkpoints/depth_anything_v2_vitl.pth` | Depth Anything V2 ViT-L | 1.34 GB | [Hugging Face: Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large) | CC-BY-NC-4.0 (non-commercial) |

Each model keeps its own license; downloading a checkpoint means accepting those terms.
The Depth Anything V2 Large weights may not be used commercially.

## Manual setup

Download the three files from the sources above, place them at the paths of the table, and
run `run_paper.py --check`: it compares the sizes and SHA-256 values with the manifest.
Do not rename another YOLO checkpoint (for example `yolo11m-seg.pt`) to `yolo11x-seg.pt`;
the size check rejects it.

To copy files you already have from another folder without overwriting anything:

```powershell
& .\prepare_local_weights.ps1 -SourceRoot 'D:\path\to\folder\with\the\same\layout'
```

## Notes

- The first inference needs network access once: HybridNets initializes an EfficientNet-B3
  ImageNet backbone from the Torch model cache before loading its own checkpoint.
- Weights are ignored by Git (`.gitignore`) and must not be committed.
- `weights_manifest.json` also lists the YOLO11s/m-seg checkpoints used only for a
  detector-swap check that is not part of the paper.
