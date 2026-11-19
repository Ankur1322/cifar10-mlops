# Input pipeline throughput

Measured on this machine (CPU), 20000 images, batch size 128, TensorFlow 2.21.0.

| Pipeline | Images/second |
|---|---|
| no augmentation | 297,415 |
| per-image augmentation | 32,416 |
| per-image + MixUp/CutMix | 27,464 |
