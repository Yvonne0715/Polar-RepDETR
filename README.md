# Polar-RepDETR

Pseudocode, module implementations, and bearing-defect data associated with Polar-RepDETR.

## Pseudocode

The [pseudocode](pseudocode/) describes the overall training procedure and four components:

- RepNCSPELAN4 with Coordinate Attention (CA)
- Efficient Perception Feature Interaction (EPFI)
- Linear Deformable Convolution (LDConv)
- Wise-Inner-MPDIoU loss

## Module Code

[Polar-RepDETR-allowable display-11.13](Polar-RepDETR-allowable%20display-11.13/) contains implementations of the four components and a dataset-splitting script.

This directory provides individual modules, rather than a complete detector training pipeline.

## Dataset

The released subset corresponds to **1,530 bearing-defect images**, covering three classes:

- **0:** Abrasion
- **1:** Groove
- **2:** Scratch

The subset was curated from augmented data and is not verified as the exact historical experimental dataset.

[Download the dataset](https://github.com/dingyunming/Polar-RepDETR/releases/tag/Bearing)
