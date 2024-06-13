# Pairwise CNN-Transformer Features
> This is the official implementation code for Pairwise CNN-Transformer Features for Human-Object Interaction Detection.
\[[__paper__](https://www.mdpi.com/1099-4300/26/3/205)\]

<img src="./assets/pct.png" align="center" height="300">

## Status
We don't update the code anymore. Please contact quanhutuo@qq.com (emil) with any questions.
If you are interested in our work, please read the UPT code first; reproducing our work is straightforward.

## Training and Testing
Refer to [`launch_template.sh`](./launch_template.sh) for training and testing commands with different options. 

To test the PCT model on HICO-DET, you can either use the Python utilities UPT implemented or the Matlab utilities provided by [Chao et al.](https://github.com/ywchao/ho-rcnn). For V-COCO, we did not implement evaluation utilities, and instead use the utilities provided by [Gupta et al.](https://github.com/s-gupta/v-coco#evaluation). Refer to these [instructions](https://github.com/fredzzhang/upt/discussions/14) for more details.

## Model Zoo
UPT provides weights for fine-tuned DETR models to facilitate reproducibility. To attempt fine-tuning the DETR model yourself, refer to [this repository](https://github.com/fredzzhang/hicodet).

|Model|Dataset|Default Settings|PCT Weights|
|:-|:-:|:-:|:-:|
|PCT-R50|HICO-DET|(`33.63`, `28.73`, `35.10`)|[weights](https://drive.google.com/file/d/1aX7My2gdk6xmzpsZc4V0TlUKgkS-tSom/view?usp=drive_link)|
|PCT-R101|HICO-DET|(`33.79`, `29.70`, `35.00`)|[weights](https://drive.google.com/file/d/1gXU4Wa5kUxWVi_JXZ6sJwDQytGnrs4T6/view?usp=drive_link)|

|Model|Dataset|Scenario 1|Scenario 2|PCT Weights|
|:-|:-:|:-:|:-:|:-:|
|PCT-R50|V-COCO|`59.4`|`65.0`|[weights](https://drive.google.com/file/d/19KMHdR_Hu5x4YDUS0UKW2Ic_gQ-nehuv/view?usp=drive_link)|
|PCT-R101|V-COCO|`61.4`|`67.1`|[weights](https://drive.google.com/file/d/1erwK1Au0iXn0PF1eJEepl-qVFRTelu7R/view?usp=drive_link)|

## Acknowledgement
Many thanks to Researcher Zhang for the valuable advice.
