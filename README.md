# cnn_autoencoder_mnist
This repository is for comparing CNN and DNN autoencoders.

# Usage

```shell
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
conda install matplotlib
```

```shell
    python train.py
    python plot_result.py
```

# 2-dim Manifold result
*DNN* | *CNN* | *DAE* 
:---: | :---: | :---: 
<img src="result_img/1_dnn_autoencoder_2023_07_26_14_55_06.png" width=280px> | <img src="result_img/1_cnn_autoencoder_2023_07_26_17_01_53.png" width=280px> | <img src="result_img/1_DAE_autoencoder_2023_07_26_18_00_02.png" width=280px> 

# re-generation result
*DNN* | *CNN* | *DAE* 
:---: | :---: | :---: 
<img src="result_img/2_dnn_autoencoder_2023_07_26_14_55_06.png" width=280px> | <img src="result_img/2_cnn_autoencoder_2023_07_26_17_01_53.png" width=280px> | <img src="result_img/2_DAE_autoencoder_2023_07_26_18_00_02.png" width=280px>

# re-generation with noisy-input result 
*DNN* |                                    *CNN*                                     | *DAE* 
:---: |:----------------------------------------------------------------------------:| :---: 
<img src="result_img/3_dnn_autoencoder_2023_07_26_14_55_06.png" width=280px> | <img src="result_img/3_cnn_autoencoder_2023_07_26_17_01_53.png" width=280px> | <img src="result_img/3_DAE_autoencoder_2023_07_26_18_00_02.png" width=280px>

