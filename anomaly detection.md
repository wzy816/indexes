# anomaly detection

## General

- <https://github.com/hoya012/awesome-anomaly-detection>
- <https://github.com/rob-med/awesome-TS-anomaly-detection>
- <https://github.com/zhuyiche/awesome-anomaly-detection>

## AE

- SAIFE: Unsupervised Wireless Spectrum Anomaly Detection with Interpretable Features
  - use adversarial Autoencoder
  - prefer reconstruction-base over prediction-based
  - localization via plotting x - x_hat, which also helps interpretation
- Visual Anomaly Detection in Event Sequence Data :book:
  - use LSTM-VAE
  - score calculation method: using new data to calculate LOF(k-nearest neighbour) at latent space
  - use only visual comparison to facilitate interpretation
- Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network :book:
  - propose OmniAnomaly :framed_picture: :notebook:, a GRU+VAE model
  - use POT to automatically select threshold
  - <https://www.youtube.com/watch?v=ERb_itqarsE>
  - 对于多维时序，要从整体层面而不是单个维度层面考虑异常检测，之前多维时序异常检测，要么用了 deterministic，要么忽略了 temporal dependency
  - 做了四个模型三个数据集的实验，用 F1 score 证明自己好

## GAN

### MAD-GAN LSTM

- Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks

  - <https://github.com/LiDan456/MAD-GANsMAD-GAN>

### ALAD

- Adversarially Learned Anomaly Detection

  - <https://github.com/houssamzenati/Adversarially-Learned-Anomaly-Detection>
  - <https://github.com/houssamzenati/Efficient-GAN-Anomaly-Detection>

- Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery

  - propose anoGAN, DCGAN

  - application on eye images
    - Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery :book:
  - <https://github.com/yjucho1/anoGAN>
  - <https://github.com/LeeDoYup/AnoGAN-tf>
  - <https://github.com/fuchami/ANOGAN>
  - <https://github.com/bruvduroiu/AnoGAN-tf>
  - <https://github.com/tkwoo/anogan-keras>
  - <https://github.com/xtarx/Unsupervised-Anomaly-Detection-with-Generative-Adversarial-Networks/blob/master/README.mdUnsupervised>

### AMAD

- Adversarial Multiscale Anomaly Detection on High-Dimensional and Time-Evolving Categorical Data
  - <https://github.com/pkumc/AMADAMAD>

### GANomaly

- Semi-Supervised Anomaly Detection via Adversarial Training
  - <https://github.com/samet-akcay/ganomalyGANomaly>

### LSTM-AE + GAN,OCAN

- Adversarial Nets for Fraud Detection
  <https://github.com/PanpanZheng/OCANOne-Class>

## Benchmark

- Systematic Construction of Anomaly Detection Benchmarks from Real Data :book:
  - benchmark 要求：正常数据点来自真实世界；异常数据点来自真实世界而且语义上不同；需要很多数据；需要定义好异常问题，且系统性多样
