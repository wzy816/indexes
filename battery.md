# battery

## open course

- [ECE4708/5710: Modeling, Simulation, and Identification of Battery Dynamics](http://mocha-java.uccs.edu/ECE5710/index.html)
- [ECE5718: Battery Management and Control](http://mocha-java.uccs.edu/ECE5720/index.html)
- [Anomaly Detection Multivariate Gaussian Distribution](https://www.youtube.com/watch?v=JjB56InuTqM) :movie_camera:

## SOH & SOC

- A Neural Network Approach to Absolute State-of-Health Estimation in Electric Vehicles :book:
- Advanced Machine Learning Approach for Lithium-Ion Battery State Estimation in Electric Vehicles :book:
- An Online SOC and SOH Estimation Model for Lithium-Ion Batteries :book:
  - 计算 SOC 的三类方法：数据驱动方法，I-t 积分等；适应方法，KF 滤波等；混合方法
  - 计算 SOH，考虑电池的容量衰减和内阻增加
- State of health estimation for lithium-ion battery by combing incremental capacity analysis with Gaussian process regression :book:
  - SOH 估算需考虑两个因素，容量和内阻
  - SOH 估算的方法有三类
    1. 经验或半经验模型，受不确定因素影响，有局限
    2. 电化学或物理模型，需要积分，计算复杂
    3. 数据驱动方式，有好的非线性，但需要高质量数据

## anomaly detection

### Overview

- <https://github.com/hoya012/awesome-anomaly-detection>
- <https://github.com/rob-med/awesome-TS-anomaly-detection>
- <https://github.com/zhuyiche/awesome-anomaly-detection>
- Survey on Anomaly Detection using Data Mining Techniques :book:
- Novelty Detection in Learning Systems :book:
- A Survey of Outlier Detection Methodologies :book:
- DEEP LEARNING FOR ANOMALY DETECTION: A SURVEY :book:
- Novelty Detection: A Review Part 1: Statistical Approaches :book:
- Novelty Detection: A Review Part 2: Neural network based approaches :book:
- A Comprehensive Survey of Data Mining-based Fraud Detection Research :book:
  - summary over 10 year's research
  - 2 criticisim： lack of data and lack of good methods
  - advocate for supervised learning
- A Comprehensive Survey on Outlier Detection Methods :book:
- Anomaly Detection : A Survey :book:
  - adding two classes :information theoretic and spectral techniques

### AE

- SAIFE: Unsupervised Wireless Spectrum Anomaly Detection with Interpretable Features
  - use adversarial Autoencoder
  - prefer reconstruction-base over prediction-based
  - localization via plotting x - x_hat, which also helps interpretation
- Visual Anomaly Detection in Event Sequence Data :book:
  - use LSTM-VAE
  - score calculation method: using new data to calculate LOF(k-nearest neighbour) at latent space
  - use only visual comparison to facilitate interpretation
- Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network :book:
  - propose OmniAnomaly, a GRU+VAE model
  - use POT to automatically select threshold
  - <https://www.youtube.com/watch?v=ERb_itqarsE>
  - 对于多维时序，要从整体层面而不是单个维度层面考虑异常检测，之前多维时序异常检测，要么用了 deterministic，要么忽略了 temporal dependency
  - 做了四个模型三个数据集的实验，用 F1 score 证明自己好
  - <https://github.com/NetManAIOps/OmniAnomaly>
- Generative Probabilistic Novelty Detection with Adversarial Autoencoders :book:
  - use AAE
  - <https://github.com/podgorskiy/GPND>
- Anomaly Detection with Robust Deep Autoencoders
  - use deep AE+PCA
  - <https://github.com/zc8340311/RobustAutoencoder>
- Variational Inference for On-line Anomaly Detection in High-Dimensional Time Series :book:
  - propose RNN+AE
  - train on normal data, test on normal and anomaly
- A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder :book:
  - propose LSTM-VAE, reconstruction-based detection
  - latent space is progress-based prior
  - <https://github.com/Danyleb/Variational-Lstm-Autoencoder>
- Variational Autoencoder based Anomaly Detection using Reconstruction Probability :book:
  - use reconstruction probability instead of reconstruction error
- DEEP AUTOENCODING GAUSSIAN MIXTURE MODEL FOR UNSUPERVISED ANOMALY DETECTION :book:
  - propose deep AE as compression network + GMM as estimation network
  - <https://github.com/danieltan07/dagmmDEEP>
- Multidimensional Time Series Anomaly Detection: A GRU-based Gaussian Mixture Variational Autoencoder Approach :book:
  - propose GRU(to capture correlation)+GMM(as prior)+VAE
  - GMM param k=2, latent dimension=8
- LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection :book:
  - propose LSTM+AE for multi sensor data
- DEEP UNSUPERVISED CLUSTERING WITH GAUSSIAN MIXTURE VARIATIONAL AUTOENCODERS :book:
  - use GMVAE for clustering, gaussian misture as prior
  - <https://github.com/psanch21/VAE-GMVAEMultidimensional>
- Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications :book:
- GMVAE
  - <https://github.com/jariasf/GMVAE>
- Adversarial VAE
  - Variational Autoencoder with Gaussian Anomaly Prior Distribution for Anomaly Detection
    - <https://github.com/YeongHyeon/adVAESelf-adversarial>
  - learned one-class classifier for novelty detection
    - <https://github.com/khalooei/ALOCC-CVPR2018Adversarially>
  - Probabilistic Novelty Detection with Adversarial Autoencoders
    - <https://github.com/podgorskiy/GPNDGenerative>

### GAN

- GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training :book:
  - <https://github.com/samet-akcay/ganomalyGANomaly>
- OCAN: One-Class Adversarial Nets for Fraud Detection :book:
  - <https://github.com/PanpanZheng/OCAN>
- ABNORMAL EVENT DETECTION IN VIDEOS USING GENERATIVE ADVERSARIAL NETS :book:
- Novelty Detection with GAN
  - 将多类别分类和异常检测联系在一起
  - a mixture generator trained with the Feature Matching loss as simultaneous classification
  - a novelty detection discrimintor trained with a generator that generates both nominal and novel sample
  - turn this problem into a supervised learning problem without collecting “background-class” data.
- AMAD: Adversarial Multiscale Anomaly Detection on High-Dimensional and Time-Evolving Categorical Data
  - train on normal categorical data wit noise
  - model combine AE and GAN, use adversial autoencoder
  - <https://github.com/pkumc/AMADAMAD>
- MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks :book:
  - based on LSTM
  - <https://github.com/LiDan456/MAD-GANsMAD-GAN>
- Adversarially Learned Anomaly Detection :book:
  - propose ALAD, bi-directional GAN，
  - unlike normal GAN, it can learn latent space mapping during training
  - anomaly detection based on reconstruction
  - <https://github.com/houssamzenati/Adversarially-Learned-Anomaly-Detection>
  - <https://github.com/houssamzenati/Efficient-GAN-Anomaly-Detection>
- Self-adversarial Variational Autoencoder with Gaussian Anomaly Prior Distribution for Anomaly Detection :book:
  - proose Self-adversarial Variational Autoencoder (adVAE)
  - use self-adversarial vae
  - nice introduction on limitation
  - 传统认为 normal 是高斯分布，异常是 complementary set；本文认为 normal 和异常均是高斯，在 latent space 里 overlap
  - <https://github.com/YeongHyeon/adVAE>
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
- Adversarial Multiscale Anomaly Detection on High-Dimensional and Time-Evolving Categorical Data
  - propose AMAD
- Adversarial Nets for Fraud Detection
  <https://github.com/PanpanZheng/OCANOne-Class>

### Other Models

- Learning Representations of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection :book:
- Feedforward Neural Network for Time Series Anomaly Detection :bok:
  - train on labeld raw time-series, supervised learning
  - time-series to vector process
- Long Short Term Memory Networks for Anomaly Detection in Time Serie :book:
  - use stacked LSTM
  - train on only normal data and make prediction
  - eval precision，recall，F0.1
- Time-Series Anomaly Detection Service at Microsoft :book:
  - propose Spectral Residual + CNN
- Deep Anomaly Detection with Deviation Networks :book:
  - propose devNet, and end-to-end framework
  - semi-supervised training, use a few labeled data as prior
- A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data :book:
  - propose multi-scale convolutional recurrent encoder-decoder
  - need both inter-sensor correlation and temporal dependency
- Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding :book:
- Multivariate Industrial Time Series with Cyber-Attack Simulation: Fault Detection Using an LSTM-based Predictive Data Model :book:
  - Two stacked LSTM layer, MSE as loss, RMSProp as opt
  - use MSE threshold， 0.999 percentile empirical error
- Detecting Anomalies in Space using Multivariate Convolutional LSTM with Mixtures of Probabilistic PCA :book:
  - on multi-channel/multivariate data
  - use CONV LSTM + MPPCA
- A Data-Driven Health Monitoring Method for Satellite Housekeeping Data Based on Probabilistic Clustering and Dimensionality Reduction
  - use MPPCA model
  - not considering temporal dependence
  - use percentile to remove anomaly; use previous value to imputate; nomalize 0-1，use 99.9% and 0.1% as threshold
- Battery Capacity Anomaly Detection and Data Fusion :book:
  - use kalman filter
- Collective Anomaly Detection based on Long Short Term Memory Recurrent Neural Network :book:
  - use LSTM to train on normal data
  - use prediction error detect collective anomly
- Adversarially Learned One-Class Classifier for Novelty Detection :book:
  - propose GAN +one-class classification
  - <https://github.com/khalooei/ALOCC-CVPR2018>
- Online Detection of Unusual Events in Videos via Dynamic Sparse Coding :book:
  - beautifully written
  - use dynamic sparse coding (out-of-date model?)
- A Symbolic Representation of Time Series, with Implications for Streaming Algorithms :book:
- Modeling Extreme Events in Time Series Prediction :book:
  - optimized extreme value modeling via deep learning
  - propose extreme value loss and memory network
- Adaptive-Halting Policy Network for Early Classification :book:
  - a model to handle both earliness and accuracy of classification of time series
  - propose LSTM to generate low-dimensional representation, a controller network
- Anomaly Detection: Algorithms, Explanations, Applications :book:
  - <https://www.youtube.com/watch?v=12Xq9OLdQwQ>

### Benchmark

- Systematic Construction of Anomaly Detection Benchmarks from Real Data :book:
  - benchmark 要求：正常数据点来自真实世界；异常数据点来自真实世界而且语义上不同；需要很多数据；需要定义好异常问题，且系统性多样
