# VAE

## Models

### GRU-VAE

#### Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network @ KDD 2019

- paper <https://dl.acm.org/doi/pdf/10.1145/3292500.3330672>
- code <https://github.com/NetManAIOps/OmniAnomaly>
- [anatomy](OmniAnomaly/anatomy.png)
- [graph](OmniAnomaly/graph.png)
- reference
  - https://github.com/NetManAIOps/OmniAnomaly/blob/master/main.py
  - https://github.com/NetManAIOps/OmniAnomaly/blob/master/omni_anomaly/model.py
  - https://github.com/NetManAIOps/OmniAnomaly/blob/master/omni_anomaly/vae.py
  - https://github.com/haowen-xu/tfsnippet/blob/v0.2.0-alpha1/tfsnippet/bayes.py
  - https://github.com/haowen-xu/tfsnippet/blob/v0.2.0-alpha1/tfsnippet/distributions/flow.py
  - https://github.com/thu-ml/zhusuan/blob/master/zhusuan/distributions/univariate.py

### GMM-GRU-VAE

#### Time Series Anomaly Detection: A GRU-based Gaussian Mixture Variational Autoencoder Approach @ ACML 2018

- paper <http://proceedings.mlr.press/v95/guo18a/guo18a.pdf>

### GM-VAE

#### jariasf_GMVAE

- code <https://github.com/jariasf/GMVAE>
- [anatomy](jariasf_GMVAE/anatomy.uml)
- [graph](jariasf_GMVAE/graph.png)
- references
  - https://github.com/jariasf/GMVAE/blob/master/tensorflow/main.py
  - https://github.com/jariasf/GMVAE/blob/master/tensorflow/model/GMVAE.py
  - https://github.com/jariasf/GMVAE/blob/master/tensorflow/networks/Networks.py
  - https://github.com/jariasf/GMVAE/blob/master/tensorflow/losses/LossFunctions.py

### VAE with gumble softmax distribution

- gumble-softmax paper
  - <https://arxiv.org/abs/1611.01144>
  - <https://arxiv.org/abs/1611.00712>
- code <https://github.com/ericjang/gumbel-softmax>

| model           | paper                                                                                              | implementation                                                                                                                                                          | anatomy |
| --------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| DAGMM           | AUTOENCODING GAUSSIAN MIXTURE MODEL FOR UNSUPERVISED ANOMALY DETECTION                             | [dagmmDEEP]<https://github.com/danieltan07/dagmmDEEP>                                                                                                                   |
| GMM-GRU-VAE     | DEEP UNSUPERVISED CLUSTERING PWITH GAUSSIAN MIXTURE VARIATIONAL AUTOENCODERS                       | [VAE-GMVAEMultidimensional]<https://github.com/psanch21/VAE-GMVAEMultidimensional>                                                                                      |
| LSTM-VAE        | Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder | [Variational-Lstm-Autoencoder]<https://github.com/Danyleb/Variational-Lstm-Autoencoder>                                                                                 |
| Adversarial VAE | Variational Autoencoder with Gaussian Anomaly Prior Distribution for Anomaly Detection             | [adVAESelf-adversarial]<https://github.com/YeongHyeon/adVAESelf-adversarial>                                                                                            |
| Adversarial VAE | learned one-class classifier for novelty detection                                                 | [ALOCC-CVPR2018Adversarially]<https://github.com/khalooei/ALOCC-CVPR2018Adversarially>                                                                                  |
| Adversarial VAE | Probabilistic Novelty Detection with Adversarial Autoencoders                                      | [GPNDGenerative]<https://github.com/podgorskiy/GPNDGenerative>                                                                                                          |
| GAN+VAE         | Disentangling factors of variation in deep representations using adversarial training              | [disentangling]<https://github.com/MichaelMathieu/factors-variation><br><https://github.com/ananyahjha93/disentangling-factors-of-variation-using-adversarial-training> |
| HFVAE v1        | Unsupervised Learning of Disentangled and Interpretable Representations from Sequential Data       | [FactorizedHierarchicalVAE]<https://github.com/wnhsu/FactorizedHierarchicalVAE>                                                                                         |
| HFVAE v2        | Scalable Factorized Hierarchical Variational Autoencoder Training                                  | [ScalableFHVAE]<https://github.com/wnhsu/ScalableFHVAE>                                                                                                                 |
