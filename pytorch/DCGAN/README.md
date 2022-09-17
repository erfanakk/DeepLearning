
### Deep Convolutional GAN
_Deep Convolutional Generative Adversarial Network_

#### Authors
Alec Radford, Luke Metz, Soumith Chintala

#### Abstract
In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.

[[Paper]](https://arxiv.org/abs/1511.06434) [[Code]](model_builder.py)

<table>
  <tbody>
    <tr>
      <th>Results</th>
      <th>Configuration</th>
    </tr>
    <tr>
      <td><img src="" height="250"></td>
      <td width="50%">
        <ul>
          <li>no pre-training</li>
          <li>batch_size = 64</li>
          <li>epoch = 20</li>
          <li>noise_len = 100</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>





Training DCGAN successfully is difficult as we are trying to train two models that compete with each other at the same time, and optimisation can oscillate between solutions so much that the generator can collapse. Below are some tips on how to train a DCGAN succesfully.
1. Increase length of input noise vectors - Start with 32 and try 128 and 256
2. Decrease batch size - Start with 64 and try 32, 16 and 8. Smaller batch size generally leads to rapid learning but a volatile learning process with higher variance in the classification accuracy. Whereas larger batch sizes slow down the learning process but the final stages result in a convergence to a more stable model exemplified by lower variance in classification accuracy.
3. Add pre-training of discriminator
4. Training longer does not necessarily lead to better results - So don't set the epoch parameter too high
5. The discriminator model needs to be really good at distinguishing the fake from real images but it cannot overpower the generator, therefore both of these models should be as good as possible through maximising the depth of the network that can be supported by your machine

You can also try to configure the below settings.
1. GAN network architecture
2. Values of dropout, LeakyReLU alpha, BatchNormalization momentum
3. Change activation of generator to 'sigmoid'
4. Change optimiser from RMSProp to Adam
5. Change optimisation metric
6. Try various kinds of noise sampling, e.g. uniform sampling
7. Hard labelling
8. Separate batches of real and fake images when training discriminator
#### Run Example
```
$ python output.py
```
