
### Wasserstein GAN
_Wasserstein GAN_

#### Authors
Martin Arjovsky, Soumith Chintala, Léon Bottou

#### Abstract
We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions.

[[Paper]](https://arxiv.org/abs/1701.07875) [[Code]](model_builder.py)

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






#### Run Example
```
$ python output.py
```
