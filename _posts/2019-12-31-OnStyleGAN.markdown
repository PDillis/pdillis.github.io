---
layout: page
title:  "On the StyleGAN"
date:   2019-12-31 18:00:00
categories: main

---

# **Work in progress, bear with me**

* [StyleGAN](#stylegan)
    - [Progressive Growing](#progan)
* [StyleGAN2](#stylegan2)


# Getting Started

It is no surprise that a lot of computing power will be needed to generate large generated images.

Should only technical users be able to have fun with new emerging technologies? Most likely this will be the case in the beginning, but this will for sure change as we slowly mature both the algorithms and necessary technologies in order to access to these. Likewise, the necessary resources required not only for research but for [reproducing some of the results](https://github.com/ajbrock/BigGAN-PyTorch) are, quite frankly, impossible for many.

<a name="stylegan"></a>
# StyleGAN

While there is always need for more understanding of what the StyleGAN is truly doing[^faceediting] [^image2stylegan],



<a name="progan"></a>
## Progressive Growing

First, we must start with the base architecture of the StyleGAN: the [ProGAN](https://arxiv.org/abs/1710.10196). This architecture led to astounding results

<a name="wgan"></a>
### WGAN


<a name="generator"></a>
## The Generator


<a name="architecture"></a>
### Architecture

<a name="mapping"></a>
#### Mapping


```python
def G_mapping(
    latents_in,
    labels_in,
    latent_size             = 512,
    label_size              = 0,
    dlatent_size            = 512,
    dlatent_broadcast       = None,
    mapping_layers          = 8,
    mapping_fmaps           = 512,
    mapping_lrmul           = 0.01,
    mapping_nonlinearity    = 'lrelu',
    use_wscale              = True,
    normalize_latents       = True,
    dtype                   = 'float32',
    **_kwargs):

    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)),
                 'lrelu': (leaky_relu, np.sqrt(2))}[mapping_nonlinearity]

    # Inputs.
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    x = latents_in

    # Embed labels and concatenate them with latents.
    if label_size:
        with tf.compat.v1.variable_scope('LabelConcat'):
            w = tf.get_variable('weight',
                                shape=[label_size, latent_size],
                                initializer=tf.initializers.random_normal())
            y = tf.matmul(labels_in, tf.cast(w, dtype))
            x = tf.concat([x, y], axis=1)

    # Normalize latents.
    if normalize_latents:
        x = pixel_norm(x)

    # Mapping layers.
    for layer_idx in range(mapping_layers):
        with tf.compat.v1.variable_scope('Dense%d' % layer_idx):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            x = dense(x,
                      fmaps=fmaps,
                      gain=gain,
                      use_wscale=use_wscale,
                      lrmul=mapping_lrmul)
            x = apply_bias(x, lrmul=mapping_lrmul)
            x = act(x)

    # Broadcast.
    if dlatent_broadcast is not None:
        with tf.compat.v1.variable_scope('Broadcast'):
            x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')
```

<a name="synthesis"></a>
#### Synthesis

<a name="latentspaces"></a>
### The latent spaces: $\mathcal{Z}$ and $\mathcal{W}$


<a name="discriminator"></a>
## The Discriminator


<a name="threads"></a>
# On Threads

But how to apply all of this to a concrete project? While we can of course download the pretrained models done by the official authors, it is also fun to try our own experiments and monumentally fail at every step of the way, until we achieve something that captivates our eyes.

As [Helena Sarin notes](https://twitter.com/glagolista/status/1200819679209627648?s=20), a gut feeling is still needed whenever deciding if a generated image, video or work is truly worth sharing, and Threads is no


At the end, while the work I present here is certainly beautiful, I still am far more enamored with what I've [previously generated](https://blog.diegoporres.com/main/2019/07/17/UnsupervisingArt/) with ~~*lower class*~~ not state-of-the-art GANs[^dcgan]. [Bigger GANs](https://www.artnome.com/news/2018/11/14/helena-sarin-why-bigger-isnt-always-better-with-gans-and-ai-art) or better algorithms won't necessarily bring forth what we wish to express, but they should be explored in order to determine firsthand if what they capture in their latent space is close to what we wish to convey.
<iframe width="800" height="533" src="https://www.youtube.com/embed/4nktYGjSVHg?&autoplay=1&loop=1&playlist=4nktYGjSVHg" align="middle" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


<a name="stylegan2"></a>
# StyleGAN2


{% include disqus.html %}


[^faceediting]: Y. Shen, J. Gu, X. Tang & B. Zhou, [*Interpreting the Latent Space of GANs for Semantic Face Editing*](https://arxiv.org/abs/1907.10786),  2019.
[^image2stylegan]: R. Abdal, Y. Qin & P. Wonka, [*Image2StyleGAN: How to Embed Images into the StyleGAN Latent Space?*](https://arxiv.org/abs/1904.03189), 2019.
[^dcgan]: A. Radford, L. Metz & Soumith Chintala, [*Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*](https://arxiv.org/abs/1511.06434), 2016.
