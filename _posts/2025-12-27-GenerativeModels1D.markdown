---
layout: post
title:  "Interactive 1D Generative Models: From GANs to Diffusion"
date:   2025-12-27 12:00:00
categories: main
tags: [generative-AI, deep-learning, tensorflow-js, interactive, tutorial]
---

This post is based on an exercise I developed for the [Master in Computer Vision](https://pagines.uab.cat/mcv/) program, organized in collaboration with the [Computer Vision Center (CVC)](https://www.cvc.uab.es/) at the Universitat Autònoma de Barcelona. The goal was to help students understand the mechanics of Generative Adversarial Networks by stripping away the complexity of image generation and focusing on the simplest possible case: one-dimensional data.

I've adapted this material for a broader audience and added interactive TensorFlow.js components, so you can train these models directly in your browser and observe their learning dynamics in real-time.

## Introduction

[Generative Adversarial Networks](https://en.wikipedia.org/wiki/Generative_adversarial_network) (GANs) are one of the most widely known algorithms in Machine Learning. Being from the branch of [Unsupervised Learning](https://sites.google.com/view/berkeley-cs294-158-sp20/home), they are tasked with learning an *implicit* representation of the training dataset distribution $p_\text{data}$.

GANs became a hot topic after [Yann LeCun](http://yann.lecun.com/) called them "the most interesting idea in the last 10 years in ML" back in [2016](https://qr.ae/TQiKBM). The research community responded: the number of GAN variants exploded.

![GANs are a hot topic](https://raw.githubusercontent.com/hindupuravinash/the-gan-zoo/master/cumulative_gans.jpg "Cumulative number of named GAN papers by month")
*[Image Source: The GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo)*

Perhaps what has truly astonished many is the ability to generate, in a completely unsupervised way, new images that belong to a specific dataset distribution, such as human faces. We have seemingly bypassed the [uncanny valley](https://en.wikipedia.org/wiki/Uncanny_valley) altogether.

![4.5 year GAN progress](https://user-images.githubusercontent.com/24496178/75048968-add81280-54c9-11ea-9ea7-39dba36e7f52.png "4.5 years of GAN progress on faces")
*[Image Source: Ian Goodfellow's slides](https://www.iangoodfellow.com/slides/)*

**So why focus on 1D data instead of images?**

Most GAN tutorials jump straight to generating handwritten digits or faces. While visually impressive, many details get lost or overshadowed by the network architecture itself. By working with one-dimensional data, we can concentrate on the training loop and what the model is actually accomplishing.

## Theory: The Inverse Transform Perspective

Let's understand what GANs are doing from a different angle. Before diving into neural networks, consider a fundamental question: how does your computer generate random numbers?

### On Random Numbers

Computers generate [pseudorandom numbers](https://en.wikipedia.org/wiki/Pseudorandom_number_generator) via specific algorithms. These are sequences that approximate the properties of random numbers but are completely deterministic given the **seed** that initialized them.

```python
>>> import numpy as np
>>> np.random.seed(42)
>>> np.random.randn()
0.4967141530112327
>>> np.random.randn()
-0.1382643126401863
>>> np.random.seed(42)  # Reset seed
>>> np.random.randn()   # Same sequence!
0.4967141530112327
```

<div class="callout note">
<strong>Note:</strong> <a href="https://blog.semicolonsoftware.de/the-most-popular-random-seeds/">Many programmers have a preferred seed</a> that they always use. Feel free to take a side in this pointless "war", as we have all done before.
</div>

### From Uniform to Complex Distributions

It's "easy" to generate uniformly distributed random numbers in $[0,1]$ using algorithms like the [Mersenne Twister](https://en.wikipedia.org/wiki/Mersenne_Twister). But what if we need samples from a more complex distribution, like the [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution)?

![Exponential Distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/Exponential_probability_density.svg/800px-Exponential_probability_density.svg.png "Exponential distribution")

Thanks to the [inverse transform method](http://www.columbia.edu/~ks20/4404-Sigman/4404-Notes-ITM.pdf), we can achieve this:

1. Generate $U \sim \mathcal{U}(0,1)$
2. Obtain the inverse CDF $F_X^{-1}$
3. Compute $X = F_X^{-1}(U)$, and $X$ will have the desired distribution

<div class="callout example">
<strong>Example:</strong> Suppose we wish to generate random numbers $X$ that are exponentially distributed with $\lambda=0.5$, i.e., $X \sim \text{Exp}(\lambda=0.5)$ (the <span style="color: orange;">orange</span> curve above). We get numbers $U$ uniformly distributed in $[0,1]$, then pass them through the inverse CDF:

$$X = F_X^{-1}(U) = -\frac{1}{\lambda}\log(1-U) \sim \text{Exp}(\lambda)$$
</div>

<div id="inverse-cdf-demo" class="interactive-demo">
  <h4>Interactive: Inverse CDF Sampling</h4>
  <p>See how uniform samples transform into different distributions:</p>
  <div class="demo-controls">
    <div class="control-group">
      <label>Target Distribution:</label>
      <select id="target-dist">
        <option value="normal">Normal(4, 0.5)</option>
        <option value="bimodal">Bimodal Gaussian</option>
        <option value="exponential">Exponential(0.5)</option>
      </select>
    </div>
    <button id="sample-btn" class="demo-btn">Sample 1000 Points</button>
  </div>
  <div id="inverse-cdf-plot" style="width:100%; height:300px;"></div>
</div>

### Higher-Dimensional Random Numbers

Now comes the key insight. We can represent images as high-dimensional vectors. An image of size $224 \times 224 \times 3$ is just a massive vector with over 150,000 dimensions.

How many *unique* images of this size exist?

$$(256^3)^{224 \times 224} = 2^{1,204,224}$$

An astronomically large number that makes the [$1.2$ trillion images taken in 2018](https://theconversation.com/of-the-trillion-photos-taken-in-2018-which-were-the-most-memorable-108815) pale in comparison.

<div class="callout note">
<strong>Note:</strong> While this number is astronomically large, it is still <strong>finite</strong>. This begs the question: <a href="https://www.researchgate.net/profile/Kim_Williams10/publication/226211320_From_Tiling_the_Plane_to_Paving_Town_Square/links/0c9605375fd52b78fd000000/From-Tiling-the-Plane-to-Paving-Town-Square.pdf#page=30">is graphic art finite?</a> Even <a href="https://youtu.be/DAcjV60RnRw">music isn't safe from this question</a>.
</div>

Generating random pixels will never produce a meaningful image—the space of "gibberish" vastly exceeds the space of meaningful images. But what if we focus on a specific subset, say, images of **dogs**?

![Image Manifold](https://user-images.githubusercontent.com/24496178/75723169-0e422d80-5cdc-11ea-88e8-0c0685a07372.png "Image Manifold")
*[Image Source: Generative Deep Learning, O'Reilly](https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/)*

The [Manifold Hypothesis](https://arxiv.org/abs/1310.0425) suggests that all dog images lie on a lower-dimensional manifold characterized by some distribution $p_{\text{dogs}}$.

If we could find this distribution, we could sample from it to generate new dog images:

1. Generate easy random numbers $U$ (uniform or normal)
2. Find $F_{\text{dogs}}^{-1}$
3. Compute $X = F_{\text{dogs}}^{-1}(U)$ — images of dogs!

Writing $F_{\text{dogs}}^{-1}$ explicitly is impossible. But neural networks are [universal function approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem). The Generator $G$ will learn to act as this inverse CDF:

$$G(z) = x \sim p_{\text{dogs}}$$

We don't need the explicit form—the Generator provides it implicitly.

## A Game Theory Perspective

How do we train this Generator? We start with a clever trick. From a Machine Learning perspective, we've usually dealt with [discriminative models](https://en.wikipedia.org/wiki/Discriminative_model):

![A Discriminative network](https://user-images.githubusercontent.com/24496178/73466532-44f5f280-4382-11ea-9fc7-3dadeacda596.png "A Discriminative network")

The trick lies in adding another component: the Generator. Why not let both networks compete in a clever way?

![Typical GAN Architecture](https://user-images.githubusercontent.com/24496178/73466054-96ea4880-4381-11ea-9898-3e0dcbfaa451.png "Typical GAN Architecture")
*[Image Source](https://www.freecodecamp.org/news/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394/)*

They play a [minimax game](https://en.wikipedia.org/wiki/Minimax):

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

<div class="callout hint">
<strong>Intuition:</strong> $D$ wants to distinguish between real and fake, so $D(x)$ should be close to $1$, whereas $D(G(z))$ should be close to $0$. On the other hand, since $G$ wishes to fool $D$, it seeks to push $D(G(z))$ close to $1$. Try plotting these functions to verify this conclusion.
</div>

### Training Dynamics

**This is the key figure we aim to recreate:**

![GAN Training Distribution](https://user-images.githubusercontent.com/24496178/73085503-1635d300-3ecf-11ea-85de-1514d8085c43.png "GAN Training Distribution")
*[Image Source: Original GAN paper](https://arxiv.org/abs/1406.2661)*

At the beginning (a), the <span style="color: green;">generated</span> and real distributions are distant, but as training progresses, they will (hopefully) match. In (b), we train $D$ and it converges to the optimal solution:

$$D^{*}(x)=\frac{p_\text{data}(x)}{p_\text{data}(x)+p_g(x)}$$

In (c), we train $G$ using $D^*$'s signal and the generated distribution moves closer to the real. We continue until (d), where both match and $D^*(x) = 1/2$ everywhere—a coin flip!

<div class="callout warning">
<strong>Warning:</strong> This perfect equilibrium almost never happens, but it's the theoretical ideal. Be attentive whenever the Discriminator consistently outputs $1/2$, as it may have converged. For more details, check <a href="https://colinraffel.com/blog/gans-and-divergence-minimization.html">Colin Raffel's blog post</a>.
</div>

## Train Your Own 1D GAN

Now let's see this in action! The interactive demo below lets you train a GAN on 1D distributions directly in your browser.

<div id="training-controls" class="training-panel">
  <h3>GAN Training Playground</h3>
  <div class="control-grid">
    <div class="control-group">
      <label>Target Distribution:</label>
      <select id="data-distribution">
        <option value="normal">Normal(4.0, 0.5)</option>
        <option value="bimodal">Bimodal Gaussian</option>
        <option value="uniform">Uniform(2, 6)</option>
        <option value="exponential">Exponential(0.5)</option>
        <option value="mixture">Gaussian Mixture (3 modes)</option>
      </select>
    </div>
    <div class="control-group">
      <label>Dataset Size:</label>
      <select id="dataset-size">
        <option value="256">256</option>
        <option value="512">512</option>
        <option value="1024" selected>1024</option>
        <option value="2048">2048</option>
        <option value="4096">4096</option>
        <option value="8192">8192</option>
      </select>
    </div>
    <div class="control-group">
      <label>Learning Rate:</label>
      <input type="range" id="learning-rate" min="-4" max="-2" step="0.1" value="-3">
      <span id="lr-display">0.001</span>
    </div>
    <div class="control-group">
      <label>Batch Size:</label>
      <select id="batch-size">
        <option value="32">32</option>
        <option value="64">64</option>
        <option value="128">128</option>
        <option value="256" selected>256</option>
        <option value="512">512</option>
      </select>
    </div>
    <div class="control-group">
      <label>Latent Dim:</label>
      <input type="range" id="latent-dim" min="1" max="16" value="5">
      <span id="latent-display">5</span>
    </div>
    <div class="control-group">
      <label>G Hidden Units:</label>
      <input type="range" id="g-hidden" min="4" max="64" value="16">
      <span id="g-hidden-display">16</span>
    </div>
    <div class="control-group">
      <label>D Hidden Units:</label>
      <input type="range" id="d-hidden" min="4" max="64" value="32">
      <span id="d-hidden-display">32</span>
    </div>
  </div>
  <div class="button-group">
    <button id="train-btn" class="demo-btn primary">Start Training</button>
    <button id="reset-btn" class="demo-btn">Reset</button>
    <button id="step-btn" class="demo-btn">Step (1 epoch)</button>
  </div>
  <div id="training-status">
    <span id="epoch-counter">Epoch: 0</span>
    <span id="loss-display">G Loss: — | D Loss: —</span>
  </div>
</div>

<div id="training-visualization" class="viz-container">
  <div class="viz-row">
    <div id="distribution-plot" style="width:100%; height:350px;"></div>
  </div>
  <div class="viz-row">
    <div id="loss-plot" style="width:48%; height:250px; display:inline-block;"></div>
    <div id="discriminator-plot" style="width:48%; height:250px; display:inline-block;"></div>
  </div>
</div>

<div class="callout note">
<strong>On Batch Size:</strong> Generally in Deep Learning, we use small batch sizes ($2 \leq \texttt{batch\_size} \leq 32$) as noted in <a href="https://arxiv.org/abs/1804.07612">this paper</a>. However, since our data is extremely simple, larger batches work fine here. The authors of <a href="https://arxiv.org/abs/1809.11096">BigGAN</a> even showed that large batches (2048!) can produce better quality images.
</div>

## What to Observe

As you train the GAN, pay attention to:

1. **Distribution Matching**: The generated distribution (green) should gradually align with the target (blue)
2. **Loss Curves**: Both losses should oscillate and eventually stabilize around $\log(2) \approx 0.693$
3. **Discriminator Response**: Watch how $D(x)$ evolves—it should flatten toward $0.5$ as the generator improves

<div class="callout hint">
<strong>Experiments to Try:</strong>
<ul>
<li><strong>Bimodal distribution</strong>: Can the GAN learn two separate modes without mode collapse?</li>
<li><strong>Higher latent dimension</strong>: Does it help with more complex distributions? (StyleGAN uses 512!)</li>
<li><strong>Imbalanced capacity</strong>: What happens if D is much stronger than G?</li>
<li><strong>Learning rate</strong>: Too high causes instability; too low causes slow convergence. Try $3 \times 10^{-4}$ as suggested by <a href="https://twitter.com/karpathy/status/801621764144971776">Andrej Karpathy</a>.</li>
</ul>
</div>

## Future Directions

This modular foundation extends naturally to other generative models:

- **VAE**: Replace adversarial loss with reconstruction + KL divergence
- **Diffusion**: Learn to denoise progressively corrupted data
- **Flow Matching**: Learn the vector field that transforms noise to data

The 1D setting makes these dynamics transparent before scaling to images. For a deeper dive, check out [GAN Lab](https://poloclub.github.io/ganlab/)—a 2D interactive GAN visualization that inspired parts of this demo.

## Further Resources

- [Original GAN paper](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), Goodfellow et al., 2014
- [NIPS 2016 Tutorial: GANs](https://arxiv.org/abs/1701.00160), Ian Goodfellow ([video](https://youtu.be/HGYYEUSm-0Q))
- [From GAN to WGAN](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html), Lilian Weng
- [Open Questions About GANs](https://distill.pub/2019/gan-open-problems/), Augustus Odena
- [A Review on GANs](https://arxiv.org/abs/2001.06937), Gui et al., 2020

---

*This post is adapted from teaching materials developed for the [Visual Recognition](https://pagines.uab.cat/mcv/content/m5-visual-recognition) module of the Master in Computer Vision at UAB/CVC. The full exercise notebook with additional theory and exercises is available for students of the program.*

<style>
/* Callout boxes */
.callout {
  padding: 1rem 1.25rem;
  margin: 1.5rem 0;
  border-radius: 0.5rem;
  border-left: 4px solid;
}

.callout.note {
  background: rgba(59, 130, 246, 0.1);
  border-color: #3b82f6;
}

.callout.hint {
  background: rgba(16, 185, 129, 0.1);
  border-color: #10b981;
}

.callout.warning {
  background: rgba(245, 158, 11, 0.1);
  border-color: #f59e0b;
}

.callout.example {
  background: rgba(139, 92, 246, 0.1);
  border-color: #8b5cf6;
}

.dark-mode .callout.note { background: rgba(59, 130, 246, 0.15); }
.dark-mode .callout.hint { background: rgba(16, 185, 129, 0.15); }
.dark-mode .callout.warning { background: rgba(245, 158, 11, 0.15); }
.dark-mode .callout.example { background: rgba(139, 92, 246, 0.15); }

/* Interactive demos */
.interactive-demo, .training-panel {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 0.75rem;
  padding: 1.5rem;
  margin: 2rem 0;
}

.demo-controls, .control-grid {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 1rem;
  align-items: flex-end;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  min-width: 120px;
}

.control-group label {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--color-text-secondary);
}

.control-group select,
.control-group input[type="range"] {
  padding: 0.4rem 0.5rem;
  border: 1px solid var(--color-border);
  border-radius: 0.375rem;
  background: var(--color-bg);
  color: var(--color-text-primary);
  font-family: inherit;
  font-size: 0.9rem;
}

/* Fix dark mode dropdown text */
.dark-mode .control-group select {
  background: var(--color-surface);
  color: var(--color-text-primary);
}

.dark-mode .control-group select option {
  background: var(--color-surface);
  color: var(--color-text-primary);
}

.button-group {
  display: flex;
  gap: 0.75rem;
  margin: 1rem 0;
}

.demo-btn {
  padding: 0.6rem 1.2rem;
  border: 1px solid var(--color-border);
  border-radius: 0.5rem;
  background: var(--color-bg);
  color: var(--color-text-primary);
  font-weight: 600;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s;
}

.demo-btn:hover {
  background: var(--color-primary);
  color: white;
  border-color: var(--color-primary);
}

.demo-btn.primary {
  background: var(--color-primary);
  color: white;
  border-color: var(--color-primary);
}

.demo-btn.primary:hover {
  background: var(--color-link-hover);
}

#training-status {
  display: flex;
  gap: 2rem;
  padding: 0.75rem 1rem;
  background: var(--color-code-bg);
  border-radius: 0.375rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
}

.viz-container { margin: 2rem 0; }
.viz-row { margin-bottom: 1rem; }
</style>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>
<script>
// ============================================
// 1D GAN - TensorFlow.js Implementation
// ============================================

let generator = null;
let discriminator = null;
let gOptimizer = null;
let dOptimizer = null;
let isTraining = false;
let epoch = 0;
let gLosses = [];
let dLosses = [];
let datasetTensor = null;

const config = {
  datasetSize: 1024,
  batchSize: 256,
  latentDim: 5,
  gHidden: 16,
  dHidden: 32,
  learningRate: 0.001,
  distribution: 'normal'
};

// ============================================
// Sampling Functions
// ============================================

function sampleTargetDistribution(n, type) {
  return tf.tidy(() => {
    switch(type) {
      case 'normal':
        return tf.randomNormal([n, 1], 4.0, 0.5);
      case 'bimodal':
        const mask = tf.randomUniform([n, 1]).greater(0.5);
        const left = tf.randomNormal([n, 1], 2.0, 0.4);
        const right = tf.randomNormal([n, 1], 6.0, 0.4);
        return tf.where(mask, right, left);
      case 'uniform':
        return tf.randomUniform([n, 1], 2, 6);
      case 'exponential':
        const u = tf.randomUniform([n, 1], 0.001, 0.999);
        return tf.neg(tf.log(u)).div(0.5);
      case 'mixture':
        const r = tf.randomUniform([n, 1]);
        const m1 = tf.randomNormal([n, 1], 1.5, 0.3);
        const m2 = tf.randomNormal([n, 1], 4.0, 0.4);
        const m3 = tf.randomNormal([n, 1], 7.0, 0.25);
        const c1 = r.less(0.33);
        const c2 = r.less(0.66);
        return tf.where(c1, m1, tf.where(c2, m2, m3));
      default:
        return tf.randomNormal([n, 1], 4.0, 0.5);
    }
  });
}

function sampleLatent(n, dim) {
  return tf.randomNormal([n, dim]);
}

function generateDataset() {
  if (datasetTensor) datasetTensor.dispose();
  datasetTensor = sampleTargetDistribution(config.datasetSize, config.distribution);
}

function getBatch() {
  return tf.tidy(() => {
    const indices = tf.randomUniform([config.batchSize], 0, config.datasetSize, 'int32');
    return tf.gather(datasetTensor, indices);
  });
}

// ============================================
// Model Creation
// ============================================

function createGenerator(latentDim, hiddenDim) {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: hiddenDim,
    inputShape: [latentDim],
    activation: 'relu',
    kernelInitializer: 'glorotNormal'
  }));
  model.add(tf.layers.dense({
    units: hiddenDim,
    activation: 'relu',
    kernelInitializer: 'glorotNormal'
  }));
  model.add(tf.layers.dense({
    units: 1,
    activation: 'linear'
  }));
  return model;
}

function createDiscriminator(hiddenDim) {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: hiddenDim,
    inputShape: [1],
    activation: 'relu',
    kernelInitializer: 'glorotNormal'
  }));
  model.add(tf.layers.dense({
    units: hiddenDim,
    activation: 'relu',
    kernelInitializer: 'glorotNormal'
  }));
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  }));
  return model;
}

// ============================================
// Training
// ============================================

function trainStep() {
  let dLossVal, gLossVal;

  // Train Discriminator
  tf.tidy(() => {
    const realBatch = getBatch();
    const latent = sampleLatent(config.batchSize, config.latentDim);
    const fakeBatch = generator.predict(latent);

    // Compute D loss and gradients
    const dGrads = tf.variableGrads(() => {
      const realPred = discriminator.predict(realBatch);
      const fakePred = discriminator.predict(fakeBatch);

      const realLoss = tf.losses.sigmoidCrossEntropy(
        tf.ones([config.batchSize, 1]),
        realPred
      );
      const fakeLoss = tf.losses.sigmoidCrossEntropy(
        tf.zeros([config.batchSize, 1]),
        fakePred
      );
      return realLoss.add(fakeLoss);
    });

    dLossVal = dGrads.value.dataSync()[0];
    dOptimizer.applyGradients(dGrads.grads);

    // Dispose gradients
    Object.values(dGrads.grads).forEach(g => g.dispose());
  });

  // Train Generator
  tf.tidy(() => {
    const gGrads = tf.variableGrads(() => {
      const latent = sampleLatent(config.batchSize, config.latentDim);
      const fakeBatch = generator.predict(latent);
      const fakePred = discriminator.predict(fakeBatch);

      // Non-saturating loss: maximize log(D(G(z)))
      return tf.losses.sigmoidCrossEntropy(
        tf.ones([config.batchSize, 1]),
        fakePred
      );
    });

    gLossVal = gGrads.value.dataSync()[0];
    gOptimizer.applyGradients(gGrads.grads);

    // Dispose gradients
    Object.values(gGrads.grads).forEach(g => g.dispose());
  });

  return { gLoss: gLossVal, dLoss: dLossVal };
}

async function trainEpoch() {
  const numBatches = Math.max(1, Math.floor(config.datasetSize / config.batchSize));
  let totalGLoss = 0, totalDLoss = 0;

  for (let i = 0; i < numBatches; i++) {
    const { gLoss, dLoss } = trainStep();
    totalGLoss += gLoss;
    totalDLoss += dLoss;
  }

  epoch++;
  gLosses.push(totalGLoss / numBatches);
  dLosses.push(totalDLoss / numBatches);

  updateDisplay();
  updatePlots();

  // Memory cleanup every 10 epochs
  if (epoch % 10 === 0) {
    await tf.nextFrame();
  }
}

async function trainingLoop() {
  while (isTraining) {
    await trainEpoch();
    await tf.nextFrame();
  }
}

// ============================================
// Visualization
// ============================================

function updateDisplay() {
  document.getElementById('epoch-counter').textContent = `Epoch: ${epoch}`;
  const gL = gLosses.length > 0 ? gLosses[gLosses.length - 1].toFixed(4) : '—';
  const dL = dLosses.length > 0 ? dLosses[dLosses.length - 1].toFixed(4) : '—';
  document.getElementById('loss-display').textContent = `G Loss: ${gL} | D Loss: ${dL}`;
}

function getPlotColors() {
  const isDark = document.documentElement.classList.contains('dark-mode');
  return {
    text: isDark ? '#e2e8f0' : '#1e293b',
    grid: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'
  };
}

function updatePlots() {
  const numSamples = 1000;
  const colors = getPlotColors();

  tf.tidy(() => {
    // Real data from dataset
    const realData = sampleTargetDistribution(numSamples, config.distribution);
    const realValues = Array.from(realData.dataSync());

    // Generated data
    const latent = sampleLatent(numSamples, config.latentDim);
    const fakeData = generator.predict(latent);
    const fakeValues = Array.from(fakeData.dataSync());

    // Distribution plot
    Plotly.react('distribution-plot', [
      { x: realValues, type: 'histogram', name: 'Real Data', opacity: 0.7,
        marker: { color: '#3b82f6' }, histnorm: 'probability density', nbinsx: 50 },
      { x: fakeValues, type: 'histogram', name: 'Generated', opacity: 0.7,
        marker: { color: '#10b981' }, histnorm: 'probability density', nbinsx: 50 }
    ], {
      title: { text: `Epoch ${epoch}: Real vs Generated`, font: { color: colors.text } },
      barmode: 'overlay',
      xaxis: { title: 'Value', range: [-2, 12], color: colors.text, gridcolor: colors.grid },
      yaxis: { title: 'Density', color: colors.text, gridcolor: colors.grid },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: colors.text },
      legend: { font: { color: colors.text } }
    }, {responsive: true});

    // Loss plot
    if (gLosses.length > 0) {
      const epochs = Array.from({length: gLosses.length}, (_, i) => i);
      Plotly.react('loss-plot', [
        { x: epochs, y: gLosses, type: 'scatter', name: 'G Loss', line: { color: '#10b981' } },
        { x: epochs, y: dLosses, type: 'scatter', name: 'D Loss', line: { color: '#3b82f6' } },
        { x: epochs, y: epochs.map(() => Math.log(2)), type: 'scatter', name: 'log(2)',
          line: { color: '#f59e0b', dash: 'dash' } }
      ], {
        title: { text: 'Training Losses', font: { color: colors.text } },
        xaxis: { title: 'Epoch', color: colors.text, gridcolor: colors.grid },
        yaxis: { title: 'Loss', color: colors.text, gridcolor: colors.grid },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: colors.text },
        legend: { x: 0.7, y: 1, font: { color: colors.text } }
      }, {responsive: true});
    }

    // Discriminator response plot
    const xRange = tf.linspace(-2, 12, 100);
    const dPred = discriminator.predict(xRange.reshape([100, 1]));
    const xVals = Array.from(xRange.dataSync());
    const dVals = Array.from(dPred.dataSync());

    Plotly.react('discriminator-plot', [
      { x: xVals, y: dVals, type: 'scatter', name: 'D(x)', line: { color: '#8b5cf6', width: 2 } },
      { x: xVals, y: xVals.map(() => 0.5), type: 'scatter', name: 'D=0.5',
        line: { color: '#f59e0b', dash: 'dash' } }
    ], {
      title: { text: 'Discriminator Response', font: { color: colors.text } },
      xaxis: { title: 'x', color: colors.text, gridcolor: colors.grid },
      yaxis: { title: 'D(x)', range: [0, 1], color: colors.text, gridcolor: colors.grid },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: colors.text },
      legend: { font: { color: colors.text } }
    }, {responsive: true});
  });
}

function initPlots() {
  const colors = getPlotColors();
  const layout = {
    xaxis: { title: 'Value', range: [-2, 12], color: colors.text, gridcolor: colors.grid },
    yaxis: { title: 'Density', color: colors.text, gridcolor: colors.grid },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: colors.text }
  };

  tf.tidy(() => {
    const realData = sampleTargetDistribution(1000, config.distribution);
    Plotly.newPlot('distribution-plot', [
      { x: Array.from(realData.dataSync()), type: 'histogram', name: 'Target',
        opacity: 0.7, marker: { color: '#3b82f6' }, histnorm: 'probability density', nbinsx: 50 }
    ], { ...layout, title: { text: 'Target Distribution (click Start Training)', font: { color: colors.text } } }, {responsive: true});
  });

  Plotly.newPlot('loss-plot', [], { ...layout, title: { text: 'Training Losses', font: { color: colors.text } }, yaxis: { title: 'Loss', color: colors.text } }, {responsive: true});
  Plotly.newPlot('discriminator-plot', [], { ...layout, title: { text: 'Discriminator Response', font: { color: colors.text } }, yaxis: { title: 'D(x)', range: [0,1], color: colors.text } }, {responsive: true});
}

// ============================================
// Initialization
// ============================================

function initModels() {
  if (generator) generator.dispose();
  if (discriminator) discriminator.dispose();

  generator = createGenerator(config.latentDim, config.gHidden);
  discriminator = createDiscriminator(config.dHidden);
  gOptimizer = tf.train.adam(config.learningRate);
  dOptimizer = tf.train.adam(config.learningRate);

  generateDataset();

  epoch = 0;
  gLosses = [];
  dLosses = [];
  updateDisplay();
}

function reset() {
  isTraining = false;
  document.getElementById('train-btn').textContent = 'Start Training';
  initModels();
  initPlots();
}

// ============================================
// Event Listeners
// ============================================

document.addEventListener('DOMContentLoaded', function() {
  // Wait for TensorFlow.js to load
  if (typeof tf === 'undefined') {
    const checkTF = setInterval(() => {
      if (typeof tf !== 'undefined') {
        clearInterval(checkTF);
        initAll();
      }
    }, 100);
  } else {
    initAll();
  }

  function initAll() {
    initModels();
    initPlots();

    document.getElementById('train-btn').addEventListener('click', async () => {
      if (isTraining) {
        isTraining = false;
        document.getElementById('train-btn').textContent = 'Start Training';
      } else {
        isTraining = true;
        document.getElementById('train-btn').textContent = 'Stop Training';
        trainingLoop();
      }
    });

    document.getElementById('reset-btn').addEventListener('click', reset);

    document.getElementById('step-btn').addEventListener('click', async () => {
      if (!isTraining) await trainEpoch();
    });

    document.getElementById('data-distribution').addEventListener('change', (e) => {
      config.distribution = e.target.value;
      reset();
    });

    document.getElementById('dataset-size').addEventListener('change', (e) => {
      config.datasetSize = parseInt(e.target.value);
      generateDataset();
    });

    document.getElementById('batch-size').addEventListener('change', (e) => {
      config.batchSize = parseInt(e.target.value);
    });

    document.getElementById('learning-rate').addEventListener('input', (e) => {
      config.learningRate = Math.pow(10, parseFloat(e.target.value));
      document.getElementById('lr-display').textContent = config.learningRate.toFixed(4);
      gOptimizer = tf.train.adam(config.learningRate);
      dOptimizer = tf.train.adam(config.learningRate);
    });

    document.getElementById('latent-dim').addEventListener('input', (e) => {
      config.latentDim = parseInt(e.target.value);
      document.getElementById('latent-display').textContent = config.latentDim;
    });

    document.getElementById('g-hidden').addEventListener('input', (e) => {
      config.gHidden = parseInt(e.target.value);
      document.getElementById('g-hidden-display').textContent = config.gHidden;
    });

    document.getElementById('d-hidden').addEventListener('input', (e) => {
      config.dHidden = parseInt(e.target.value);
      document.getElementById('d-hidden-display').textContent = config.dHidden;
    });

    // Rebuild models when architecture changes
    ['latent-dim', 'g-hidden', 'd-hidden'].forEach(id => {
      document.getElementById(id).addEventListener('change', () => {
        if (!isTraining) reset();
      });
    });
  }

  // Inverse CDF demo
  const sampleBtn = document.getElementById('sample-btn');
  if (sampleBtn) {
    sampleBtn.addEventListener('click', () => {
      const distType = document.getElementById('target-dist').value;
      const colors = getPlotColors();

      tf.tidy(() => {
        const uniform = tf.randomUniform([1000, 1]);
        const transformed = sampleTargetDistribution(1000, distType);

        Plotly.react('inverse-cdf-plot', [
          { x: Array.from(uniform.dataSync()), type: 'histogram', name: 'Uniform [0,1]',
            opacity: 0.6, marker: { color: '#64748b' }, histnorm: 'probability density', nbinsx: 30 },
          { x: Array.from(transformed.dataSync()), type: 'histogram', name: 'Transformed',
            opacity: 0.7, marker: { color: '#3b82f6' }, histnorm: 'probability density', nbinsx: 30 }
        ], {
          title: { text: 'Inverse CDF Sampling', font: { color: colors.text } },
          barmode: 'overlay',
          xaxis: { title: 'Value', color: colors.text, gridcolor: colors.grid },
          yaxis: { title: 'Density', color: colors.text, gridcolor: colors.grid },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: colors.text }
        }, {responsive: true});
      });
    });

    // Initial sample
    setTimeout(() => sampleBtn.click(), 500);
  }
});
</script>
