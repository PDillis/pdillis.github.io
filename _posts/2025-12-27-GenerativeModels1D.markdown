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

GANs became a hot topic after [Yann LeCun](http://yann.lecun.com/) called them "the most interesting idea in the last 10 years in ML" back in [2016](https://qr.ae/TQiKBM). The research community responded: the number of GAN variants exploded, and the quality of generated images improved dramatically.

Perhaps what has truly astonished many is the ability to generate, in a completely unsupervised way, new images that should belong to a specific dataset distribution, such as human faces. We have seemingly bypassed the [uncanny valley](https://en.wikipedia.org/wiki/Uncanny_valley) altogether.

**So why focus on 1D data instead of images?**

Most GAN tutorials jump straight to generating handwritten digits or faces. While visually impressive, many details get lost or overshadowed by the network architecture itself. By working with one-dimensional data, we can concentrate on the training loop and what the model is actually accomplishing.

## Theory: The Inverse Transform Perspective

Let's understand what GANs are doing from a different angle. Before diving into neural networks, consider a fundamental question: how does your computer generate random numbers?

### On Random Numbers

Computers generate [pseudorandom numbers](https://en.wikipedia.org/wiki/Pseudorandom_number_generator) via specific algorithms. These are sequences that approximate the properties of random numbers but are completely deterministic given the **seed** that initialized them.

```javascript
// Setting a seed gives reproducible "random" sequences
Math.seedrandom(42);
console.log(Math.random()); // Always 0.3745...
```

### From Uniform to Complex Distributions

It's "easy" to generate uniformly distributed random numbers in $[0,1]$. But what if we need samples from a more complex distribution, like the [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution)?

Thanks to the [inverse transform method](http://www.columbia.edu/~ks20/4404-Sigman/4404-Notes-ITM.pdf), we can achieve this:

1. Generate $U \sim \mathcal{U}(0,1)$
2. Obtain the inverse CDF $F_X^{-1}$
3. Compute $X = F_X^{-1}(U)$, and $X$ will have the desired distribution

**Example**: To generate exponentially distributed samples with $\lambda=0.5$:

$$X = F_X^{-1}(U) = -\frac{1}{\lambda}\log(1-U) \sim \text{Exp}(\lambda)$$

<div id="inverse-cdf-demo" class="interactive-demo">
  <h4>Interactive: Inverse CDF Sampling</h4>
  <p>See how uniform samples transform into different distributions:</p>
  <div class="demo-controls">
    <label>Target Distribution:
      <select id="target-dist">
        <option value="normal">Normal(4, 0.5)</option>
        <option value="bimodal">Bimodal Gaussian</option>
        <option value="exponential">Exponential(0.5)</option>
      </select>
    </label>
    <button id="sample-btn" class="demo-btn">Sample 1000 Points</button>
  </div>
  <div id="inverse-cdf-plot" style="width:100%; height:300px;"></div>
</div>

### Higher-Dimensional Random Numbers

Now comes the key insight. We can represent images as high-dimensional vectors. An image of size $224 \times 224 \times 3$ is just a massive vector with over 150,000 dimensions.

How many *unique* images of this size exist? Since each pixel has values 0-255 across three channels:

$$(256^3)^{224 \times 224} = 2^{1,204,224}$$

An astronomically large number. Generating random pixels will never produce a meaningful image—the space of "gibberish" vastly exceeds the space of meaningful images.

But what if we focus on a specific subset, say, images of **dogs**? The [Manifold Hypothesis](https://arxiv.org/abs/1310.0425) suggests that all dog images lie on a lower-dimensional manifold characterized by some distribution $p_{\text{dogs}}$.

If we could find this distribution, we could sample from it to generate new dog images:

1. Generate easy random numbers $U$ (uniform or normal)
2. Find $F_{\text{dogs}}^{-1}$
3. Compute $X = F_{\text{dogs}}^{-1}(U)$ — images of dogs!

Writing $F_{\text{dogs}}^{-1}$ explicitly is impossible. But neural networks are [universal function approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem). The Generator $G$ will learn to act as this inverse CDF:

$$G(z) = x \sim p_{\text{dogs}}$$

We don't need the explicit form—the Generator provides it implicitly.

## A Game Theory Perspective

How do we train this Generator? We introduce a clever adversarial setup with two networks:

- **Discriminator (D)**: Classifies inputs as real or fake
- **Generator (G)**: Transforms random noise into fake samples to fool D

They play a [minimax game](https://en.wikipedia.org/wiki/Minimax):

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

The Discriminator wants $D(x) \to 1$ for real data and $D(G(z)) \to 0$ for fakes. The Generator wants $D(G(z)) \to 1$ to fool the Discriminator.

### Training Dynamics

At equilibrium, if the Generator perfectly matches the real distribution, the Discriminator outputs $D(x) = 0.5$ everywhere—a coin flip, unable to distinguish real from fake.

This rarely happens in practice, but it's the theoretical ideal. The following figure from [Goodfellow's original paper](https://arxiv.org/abs/1406.2661) illustrates this progression:

- **(a)**: Generated and real distributions are distant
- **(b)**: Train D to optimally discriminate
- **(c)**: Train G using D's signal; distributions converge
- **(d)**: Perfect match; D outputs 0.5 everywhere

**This is exactly what we'll visualize in this post.**

## The Training Data

We'll use a simple Gaussian distribution: $p_{\text{data}} = \mathcal{N}(\mu=4.0, \sigma^2=0.25)$. This lets us clearly see when the Generator successfully learns the target.

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
      <label>Learning Rate:</label>
      <input type="range" id="learning-rate" min="-5" max="-2" step="0.1" value="-3">
      <span id="lr-display">0.001</span>
    </div>
    <div class="control-group">
      <label>Latent Dimension:</label>
      <input type="range" id="latent-dim" min="1" max="10" value="5">
      <span id="latent-display">5</span>
    </div>
    <div class="control-group">
      <label>Batch Size:</label>
      <input type="range" id="batch-size" min="6" max="10" step="1" value="8">
      <span id="batch-display">256</span>
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

## The Generator

The Generator takes a latent vector $z$ from an "easy" distribution (standard normal) and transforms it into samples that should match our target distribution.

For 1D data, we use a simple architecture:
- Input: latent vector of dimension 5
- Hidden layer: 15 neurons with ReLU activation
- Output: 1 neuron (linear activation)

```javascript
function createGenerator(latentDim, hiddenDim = 15) {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: hiddenDim,
    inputShape: [latentDim],
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));
  model.add(tf.layers.dense({
    units: hiddenDim,
    activation: 'relu'
  }));
  model.add(tf.layers.dense({
    units: 1,
    activation: 'linear'
  }));
  return model;
}
```

## The Discriminator

The Discriminator takes a sample (real or fake) and outputs the probability that it's real.

Architecture:
- Input: 1 (the sample value)
- Hidden layer: 25 neurons with ReLU
- Output: 1 neuron with sigmoid (probability)

```javascript
function createDiscriminator(hiddenDim = 25) {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: hiddenDim,
    inputShape: [1],
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));
  model.add(tf.layers.dense({
    units: hiddenDim,
    activation: 'relu'
  }));
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  }));
  return model;
}
```

## The Training Algorithm

The training alternates between updating D and G:

**For each batch:**

1. **Train Discriminator:**
   - Sample real data from the dataset
   - Generate fake data: $G(z)$ where $z \sim \mathcal{N}(0, I)$
   - Update D to maximize: $\log D(x_{\text{real}}) + \log(1 - D(x_{\text{fake}}))$

2. **Train Generator:**
   - Generate fake data
   - Update G to maximize: $\log D(G(z))$ (fool D into thinking fakes are real)

Watch the training visualization above. Initially, the Generator outputs random values. As training progresses, the green histogram (generated) should converge to match the blue histogram (real data).

## What to Observe

As you train the GAN, pay attention to:

1. **Distribution Matching**: The generated distribution should gradually align with the target
2. **Loss Curves**: Both losses should oscillate and eventually stabilize. At equilibrium, both converge toward $\log(2) \approx 0.693$
3. **Discriminator Confidence**: Watch how the discriminator's predictions evolve—it should become less confident as the generator improves

Try different settings:
- **Bimodal distribution**: Can the GAN learn two separate modes?
- **Higher latent dimension**: Does it help with more complex distributions?
- **Learning rate**: Too high causes instability; too low causes slow convergence

## Future Directions

This modular foundation extends naturally to other generative models:

- **VAE**: Replace adversarial loss with reconstruction + KL divergence
- **Diffusion**: Learn to denoise progressively corrupted data
- **Flow Matching**: Learn the vector field that transforms noise to data

The 1D setting makes these dynamics transparent before scaling to images.

---

*This post is adapted from teaching materials developed for the [Visual Recognition](https://pagines.uab.cat/mcv/content/m5-visual-recognition) module of the Master in Computer Vision at UAB/CVC. The full exercise notebook with additional theory and exercises is available for students of the program.*

<style>
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
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  min-width: 150px;
}

.control-group label {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--color-text-secondary);
}

.control-group select, .control-group input[type="range"] {
  padding: 0.5rem;
  border: 1px solid var(--color-border);
  border-radius: 0.375rem;
  background: var(--color-bg);
  color: var(--color-text-primary);
  font-family: inherit;
}

.button-group {
  display: flex;
  gap: 0.75rem;
  margin: 1rem 0;
}

.demo-btn {
  padding: 0.75rem 1.5rem;
  border: 1px solid var(--color-border);
  border-radius: 0.5rem;
  background: var(--color-bg);
  color: var(--color-text-primary);
  font-weight: 600;
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
  font-size: 0.9rem;
}

.viz-container {
  margin: 2rem 0;
}

.viz-row {
  margin-bottom: 1rem;
}
</style>

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

const config = {
  batchSize: 256,
  latentDim: 5,
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

// ============================================
// Model Creation
// ============================================

function createGenerator(latentDim, hiddenDim = 15) {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: hiddenDim,
    inputShape: [latentDim],
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));
  model.add(tf.layers.dense({
    units: hiddenDim,
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));
  model.add(tf.layers.dense({
    units: 1,
    activation: 'linear'
  }));
  return model;
}

function createDiscriminator(hiddenDim = 25) {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: hiddenDim,
    inputShape: [1],
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));
  model.add(tf.layers.dense({
    units: hiddenDim,
    activation: 'relu',
    kernelInitializer: 'heNormal'
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

async function trainStep() {
  const realBatch = sampleTargetDistribution(config.batchSize, config.distribution);

  // Train Discriminator
  const dLossVal = tf.tidy(() => {
    const grads = tf.variableGrads(() => {
      const latent = sampleLatent(config.batchSize, config.latentDim);
      const fakeBatch = generator.predict(latent);
      const realPred = discriminator.predict(realBatch);
      const fakePred = discriminator.predict(fakeBatch);
      const realLoss = tf.losses.sigmoidCrossEntropy(
        tf.ones([config.batchSize, 1]), realPred
      );
      const fakeLoss = tf.losses.sigmoidCrossEntropy(
        tf.zeros([config.batchSize, 1]), fakePred
      );
      return realLoss.add(fakeLoss);
    }, discriminator.trainableWeights);

    dOptimizer.applyGradients(grads.grads);
    return grads.value.dataSync()[0];
  });

  // Train Generator
  const gLossVal = tf.tidy(() => {
    const grads = tf.variableGrads(() => {
      const latent = sampleLatent(config.batchSize, config.latentDim);
      const fakeBatch = generator.predict(latent);
      const fakePred = discriminator.predict(fakeBatch);
      return tf.losses.sigmoidCrossEntropy(
        tf.ones([config.batchSize, 1]), fakePred
      );
    }, generator.trainableWeights);

    gOptimizer.applyGradients(grads.grads);
    return grads.value.dataSync()[0];
  });

  realBatch.dispose();
  return { gLoss: gLossVal, dLoss: dLossVal };
}

async function trainEpoch() {
  const numBatches = 10;
  let totalGLoss = 0, totalDLoss = 0;

  for (let i = 0; i < numBatches; i++) {
    const { gLoss, dLoss } = await trainStep();
    totalGLoss += gLoss;
    totalDLoss += dLoss;
  }

  epoch++;
  gLosses.push(totalGLoss / numBatches);
  dLosses.push(totalDLoss / numBatches);

  updateDisplay();
  updatePlots();
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

function updatePlots() {
  const numSamples = 1000;
  const textColor = getComputedStyle(document.documentElement).getPropertyValue('--color-text-primary') || '#333';

  tf.tidy(() => {
    const realData = sampleTargetDistribution(numSamples, config.distribution);
    const realValues = Array.from(realData.dataSync());

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
      title: `Epoch ${epoch}: Real vs Generated`, barmode: 'overlay',
      xaxis: { title: 'Value', range: [-2, 12] }, yaxis: { title: 'Density' },
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: textColor }
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
        title: 'Training Losses', xaxis: { title: 'Epoch' }, yaxis: { title: 'Loss' },
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: textColor }, legend: { x: 0.7, y: 1 }
      }, {responsive: true});
    }

    // Discriminator response plot
    const xRange = tf.linspace(-2, 12, 100);
    const dPred = discriminator.predict(xRange.reshape([100, 1]));
    const xVals = Array.from(xRange.dataSync());
    const dVals = Array.from(dPred.dataSync());

    Plotly.react('discriminator-plot', [
      { x: xVals, y: dVals, type: 'scatter', name: 'D(x)', line: { color: '#8b5cf6' } },
      { x: xVals, y: xVals.map(() => 0.5), type: 'scatter', name: 'D(x)=0.5',
        line: { color: '#f59e0b', dash: 'dash' } }
    ], {
      title: 'Discriminator Response', xaxis: { title: 'x' }, yaxis: { title: 'D(x)', range: [0, 1] },
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: textColor }
    }, {responsive: true});
  });
}

function initPlots() {
  const textColor = getComputedStyle(document.documentElement).getPropertyValue('--color-text-primary') || '#333';
  const layout = {
    xaxis: { title: 'Value', range: [-2, 12] }, yaxis: { title: 'Density' },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: textColor }
  };

  tf.tidy(() => {
    const realData = sampleTargetDistribution(1000, config.distribution);
    Plotly.newPlot('distribution-plot', [
      { x: Array.from(realData.dataSync()), type: 'histogram', name: 'Target',
        opacity: 0.7, marker: { color: '#3b82f6' }, histnorm: 'probability density', nbinsx: 50 }
    ], { ...layout, title: 'Target Distribution (click Start Training)' }, {responsive: true});
  });

  Plotly.newPlot('loss-plot', [], { ...layout, title: 'Training Losses', yaxis: { title: 'Loss' } }, {responsive: true});
  Plotly.newPlot('discriminator-plot', [], { ...layout, title: 'Discriminator Response', yaxis: { title: 'D(x)', range: [0,1] } }, {responsive: true});
}

// ============================================
// Initialization
// ============================================

function initModels() {
  if (generator) generator.dispose();
  if (discriminator) discriminator.dispose();

  generator = createGenerator(config.latentDim);
  discriminator = createDiscriminator();
  gOptimizer = tf.train.adam(config.learningRate);
  dOptimizer = tf.train.adam(config.learningRate);

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
  if (typeof tf === 'undefined') {
    console.error('TensorFlow.js not loaded');
    return;
  }

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

  document.getElementById('learning-rate').addEventListener('input', (e) => {
    config.learningRate = Math.pow(10, parseFloat(e.target.value));
    document.getElementById('lr-display').textContent = config.learningRate.toFixed(4);
    gOptimizer = tf.train.adam(config.learningRate);
    dOptimizer = tf.train.adam(config.learningRate);
  });

  document.getElementById('latent-dim').addEventListener('input', (e) => {
    config.latentDim = parseInt(e.target.value);
    document.getElementById('latent-display').textContent = config.latentDim;
    reset();
  });

  document.getElementById('batch-size').addEventListener('input', (e) => {
    config.batchSize = Math.pow(2, parseInt(e.target.value));
    document.getElementById('batch-display').textContent = config.batchSize;
  });

  // Inverse CDF demo
  document.getElementById('sample-btn').addEventListener('click', () => {
    const distType = document.getElementById('target-dist').value;
    const textColor = getComputedStyle(document.documentElement).getPropertyValue('--color-text-primary') || '#333';

    tf.tidy(() => {
      const uniform = tf.randomUniform([1000, 1]);
      const transformed = sampleTargetDistribution(1000, distType);

      Plotly.react('inverse-cdf-plot', [
        { x: Array.from(uniform.dataSync()), type: 'histogram', name: 'Uniform [0,1]',
          opacity: 0.6, marker: { color: '#64748b' }, histnorm: 'probability density', nbinsx: 30 },
        { x: Array.from(transformed.dataSync()), type: 'histogram', name: 'Transformed',
          opacity: 0.7, marker: { color: '#3b82f6' }, histnorm: 'probability density', nbinsx: 30 }
      ], {
        title: 'Inverse CDF Sampling', barmode: 'overlay',
        xaxis: { title: 'Value' }, yaxis: { title: 'Density' },
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: textColor }
      }, {responsive: true});
    });
  });

  document.getElementById('sample-btn').click();
});
</script>
