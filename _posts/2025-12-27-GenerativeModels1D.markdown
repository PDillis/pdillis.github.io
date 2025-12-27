---
layout: post
title:  "Interactive 1D Generative Models: From GANs to Diffusion"
date:   2025-12-27 12:00:00
categories: main
tags: [generative-AI, deep-learning, tensorflow-js, interactive, tutorial]
---

Generative Adversarial Networks, Variational Autoencoders, Diffusion Models—these algorithms power much of the recent progress in image synthesis. Yet when learning about them, most tutorials jump straight to generating images. While visually impressive, this approach obscures the underlying mechanics behind layers of convolutions, normalization tricks, and massive datasets.

This post takes a different approach: we'll implement and train generative models on **one-dimensional data**, directly in your browser using TensorFlow.js. By stripping away the complexity of images, we can focus on what these models actually do and observe their training dynamics in real-time.

## Why 1D?

Working with 1D distributions offers several pedagogical advantages:

1. **Visualize everything**: We can plot the true distribution, generated samples, and model outputs on simple 2D graphs
2. **Train instantly**: Models converge in seconds, not hours
3. **See the dynamics**: Watch how sampling works in diffusion models or how GANs reach equilibrium
4. **Compare fairly**: Same data, different algorithms, clear differences

## The Inverse Transform Sampling Perspective

Before diving into neural networks, let's understand what we're actually trying to do. Suppose we want to generate random numbers from a specific distribution. Computers can easily generate uniform random numbers in $[0, 1]$, but how do we get samples from, say, a Gaussian?

The answer is **inverse transform sampling**:

1. Generate $U \sim \text{Uniform}(0, 1)$
2. Apply the inverse CDF: $X = F^{-1}(U)$
3. $X$ now follows the target distribution!

For simple distributions (exponential, normal), we know $F^{-1}$ analytically. But for complex distributions like "all images of dogs"? That's where neural networks come in: they *learn* to transform simple noise into complex distributions.

<div id="inverse-cdf-demo" class="interactive-demo">
  <h4>Interactive: Inverse CDF Sampling</h4>
  <div class="demo-controls">
    <label>Target Distribution:
      <select id="target-dist">
        <option value="normal">Normal(4, 0.5)</option>
        <option value="bimodal">Bimodal</option>
        <option value="exponential">Exponential(0.5)</option>
      </select>
    </label>
    <button id="sample-btn" class="demo-btn">Sample 1000 Points</button>
  </div>
  <div id="inverse-cdf-plot" style="width:100%; height:300px;"></div>
</div>

## Setting Up the Playground

Let's set up our interactive training environment. All the code runs in your browser using TensorFlow.js.

<div id="training-controls" class="training-panel">
  <h3>Training Configuration</h3>
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
      <label>Model Type:</label>
      <select id="model-type">
        <option value="gan">GAN</option>
        <option value="vae" disabled>VAE (coming soon)</option>
        <option value="diffusion" disabled>Diffusion (coming soon)</option>
        <option value="flow" disabled>Flow Matching (coming soon)</option>
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
  </div>
  <div class="button-group">
    <button id="train-btn" class="demo-btn primary">Start Training</button>
    <button id="reset-btn" class="demo-btn">Reset</button>
    <button id="step-btn" class="demo-btn">Step (1 epoch)</button>
  </div>
  <div id="training-status">
    <span id="epoch-counter">Epoch: 0</span>
    <span id="loss-display">G Loss: - | D Loss: -</span>
  </div>
</div>

<div id="training-visualization" class="viz-container">
  <div class="viz-row">
    <div id="distribution-plot" style="width:100%; height:350px;"></div>
  </div>
  <div class="viz-row">
    <div id="loss-plot" style="width:48%; height:250px; display:inline-block;"></div>
    <div id="latent-plot" style="width:48%; height:250px; display:inline-block;"></div>
  </div>
</div>

## How GANs Learn Distributions

A GAN consists of two neural networks playing a game:

- **Generator (G)**: Takes random noise $z \sim \mathcal{N}(0, 1)$ and transforms it into samples
- **Discriminator (D)**: Tries to distinguish real data from generated samples

The training objective is:

$$\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

Watch the training above: initially, the generator outputs random values. As training progresses, the green curve (generated distribution) should match the blue curve (real distribution).

## The Code

Here's the modular TensorFlow.js implementation. The architecture is designed to be easily extended to other generative models:

```javascript
// === UTILITY FUNCTIONS (shared across all models) ===

// Sample from target distribution
function sampleTargetDistribution(n, type = 'normal') {
  switch(type) {
    case 'normal':
      return tf.randomNormal([n, 1], 4.0, 0.5);
    case 'bimodal':
      const mask = tf.randomUniform([n, 1]).greater(0.5);
      const left = tf.randomNormal([n, 1], 2.0, 0.3);
      const right = tf.randomNormal([n, 1], 6.0, 0.3);
      return tf.where(mask, right, left);
    case 'exponential':
      const u = tf.randomUniform([n, 1]);
      return tf.neg(tf.log(u)).div(0.5);
    default:
      return tf.randomNormal([n, 1], 4.0, 0.5);
  }
}

// Sample latent vectors
function sampleLatent(n, dim) {
  return tf.randomNormal([n, dim]);
}

// === GAN IMPLEMENTATION ===

function createGenerator(latentDim, hiddenDim = 15) {
  return tf.sequential({
    layers: [
      tf.layers.dense({
        units: hiddenDim,
        inputShape: [latentDim],
        activation: 'relu'
      }),
      tf.layers.dense({
        units: 1,
        activation: 'linear'
      })
    ]
  });
}

function createDiscriminator(hiddenDim = 25) {
  return tf.sequential({
    layers: [
      tf.layers.dense({
        units: hiddenDim,
        inputShape: [1],
        activation: 'relu'
      }),
      tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
      })
    ]
  });
}

// Training step for GAN
async function trainGANStep(generator, discriminator,
                            realData, latentDim, lr) {
  const batchSize = realData.shape[0];

  // Train Discriminator
  const dOptimizer = tf.train.adam(lr);
  const dLoss = await tf.tidy(() => {
    const latent = sampleLatent(batchSize, latentDim);
    const fakeData = generator.predict(latent);

    const realLabels = tf.ones([batchSize, 1]);
    const fakeLabels = tf.zeros([batchSize, 1]);

    // ... discriminator training logic
  });

  // Train Generator
  const gOptimizer = tf.train.adam(lr);
  const gLoss = await tf.tidy(() => {
    // ... generator training logic
  });

  return { gLoss, dLoss };
}
```

## What's Next?

This modular foundation makes it straightforward to add new generative models:

- **VAE**: Replace the GAN objective with reconstruction + KL divergence loss
- **Diffusion**: Add noise scheduling and learn the denoising score
- **Flow Matching**: Learn the vector field that transforms noise to data

The key insight is that all these methods are doing fundamentally the same thing: learning to transform simple noise into complex distributions. The 1D setting lets us see this clearly.

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
// 1D Generative Models - TensorFlow.js Implementation
// ============================================

// Global state
let generator = null;
let discriminator = null;
let gOptimizer = null;
let dOptimizer = null;
let isTraining = false;
let epoch = 0;
let gLosses = [];
let dLosses = [];
let targetData = null;

// Configuration
const config = {
  batchSize: 256,
  latentDim: 5,
  learningRate: 0.001,
  distribution: 'normal'
};

// ============================================
// Utility Functions (shared across all models)
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
// GAN Model Factory
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
// Training Logic
// ============================================

async function trainStep() {
  const realBatch = sampleTargetDistribution(config.batchSize, config.distribution);

  // Train Discriminator
  const dLoss = await tf.tidy(() => {
    return dOptimizer.minimize(() => {
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
    }, true, discriminator.trainableWeights);
  });

  // Train Generator
  const gLoss = await tf.tidy(() => {
    return gOptimizer.minimize(() => {
      const latent = sampleLatent(config.batchSize, config.latentDim);
      const fakeBatch = generator.predict(latent);
      const fakePred = discriminator.predict(fakeBatch);

      return tf.losses.sigmoidCrossEntropy(
        tf.ones([config.batchSize, 1]), fakePred
      );
    }, true, generator.trainableWeights);
  });

  realBatch.dispose();

  return {
    gLoss: gLoss.dataSync()[0],
    dLoss: dLoss.dataSync()[0]
  };
}

async function trainEpoch() {
  const numBatches = 10;
  let totalGLoss = 0;
  let totalDLoss = 0;

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
  const gL = gLosses.length > 0 ? gLosses[gLosses.length - 1].toFixed(4) : '-';
  const dL = dLosses.length > 0 ? dLosses[dLosses.length - 1].toFixed(4) : '-';
  document.getElementById('loss-display').textContent = `G Loss: ${gL} | D Loss: ${dL}`;
}

function updatePlots() {
  // Generate samples for visualization
  const numSamples = 1000;

  tf.tidy(() => {
    // Real data
    const realData = sampleTargetDistribution(numSamples, config.distribution);
    const realValues = realData.dataSync();

    // Generated data
    const latent = sampleLatent(numSamples, config.latentDim);
    const fakeData = generator.predict(latent);
    const fakeValues = fakeData.dataSync();

    // Distribution plot
    const realTrace = {
      x: Array.from(realValues),
      type: 'histogram',
      name: 'Real Data',
      opacity: 0.7,
      marker: { color: '#3b82f6' },
      histnorm: 'probability density',
      nbinsx: 50
    };

    const fakeTrace = {
      x: Array.from(fakeValues),
      type: 'histogram',
      name: 'Generated',
      opacity: 0.7,
      marker: { color: '#10b981' },
      histnorm: 'probability density',
      nbinsx: 50
    };

    const distLayout = {
      title: `Epoch ${epoch}: Real vs Generated Distribution`,
      xaxis: { title: 'Value', range: [-2, 12] },
      yaxis: { title: 'Density' },
      barmode: 'overlay',
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: getComputedStyle(document.documentElement).getPropertyValue('--color-text-primary') }
    };

    Plotly.react('distribution-plot', [realTrace, fakeTrace], distLayout, {responsive: true});

    // Loss plot
    if (gLosses.length > 0) {
      const epochs = Array.from({length: gLosses.length}, (_, i) => i);

      const gLossTrace = {
        x: epochs,
        y: gLosses,
        type: 'scatter',
        name: 'Generator Loss',
        line: { color: '#10b981' }
      };

      const dLossTrace = {
        x: epochs,
        y: dLosses,
        type: 'scatter',
        name: 'Discriminator Loss',
        line: { color: '#3b82f6' }
      };

      const log2Line = {
        x: epochs,
        y: epochs.map(() => Math.log(2)),
        type: 'scatter',
        name: 'log(2) (optimal)',
        line: { color: '#f59e0b', dash: 'dash' }
      };

      const lossLayout = {
        title: 'Training Losses',
        xaxis: { title: 'Epoch' },
        yaxis: { title: 'Loss' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: getComputedStyle(document.documentElement).getPropertyValue('--color-text-primary') },
        legend: { x: 0.7, y: 1 }
      };

      Plotly.react('loss-plot', [gLossTrace, dLossTrace, log2Line], lossLayout, {responsive: true});
    }
  });
}

function initPlots() {
  // Initialize empty distribution plot
  const emptyLayout = {
    title: 'Distribution (click Start Training)',
    xaxis: { title: 'Value', range: [-2, 12] },
    yaxis: { title: 'Density' },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: getComputedStyle(document.documentElement).getPropertyValue('--color-text-primary') }
  };

  Plotly.newPlot('distribution-plot', [], emptyLayout, {responsive: true});
  Plotly.newPlot('loss-plot', [], {
    ...emptyLayout,
    title: 'Training Losses'
  }, {responsive: true});

  // Show initial real distribution
  tf.tidy(() => {
    const realData = sampleTargetDistribution(1000, config.distribution);
    const realValues = realData.dataSync();

    const realTrace = {
      x: Array.from(realValues),
      type: 'histogram',
      name: 'Target Distribution',
      opacity: 0.7,
      marker: { color: '#3b82f6' },
      histnorm: 'probability density',
      nbinsx: 50
    };

    Plotly.react('distribution-plot', [realTrace], emptyLayout, {responsive: true});
  });
}

// ============================================
// Initialization
// ============================================

function initModels() {
  // Clean up existing models
  if (generator) generator.dispose();
  if (discriminator) discriminator.dispose();

  // Create new models
  generator = createGenerator(config.latentDim);
  discriminator = createDiscriminator();

  // Create optimizers
  gOptimizer = tf.train.adam(config.learningRate);
  dOptimizer = tf.train.adam(config.learningRate);

  // Reset state
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
  // Check if TensorFlow.js is loaded
  if (typeof tf === 'undefined') {
    console.error('TensorFlow.js not loaded');
    return;
  }

  // Initialize
  initModels();
  initPlots();

  // Training controls
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
    if (!isTraining) {
      await trainEpoch();
    }
  });

  // Configuration controls
  document.getElementById('data-distribution').addEventListener('change', (e) => {
    config.distribution = e.target.value;
    reset();
  });

  document.getElementById('learning-rate').addEventListener('input', (e) => {
    config.learningRate = Math.pow(10, parseFloat(e.target.value));
    document.getElementById('lr-display').textContent = config.learningRate.toFixed(4);
    // Update optimizers
    gOptimizer = tf.train.adam(config.learningRate);
    dOptimizer = tf.train.adam(config.learningRate);
  });

  document.getElementById('latent-dim').addEventListener('input', (e) => {
    config.latentDim = parseInt(e.target.value);
    document.getElementById('latent-display').textContent = config.latentDim;
    reset();
  });

  // Inverse CDF demo
  document.getElementById('sample-btn').addEventListener('click', () => {
    const distType = document.getElementById('target-dist').value;

    tf.tidy(() => {
      // Generate uniform samples
      const n = 1000;
      const uniform = tf.randomUniform([n, 1]);
      const uniformValues = uniform.dataSync();

      // Transform to target distribution
      const transformed = sampleTargetDistribution(n, distType);
      const transformedValues = transformed.dataSync();

      // Plot
      const uniformTrace = {
        x: Array.from(uniformValues),
        type: 'histogram',
        name: 'Uniform [0,1]',
        opacity: 0.6,
        marker: { color: '#64748b' },
        histnorm: 'probability density',
        nbinsx: 30
      };

      const transformedTrace = {
        x: Array.from(transformedValues),
        type: 'histogram',
        name: 'Transformed',
        opacity: 0.7,
        marker: { color: '#3b82f6' },
        histnorm: 'probability density',
        nbinsx: 30
      };

      const layout = {
        title: 'Inverse CDF Sampling',
        xaxis: { title: 'Value' },
        yaxis: { title: 'Density' },
        barmode: 'overlay',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: getComputedStyle(document.documentElement).getPropertyValue('--color-text-primary') }
      };

      Plotly.react('inverse-cdf-plot', [uniformTrace, transformedTrace], layout, {responsive: true});
    });
  });

  // Trigger initial sample
  document.getElementById('sample-btn').click();
});
</script>
