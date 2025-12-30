---
layout: post
title:  "Interactive 1D Generative Models: From GANs to Diffusion"
date:   2025-12-27 12:00:00
categories: main
tags: [generative-AI, deep-learning, tensorflow-js, interactive, tutorial]
---

<div id="training-controls" class="training-panel">
  <h3>1D GAN Training Playground</h3>
  <p class="demo-intro">Train a GAN to learn 1D distributions directly in your browser. Watch the Generator learn to fool the Discriminator in real-time.</p>
  <div class="control-grid">
    <div class="control-group">
      <label>Target:</label>
      <select id="data-distribution">
        <option value="normal">Normal(4, 0.5)</option>
        <option value="bimodal">Bimodal</option>
        <option value="uniform">Uniform(2, 6)</option>
        <option value="exponential">Exponential</option>
        <option value="mixture">3-Mixture</option>
      </select>
    </div>
    <div class="control-group">
      <label>Dataset:</label>
      <select id="dataset-size">
        <option value="512">512</option>
        <option value="1024">1024</option>
        <option value="2048" selected>2048</option>
        <option value="4096">4096</option>
      </select>
    </div>
    <div class="control-group">
      <label>Batch:</label>
      <select id="batch-size">
        <option value="64">64</option>
        <option value="128">128</option>
        <option value="256" selected>256</option>
      </select>
    </div>
    <div class="control-group">
      <label>LR:</label>
      <input type="range" id="learning-rate" min="-4" max="-2" step="0.1" value="-3">
      <span id="lr-display">1e-3</span>
    </div>
    <div class="control-group">
      <label>Latent:</label>
      <input type="range" id="latent-dim" min="1" max="16" value="5">
      <span id="latent-display">5</span>
    </div>
  </div>
  <div class="control-grid">
    <div class="control-group">
      <label>G Layers:</label>
      <input type="range" id="g-layers" min="1" max="4" value="2">
      <span id="g-layers-display">2</span>
    </div>
    <div class="control-group">
      <label>G Width:</label>
      <input type="range" id="g-hidden" min="8" max="64" step="8" value="16">
      <span id="g-hidden-display">16</span>
    </div>
    <div class="control-group">
      <label>D Layers:</label>
      <input type="range" id="d-layers" min="1" max="4" value="2">
      <span id="d-layers-display">2</span>
    </div>
    <div class="control-group">
      <label>D Width:</label>
      <input type="range" id="d-hidden" min="8" max="64" step="8" value="32">
      <span id="d-hidden-display">32</span>
    </div>
  </div>
  <div class="button-group">
    <button id="train-btn" class="demo-btn primary">Start Training</button>
    <button id="reset-btn" class="demo-btn">Reset</button>
    <button id="step-btn" class="demo-btn">Step</button>
  </div>
  <div id="training-status">
    <span id="epoch-counter">Epoch: 0</span>
    <span id="loss-display">G: — | D: —</span>
  </div>
</div>

<div id="training-visualization" class="viz-container">
  <div class="viz-row">
    <div id="distribution-plot" class="plot-full"></div>
  </div>
  <div class="viz-row">
    <div id="loss-plot" class="plot-half"></div>
  </div>
  <div class="viz-row">
    <div id="discriminator-plot" class="plot-half"></div>
  </div>
  <div class="viz-row">
    <div id="stats-plot" class="plot-half"></div>
  </div>
</div>

---

## Introduction

This post is based on an exercise I developed for the [Master in Computer Vision](https://pagines.uab.cat/mcv/) at the [Computer Vision Center (CVC)](https://www.cvc.uab.es/). The goal: understand GANs by working with 1D data instead of images.

[Generative Adversarial Networks](https://en.wikipedia.org/wiki/Generative_adversarial_network) (GANs) are one of the most widely known algorithms in Machine Learning. Unlike discriminative models that learn $P(y|x)$, GANs are *generative* models that learn $P(x)$ directly—they learn to generate data.

While images are the most common application, training GANs on 1D data offers several advantages for learning:

1. **Visualization**: We can directly plot the real and generated distributions
2. **Fast iteration**: Training takes seconds, not hours
3. **No GPU required**: Everything runs in your browser
4. **Clear convergence criteria**: We know the exact target distribution

<div class="callout hint">
<strong>Try the demo above!</strong> Select different distributions and watch how the GAN learns. The loss plots show convergence toward theoretical optima.
</div>

---

## On Random Numbers

Let's begin with a fundamental question: how are random numbers generated in your computer?

Perhaps surprisingly, computers don't generate truly random numbers. They use [pseudorandom number generators](https://en.wikipedia.org/wiki/Pseudorandom_number_generator) (PRNGs) that produce deterministic sequences that *appear* random. Given the same **seed**, a PRNG will always produce the same sequence:

```python
>>> import numpy as np
>>> np.random.seed(42)
>>> np.random.rand()
0.3745401188473625
>>> np.random.rand()
0.9507143064099162

>>> np.random.seed(42)  # Reset seed
>>> np.random.rand()
0.3745401188473625     # Same value!
```

This determinism is actually useful for reproducibility—you can recreate exactly the same "random" experiment by setting the same seed.

<div class="callout hint">
<strong>Note:</strong> Many programmers have a <a href="https://blog.semicolonsoftware.de/the-most-popular-random-seeds/">preferred seed</a> they always use. Common choices include 42, 0, 1, and 1337.
</div>

---

## The Inverse Transform Method

It's relatively easy to generate uniformly distributed random numbers in $[0, 1]$ using algorithms like the [Mersenne Twister](https://en.wikipedia.org/wiki/Mersenne_Twister). But what if we want samples from other distributions like Gaussian, Exponential, or something more complex?

One elegant solution is the **inverse transform method**. Given the cumulative distribution function (CDF) $F(x) = P(X \leq x)$, we can generate samples from the distribution by:

1. Sample $U \sim \text{Uniform}(0, 1)$
2. Return $X = F^{-1}(U)$

For example, the exponential distribution with rate $\lambda$ has CDF $F(x) = 1 - e^{-\lambda x}$, which gives us the inverse:

$$F^{-1}(u) = -\frac{\ln(1-u)}{\lambda}$$

```python
>>> import numpy as np
>>> lam = 0.5
>>> U = np.random.uniform(0, 1, 10000)
>>> X = -np.log(1 - U) / lam  # Exponential samples!
```

This is exactly what's happening under the hood when you call `np.random.exponential()`.

---

## Higher Dimensions: The Manifold Hypothesis

Now here's where it gets interesting. Consider images—say, $256 \times 256$ RGB images. Each image is a point in $\mathbb{R}^{256 \times 256 \times 3} = \mathbb{R}^{196608}$.

If we tried to uniformly sample this space:

```python
>>> random_image = np.random.uniform(0, 255, (256, 256, 3))
```

We'd get noise. The number of possible images is $256^{196608}$—an astronomically large number. The vast majority of this space contains meaningless noise, not recognizable images.

Here's the key insight: **meaningful images lie on a lower-dimensional manifold**. All images of dogs, for example, can be parameterized by a relatively small number of factors: breed, pose, lighting, background, etc. This is the [Manifold Hypothesis](https://arxiv.org/abs/1310.0425).

![Space of images](https://user-images.githubusercontent.com/24496178/75053034-2b9f1c00-54d1-11ea-98a7-6ad9a4b79deb.png)
*Different regions of the high-dimensional image space contain different categories of images. The manifold of dogs is distinct from the manifold of cats.*

---

## GANs as Learned Inverse CDFs

This brings us to the core idea: **GANs learn the inverse CDF of complex distributions**.

For simple distributions like Exponential, we can write $F^{-1}$ analytically. But for "the distribution of all dog images"? That's intractable.

Instead, we let a neural network $G$ learn this transformation:

$$G: z \sim \mathcal{N}(0, I) \rightarrow x \sim p_{\text{data}}$$

The Generator takes "easy" samples from a simple distribution (typically Gaussian) and transforms them into samples from the target distribution. It's learning the inverse CDF implicitly!

<div class="callout hint">
<strong>Key insight:</strong> The Generator is essentially learning how to warp a simple distribution into a complex one—just like the inverse transform method, but learned rather than derived analytically.
</div>

---

## A Game Theory Perspective

But how do we train $G$ without knowing $p_{\text{data}}$ explicitly? We only have samples from it.

The clever trick is to introduce a second network: the **Discriminator** $D$. These two networks compete in a [minimax game](https://en.wikipedia.org/wiki/Minimax):

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

- **Discriminator** $D$: Tries to distinguish real samples from fake ones. It wants $D(x) \to 1$ for real data and $D(G(z)) \to 0$ for fake data.
- **Generator** $G$: Tries to fool $D$. It wants $D(G(z)) \to 1$.

![GAN Architecture](https://user-images.githubusercontent.com/24496178/73466054-96ea4880-4381-11ea-9898-3e0dcbfaa451.png)
*The Generator transforms noise into fake samples, while the Discriminator classifies real vs. fake.*

---

## Training Dynamics

The GAN training process can be visualized as follows:

![GAN Training](https://user-images.githubusercontent.com/24496178/73085503-1635d300-3ecf-11ea-85de-1514d8085c43.png)
*From the [original GAN paper](https://arxiv.org/abs/1406.2661). As training progresses, the generated distribution (green) matches the real distribution (black dashed).*

At **Nash equilibrium**, the Discriminator can no longer distinguish real from fake:

$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} = 0.5$$

This means the optimal Discriminator outputs 0.5 for all inputs—it's essentially guessing randomly.

### Optimal Loss Values

When the GAN reaches equilibrium:

- **Generator loss**: $\log(2) \approx 0.693$
- **Discriminator loss**: $2\log(2) \approx 1.386$

Watch the demo above—you should see the losses converge toward these values!

<div class="callout warning">
<strong>Why 2·log(2) for D?</strong> The discriminator loss is the sum of two cross-entropy terms: one for real data (target=1) and one for fake data (target=0). At equilibrium, each contributes log(2), giving a total of 2·log(2).
</div>

---

## Practical Implementation

Here's how we implement the training loop. For each iteration:

### Step 1: Train the Discriminator

```python
# Sample real data
real_batch = sample_real_data(batch_size)
# Sample noise and generate fake data
z = sample_noise(batch_size, latent_dim)
fake_batch = G(z)

# Discriminator loss: maximize log(D(real)) + log(1-D(fake))
d_loss = BCE(D(real_batch), ones) + BCE(D(fake_batch), zeros)
d_optimizer.step(d_loss)
```

### Step 2: Train the Generator

```python
# Sample new noise
z = sample_noise(batch_size, latent_dim)
fake_batch = G(z)

# Generator loss: maximize log(D(fake))
# We use the "non-saturating" trick: minimize -log(D(fake))
g_loss = BCE(D(fake_batch), ones)  # Note: ones, not zeros!
g_optimizer.step(g_loss)
```

<div class="callout hint">
<strong>The Non-Saturating Trick:</strong> Instead of minimizing log(1-D(G(z))), we maximize log(D(G(z))). This provides stronger gradients early in training when D is confident that G's output is fake.
</div>

---

## What to Observe in the Demo

When you run the training demo above, watch for:

1. **Distribution Plot**: The green histogram (Generated) should gradually match the blue histogram (Real)

2. **Loss Plot**:
   - G loss (green) should approach log(2) ≈ 0.693 (yellow dashed line)
   - D loss (blue) should approach 2·log(2) ≈ 1.386 (yellow dashed line)

3. **Discriminator Response**:
   - D(x) for real data (blue) should start near 1 and drop to 0.5
   - D(G(z)) for fake data (green) should start near 0 and rise to 0.5
   - Both converging to 0.5 (yellow dashed line) means D can't tell real from fake

4. **Statistics Tracking**:
   - Mean of generated samples (green) should approach the target mean (dashed)
   - Std of generated samples (orange) should approach the target std (dashed)

---

## Common Failure Modes

### Mode Collapse

**Mode collapse** occurs when G generates only a few samples (or even just one), regardless of the input noise. This happens because G finds a "safe" output that fools D, and D can't provide useful gradients to encourage diversity.

In the demo, try making D much stronger than G (more layers, more neurons). You might see the generated distribution "collapse" to a narrow peak.

<div class="callout warning">
<strong>Signs of mode collapse:</strong>
<ul>
<li>Generated variance is much lower than target variance</li>
<li>Loss curves oscillate without settling</li>
<li>D(G(z)) oscillates instead of converging to 0.5</li>
</ul>
</div>

### Convergence Failure

If the learning rate is too high, or the architectures are poorly matched, training may fail entirely:

- G produces garbage that D easily identifies
- Gradients vanish or explode
- Losses diverge rather than stabilize

---

## Experiments to Try

Here are some experiments to deepen your understanding:

1. **Bimodal Distribution**: Can G learn both modes? Watch for mode collapse where G only captures one peak.

2. **Increase G Depth**: Does adding more layers help for complex distributions like the 3-Mixture?

3. **Imbalanced Architectures**: What happens if D is much stronger than G? Or vice versa?

4. **Small Datasets**: How few samples can the GAN learn from? Try 512 samples with a complex distribution.

5. **Latent Dimension**: Higher latent dimension gives G more capacity to express diverse outputs. But is more always better?

---

## Interpolation in High Dimensions

Once trained, we can explore the latent space. But be careful with linear interpolation! In high dimensions, linear interpolation between two points passes through regions of low probability density.

Instead, use **spherical linear interpolation** (slerp):

$$\text{slerp}(z_1, z_2, t) = \frac{\sin((1-t)\theta)}{\sin\theta}z_1 + \frac{\sin(t\theta)}{\sin\theta}z_2$$

where $\theta = \arccos\left(\frac{z_1 \cdot z_2}{\|z_1\|\|z_2\|}\right)$

This is particularly important when your latent space is high-dimensional (e.g., 100+), as linear interpolation would pass through the "hollow center" of the Gaussian distribution.

---

## Further Resources

- [Original GAN Paper](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), Goodfellow et al.
- [NIPS 2016 GAN Tutorial](https://arxiv.org/abs/1701.00160) by Goodfellow
- [GAN Lab](https://poloclub.github.io/ganlab/) — Interactive 2D GAN visualization
- [Google's GAN Course](https://developers.google.com/machine-learning/gan)
- [The GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo) — Catalog of GAN variants

---

*This post is part of the [Visual Recognition](https://pagines.uab.cat/mcv/content/m5-visual-recognition) module at the [Master in Computer Vision](https://pagines.uab.cat/mcv/), taught at [UAB](https://www.uab.cat/) and the [Computer Vision Center (CVC)](https://www.cvc.uab.es/).*

<style>
.callout { padding: 1rem 1.2rem; margin: 1.5rem 0; border-radius: 0.5rem; border-left: 4px solid; }
.callout.hint { background: rgba(16, 185, 129, 0.1); border-color: #10b981; }
.callout.warning { background: rgba(245, 158, 11, 0.1); border-color: #f59e0b; }
.dark-mode .callout.hint { background: rgba(16, 185, 129, 0.15); }
.dark-mode .callout.warning { background: rgba(245, 158, 11, 0.15); }
.callout strong { display: block; margin-bottom: 0.3rem; }
.callout ul { margin: 0.5rem 0 0 1rem; padding: 0; }
.callout li { margin: 0.25rem 0; }

.training-panel {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 0.75rem;
  padding: 1.5rem;
  margin: 0.5rem 0 1.5rem 0;
}
.training-panel h3 { margin: 0 0 0.5rem 0; color: var(--color-primary); }
.demo-intro { margin: 0.3rem 0 1rem 0; color: var(--color-text-secondary); font-size: 0.95rem; }
.control-grid { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 0.75rem; align-items: flex-end; }
.control-group { display: flex; flex-direction: column; gap: 0.2rem; min-width: 90px; flex: 1; }
.control-group label { font-size: 0.75rem; font-weight: 600; color: var(--color-text-secondary); }
.control-group select, .control-group input[type="range"] {
  padding: 0.4rem; border: 1px solid var(--color-border); border-radius: 0.3rem;
  background: var(--color-bg); color: var(--color-text-primary); font-size: 0.85rem;
}
.dark-mode .control-group select { background: #1e293b; color: #f1f5f9; }
.dark-mode .control-group select option { background: #1e293b; color: #f1f5f9; }
.button-group { display: flex; gap: 0.5rem; margin: 0.75rem 0; flex-wrap: wrap; }
.demo-btn {
  padding: 0.5rem 1rem; border: 1px solid var(--color-border); border-radius: 0.375rem;
  background: var(--color-bg); color: var(--color-text-primary); font-weight: 600; font-size: 0.85rem; cursor: pointer;
  transition: all 0.2s;
}
.demo-btn:hover { background: var(--color-primary); color: white; border-color: var(--color-primary); }
.demo-btn.primary { background: var(--color-primary); color: white; border-color: var(--color-primary); }
#training-status {
  display: flex; gap: 1.5rem; padding: 0.5rem 0.75rem; background: var(--color-code-bg);
  border-radius: 0.375rem; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
  flex-wrap: wrap;
}
.viz-container { margin: 1rem 0; }
.viz-row { margin-bottom: 0.75rem; }
.plot-full { width: 100%; height: 320px; }
.plot-half { width: 100%; height: 280px; }

@media (max-width: 768px) {
  .control-grid { gap: 0.5rem; }
  .control-group { min-width: 70px; }
  .plot-full { height: 280px; }
  .plot-half { height: 250px; }
}
</style>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>
<script>
let G=null, D=null, gOpt=null, dOpt=null, training=false, ep=0;
let gL=[], dL=[], dxHist=[], dgzHist=[], meanHist=[], stdHist=[];
let data=null, targetMean=4, targetStd=0.5;

const cfg = { N:2048, B:256, z:5, gL:2, gH:16, dL:2, dH:32, lr:0.001, dist:'normal' };

function getDistParams(t) {
  if (t==='normal') return {mean:4, std:0.5};
  if (t==='bimodal') return {mean:4, std:2}; // approx
  if (t==='uniform') return {mean:4, std:1.15}; // (6-2)/sqrt(12)
  if (t==='exponential') return {mean:2, std:2}; // 1/lambda
  if (t==='mixture') return {mean:4.17, std:2.3}; // approx
  return {mean:4, std:0.5};
}

function sample(n, t) {
  return tf.tidy(() => {
    if (t==='normal') return tf.randomNormal([n,1],4,0.5);
    if (t==='bimodal') {
      const m=tf.randomUniform([n,1]).greater(0.5);
      return tf.where(m,tf.randomNormal([n,1],6,0.4),tf.randomNormal([n,1],2,0.4));
    }
    if (t==='uniform') return tf.randomUniform([n,1],2,6);
    if (t==='exponential') return tf.neg(tf.log(tf.randomUniform([n,1],0.001,0.999))).div(0.5);
    if (t==='mixture') {
      const r=tf.randomUniform([n,1]);
      return tf.where(r.less(0.33),tf.randomNormal([n,1],1.5,0.3),
        tf.where(r.less(0.66),tf.randomNormal([n,1],4,0.4),tf.randomNormal([n,1],7,0.25)));
    }
    return tf.randomNormal([n,1],4,0.5);
  });
}

function genData() {
  if(data) data.dispose();
  data=sample(cfg.N,cfg.dist);
  const p=getDistParams(cfg.dist);
  targetMean=p.mean; targetStd=p.std;
}
function batch() { return tf.tidy(()=>tf.gather(data,tf.randomUniform([cfg.B],0,cfg.N,'int32'))); }
function latent(n) { return tf.randomNormal([n,cfg.z]); }

function makeG() {
  const m=tf.sequential();
  m.add(tf.layers.dense({units:cfg.gH,inputShape:[cfg.z],activation:'relu',kernelInitializer:'glorotNormal'}));
  for(let i=1;i<cfg.gL;i++) m.add(tf.layers.dense({units:cfg.gH,activation:'relu',kernelInitializer:'glorotNormal'}));
  m.add(tf.layers.dense({units:1,activation:'linear'}));
  return m;
}
function makeD() {
  const m=tf.sequential();
  m.add(tf.layers.dense({units:cfg.dH,inputShape:[1],activation:'relu',kernelInitializer:'glorotNormal'}));
  for(let i=1;i<cfg.dL;i++) m.add(tf.layers.dense({units:cfg.dH,activation:'relu',kernelInitializer:'glorotNormal'}));
  m.add(tf.layers.dense({units:1,activation:'sigmoid'}));
  return m;
}

function step() {
  let dl,gl,dx,dgz;
  tf.tidy(()=>{
    const real=batch(), z=latent(cfg.B), fake=G.predict(z);
    const realPred=D.predict(real), fakePred=D.predict(fake);
    dx=tf.mean(realPred).dataSync()[0];
    dgz=tf.mean(fakePred).dataSync()[0];

    const dg=tf.variableGrads(()=>{
      const rp=D.predict(real), fp=D.predict(fake);
      return tf.losses.sigmoidCrossEntropy(tf.ones([cfg.B,1]),rp)
        .add(tf.losses.sigmoidCrossEntropy(tf.zeros([cfg.B,1]),fp));
    });
    dl=dg.value.dataSync()[0]; dOpt.applyGradients(dg.grads);
    Object.values(dg.grads).forEach(g=>g.dispose());
  });

  let genMean, genStd;
  tf.tidy(()=>{
    const gg=tf.variableGrads(()=>{
      const z=latent(cfg.B), fake=G.predict(z), fp=D.predict(fake);
      return tf.losses.sigmoidCrossEntropy(tf.ones([cfg.B,1]),fp);
    });
    gl=gg.value.dataSync()[0]; gOpt.applyGradients(gg.grads);
    Object.values(gg.grads).forEach(g=>g.dispose());

    // Calculate generated stats
    const z=latent(500), fake=G.predict(z);
    const m=tf.moments(fake);
    genMean=m.mean.dataSync()[0];
    genStd=Math.sqrt(m.variance.dataSync()[0]);
  });

  return {gl,dl,dx,dgz,genMean,genStd};
}

async function epoch() {
  const nb=Math.max(1,Math.floor(cfg.N/cfg.B));
  let tg=0,td=0,tdx=0,tdgz=0,tm=0,ts=0;
  for(let i=0;i<nb;i++){
    const{gl,dl,dx,dgz,genMean,genStd}=step();
    tg+=gl;td+=dl;tdx+=dx;tdgz+=dgz;tm+=genMean;ts+=genStd;
  }
  ep++;
  gL.push(tg/nb); dL.push(td/nb);
  dxHist.push(tdx/nb); dgzHist.push(tdgz/nb);
  meanHist.push(tm/nb); stdHist.push(ts/nb);
  updDisp(); updPlots();
  if(ep%5===0) await tf.nextFrame();
}

async function loop() { while(training){await epoch();await tf.nextFrame();} }

function updDisp() {
  document.getElementById('epoch-counter').textContent=`Epoch: ${ep}`;
  const g=gL.length?gL[gL.length-1].toFixed(3):'—';
  const d=dL.length?dL[dL.length-1].toFixed(3):'—';
  document.getElementById('loss-display').textContent=`G: ${g} | D: ${d}`;
}

function style() {
  const dk=document.documentElement.classList.contains('dark-mode');
  return {
    txt:dk?'#f1f5f9':'#1e293b',
    ax:dk?'#cbd5e1':'#475569',
    gr:dk?'rgba(255,255,255,0.1)':'rgba(0,0,0,0.06)',
    bg:'rgba(0,0,0,0)'
  };
}

function updPlots() {
  const s=style(), n=800;
  tf.tidy(()=>{
    const real=sample(n,cfg.dist), rV=Array.from(real.dataSync());
    const z=latent(n), fake=G.predict(z), fV=Array.from(fake.dataSync());

    // Distribution plot
    Plotly.react('distribution-plot',[
      {x:rV,type:'histogram',name:'Real',opacity:0.7,marker:{color:'#3b82f6'},histnorm:'probability density',nbinsx:50},
      {x:fV,type:'histogram',name:'Generated',opacity:0.7,marker:{color:'#10b981'},histnorm:'probability density',nbinsx:50}
    ],{
      title:{text:`Distribution — Epoch ${ep}`,font:{color:s.txt,size:14}},
      barmode:'overlay',
      xaxis:{title:'x',range:[-1,10],color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      yaxis:{title:'Density',color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      paper_bgcolor:s.bg,plot_bgcolor:s.bg,font:{color:s.txt},
      legend:{x:0.75,y:1,font:{color:s.txt}},margin:{t:40,b:45,l:50,r:20}
    },{responsive:true});
  });

  if(gL.length>0){
    const e=Array.from({length:gL.length},(_,i)=>i);
    const log2=Math.log(2), log2x2=2*Math.log(2);

    // Loss plot with theoretical optimal lines
    Plotly.react('loss-plot',[
      {x:e,y:gL,type:'scatter',name:'G Loss',line:{color:'#10b981',width:2}},
      {x:e,y:dL,type:'scatter',name:'D Loss',line:{color:'#3b82f6',width:2}},
      {x:[0,e.length-1],y:[log2,log2],type:'scatter',name:'G Optimal',line:{color:'#fbbf24',width:2,dash:'dash'}},
      {x:[0,e.length-1],y:[log2x2,log2x2],type:'scatter',name:'D Optimal',line:{color:'#f59e0b',width:2,dash:'dash'}}
    ],{
      title:{text:'Loss Curves',font:{color:s.txt,size:14}},
      xaxis:{title:'Epoch',color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      yaxis:{title:'Loss',color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      paper_bgcolor:s.bg,plot_bgcolor:s.bg,font:{color:s.txt},
      legend:{x:0.65,y:1,font:{color:s.txt,size:10}},margin:{t:40,b:45,l:50,r:20},
      showlegend:true
    },{responsive:true});

    // Discriminator output over time
    Plotly.react('discriminator-plot',[
      {x:e,y:dxHist,type:'scatter',name:'D(x) real',line:{color:'#3b82f6',width:2}},
      {x:e,y:dgzHist,type:'scatter',name:'D(G(z)) fake',line:{color:'#10b981',width:2}},
      {x:[0,e.length-1],y:[0.5,0.5],type:'scatter',name:'Optimal',line:{color:'#fbbf24',width:2,dash:'dash'}}
    ],{
      title:{text:'Discriminator Response',font:{color:s.txt,size:14}},
      xaxis:{title:'Epoch',color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      yaxis:{title:'D output',range:[0,1],color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      paper_bgcolor:s.bg,plot_bgcolor:s.bg,font:{color:s.txt},
      legend:{x:0.6,y:1,font:{color:s.txt,size:10}},margin:{t:40,b:45,l:50,r:20}
    },{responsive:true});

    // Statistics plot (mean and std)
    Plotly.react('stats-plot',[
      {x:e,y:meanHist,type:'scatter',name:'Gen Mean',line:{color:'#10b981',width:2}},
      {x:e,y:stdHist,type:'scatter',name:'Gen Std',line:{color:'#f97316',width:2}},
      {x:[0,e.length-1],y:[targetMean,targetMean],type:'scatter',name:'Target Mean',line:{color:'#10b981',width:2,dash:'dash'}},
      {x:[0,e.length-1],y:[targetStd,targetStd],type:'scatter',name:'Target Std',line:{color:'#f97316',width:2,dash:'dash'}}
    ],{
      title:{text:'Generated Statistics',font:{color:s.txt,size:14}},
      xaxis:{title:'Epoch',color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      yaxis:{title:'Value',color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      paper_bgcolor:s.bg,plot_bgcolor:s.bg,font:{color:s.txt},
      legend:{x:0.6,y:1,font:{color:s.txt,size:10}},margin:{t:40,b:45,l:50,r:20}
    },{responsive:true});
  }
}

function initPlots() {
  const s=style();
  tf.tidy(()=>{
    const r=sample(800,cfg.dist);
    Plotly.newPlot('distribution-plot',[
      {x:Array.from(r.dataSync()),type:'histogram',name:'Target',opacity:0.7,marker:{color:'#3b82f6'},histnorm:'probability density',nbinsx:50}
    ],{
      title:{text:'Click Start to Train',font:{color:s.txt,size:14}},
      xaxis:{title:'x',range:[-1,10],color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      yaxis:{title:'Density',color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      paper_bgcolor:s.bg,plot_bgcolor:s.bg,font:{color:s.txt},margin:{t:40,b:45,l:50,r:20}
    },{responsive:true});
  });

  const baseLay={paper_bgcolor:s.bg,plot_bgcolor:s.bg,font:{color:s.txt},
    xaxis:{title:'Epoch',color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
    yaxis:{color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
    margin:{t:40,b:45,l:50,r:20}};

  Plotly.newPlot('loss-plot',[],{...baseLay,title:{text:'Loss Curves',font:{color:s.txt,size:14}}},{responsive:true});
  Plotly.newPlot('discriminator-plot',[],{...baseLay,title:{text:'Discriminator Response',font:{color:s.txt,size:14}},yaxis:{...baseLay.yaxis,range:[0,1]}},{responsive:true});
  Plotly.newPlot('stats-plot',[],{...baseLay,title:{text:'Generated Statistics',font:{color:s.txt,size:14}}},{responsive:true});
}

function init() {
  if(G) G.dispose(); if(D) D.dispose();
  G=makeG(); D=makeD();
  gOpt=tf.train.adam(cfg.lr); dOpt=tf.train.adam(cfg.lr);
  genData(); ep=0; gL=[]; dL=[]; dxHist=[]; dgzHist=[]; meanHist=[]; stdHist=[];
  updDisp();
}

function reset() {
  training=false;
  document.getElementById('train-btn').textContent='Start Training';
  init(); initPlots();
}

document.addEventListener('DOMContentLoaded',()=>{
  const wait=setInterval(()=>{
    if(typeof tf!=='undefined' && typeof Plotly!=='undefined'){
      clearInterval(wait);
      init(); initPlots();

      document.getElementById('train-btn').onclick=async()=>{
        if(training){
          training=false;
          document.getElementById('train-btn').textContent='Start Training';
        } else {
          training=true;
          document.getElementById('train-btn').textContent='Stop';
          loop();
        }
      };
      document.getElementById('reset-btn').onclick=reset;
      document.getElementById('step-btn').onclick=async()=>{if(!training)await epoch();};

      document.getElementById('data-distribution').onchange=e=>{cfg.dist=e.target.value;reset();};
      document.getElementById('dataset-size').onchange=e=>{cfg.N=+e.target.value;genData();};
      document.getElementById('batch-size').onchange=e=>{cfg.B=+e.target.value;};
      document.getElementById('learning-rate').oninput=e=>{
        cfg.lr=Math.pow(10,+e.target.value);
        document.getElementById('lr-display').textContent=cfg.lr.toExponential(0);
        gOpt=tf.train.adam(cfg.lr);dOpt=tf.train.adam(cfg.lr);
      };

      const sliders=['latent-dim','g-layers','g-hidden','d-layers','d-hidden'];
      const keys=['z','gL','gH','dL','dH'];
      sliders.forEach((id,i)=>{
        const el=document.getElementById(id);
        el.oninput=e=>{
          cfg[keys[i]]=+e.target.value;
          document.getElementById(id+'-display').textContent=e.target.value;
        };
        el.onchange=()=>{if(!training)reset();};
      });
    }
  },100);
});
</script>
