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
    <div id="distribution-plot" style="width:100%; height:300px;"></div>
  </div>
  <div class="viz-row">
    <div id="loss-plot" style="width:48%; height:200px; display:inline-block;"></div>
    <div id="discriminator-plot" style="width:48%; height:200px; display:inline-block;"></div>
  </div>
</div>

---

## About This Demo

This post is based on an exercise I developed for the [Master in Computer Vision](https://pagines.uab.cat/mcv/) at the [Computer Vision Center (CVC)](https://www.cvc.uab.es/). The goal: understand GANs by working with 1D data instead of images.

**What to observe:**
- **Distribution**: Generated samples (green) should match the target (blue)
- **Losses**: At equilibrium, G loss → log(2) ≈ 0.693, D loss → 2·log(2) ≈ 1.386
- **Discriminator**: D(x) (blue line) and D(G(z)) (green dots) should both converge to 0.5

## How GANs Work

Two networks compete in a [minimax game](https://en.wikipedia.org/wiki/Minimax):

$$\min_G \max_D \, \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]$$

- **Generator G**: Transforms noise $z$ into fake samples
- **Discriminator D**: Classifies real vs. fake

![GAN Architecture](https://user-images.githubusercontent.com/24496178/73466054-96ea4880-4381-11ea-9898-3e0dcbfaa451.png)

<div class="callout hint">
<strong>Intuition:</strong> D wants D(x)→1 for real, D(G(z))→0 for fake. G wants D(G(z))→1.
</div>

### Training Dynamics

![GAN Training](https://user-images.githubusercontent.com/24496178/73085503-1635d300-3ecf-11ea-85de-1514d8085c43.png)
*From the [original GAN paper](https://arxiv.org/abs/1406.2661)*

At equilibrium, $D^*(x) = 0.5$ everywhere—D can't distinguish real from fake.

<div class="callout warning">
<strong>Mode collapse:</strong> If G only produces one value, it found a "cheat" that fools D. Try resetting or adjusting architecture.
</div>

## The Inverse Transform Perspective

Neural networks learn to transform "easy" distributions (uniform, normal) into complex ones—essentially learning the inverse CDF:

```python
>>> import numpy as np
>>> np.random.seed(42)
>>> np.random.randn()
0.4967141530112327
```

For simple distributions, we use closed-form inverse CDFs. For complex ones like "all dog images"? The Generator learns it implicitly:

$$G(z) = x \sim p_{\text{data}}$$

## Experiments

<div class="callout hint">
<ul>
<li><strong>Bimodal</strong>: Can G learn both modes?</li>
<li><strong>More layers</strong>: Does depth help for complex distributions?</li>
<li><strong>D stronger than G</strong>: What happens?</li>
<li><strong>Small dataset</strong>: How few samples work?</li>
</ul>
</div>

## Resources

- [GAN paper](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), Goodfellow et al.
- [GAN Lab](https://poloclub.github.io/ganlab/)—2D interactive demo
- [NIPS 2016 Tutorial](https://arxiv.org/abs/1701.00160)

---

*From the [Visual Recognition](https://pagines.uab.cat/mcv/content/m5-visual-recognition) module at UAB/CVC.*

<style>
.callout { padding: 0.8rem 1rem; margin: 1.2rem 0; border-radius: 0.4rem; border-left: 4px solid; }
.callout.hint { background: rgba(16, 185, 129, 0.1); border-color: #10b981; }
.callout.warning { background: rgba(245, 158, 11, 0.1); border-color: #f59e0b; }
.dark-mode .callout.hint { background: rgba(16, 185, 129, 0.2); }
.dark-mode .callout.warning { background: rgba(245, 158, 11, 0.2); }

.training-panel {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 0.6rem;
  padding: 1.2rem;
  margin: 0.5rem 0 1.5rem 0;
}
.demo-intro { margin: 0.3rem 0 0.8rem 0; color: var(--color-text-secondary); font-size: 0.9rem; }
.control-grid { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.6rem; align-items: flex-end; }
.control-group { display: flex; flex-direction: column; gap: 0.15rem; min-width: 80px; flex: 1; }
.control-group label { font-size: 0.7rem; font-weight: 600; color: var(--color-text-secondary); }
.control-group select, .control-group input[type="range"] {
  padding: 0.3rem; border: 1px solid var(--color-border); border-radius: 0.25rem;
  background: var(--color-bg); color: var(--color-text-primary); font-size: 0.8rem;
}
.dark-mode .control-group select { background: #1e293b; color: #f1f5f9; }
.dark-mode .control-group select option { background: #1e293b; color: #f1f5f9; }
.button-group { display: flex; gap: 0.4rem; margin: 0.6rem 0; }
.demo-btn {
  padding: 0.4rem 0.8rem; border: 1px solid var(--color-border); border-radius: 0.3rem;
  background: var(--color-bg); color: var(--color-text-primary); font-weight: 600; font-size: 0.8rem; cursor: pointer;
}
.demo-btn:hover { background: var(--color-primary); color: white; border-color: var(--color-primary); }
.demo-btn.primary { background: var(--color-primary); color: white; border-color: var(--color-primary); }
#training-status {
  display: flex; gap: 1rem; padding: 0.4rem 0.6rem; background: var(--color-code-bg);
  border-radius: 0.25rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
}
.viz-container { margin: 0.8rem 0; }
.viz-row { margin-bottom: 0.4rem; }
</style>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>
<script>
let G=null, D=null, gOpt=null, dOpt=null, training=false, ep=0, gL=[], dL=[], data=null;
const cfg = { N:2048, B:256, z:5, gL:2, gH:16, dL:2, dH:32, lr:0.001, dist:'normal' };

function sample(n, t) {
  return tf.tidy(() => {
    if (t==='normal') return tf.randomNormal([n,1],4,0.5);
    if (t==='bimodal') { const m=tf.randomUniform([n,1]).greater(0.5); return tf.where(m,tf.randomNormal([n,1],6,0.4),tf.randomNormal([n,1],2,0.4)); }
    if (t==='uniform') return tf.randomUniform([n,1],2,6);
    if (t==='exponential') return tf.neg(tf.log(tf.randomUniform([n,1],0.001,0.999))).div(0.5);
    if (t==='mixture') { const r=tf.randomUniform([n,1]); return tf.where(r.less(0.33),tf.randomNormal([n,1],1.5,0.3),tf.where(r.less(0.66),tf.randomNormal([n,1],4,0.4),tf.randomNormal([n,1],7,0.25))); }
    return tf.randomNormal([n,1],4,0.5);
  });
}
function genData() { if(data) data.dispose(); data=sample(cfg.N,cfg.dist); }
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
  let dl,gl;
  tf.tidy(()=>{
    const real=batch(), z=latent(cfg.B), fake=G.predict(z);
    const dg=tf.variableGrads(()=>{
      const rp=D.predict(real), fp=D.predict(fake);
      return tf.losses.sigmoidCrossEntropy(tf.ones([cfg.B,1]),rp).add(tf.losses.sigmoidCrossEntropy(tf.zeros([cfg.B,1]),fp));
    });
    dl=dg.value.dataSync()[0]; dOpt.applyGradients(dg.grads);
    Object.values(dg.grads).forEach(g=>g.dispose());
  });
  tf.tidy(()=>{
    const gg=tf.variableGrads(()=>{
      const z=latent(cfg.B), fake=G.predict(z), fp=D.predict(fake);
      return tf.losses.sigmoidCrossEntropy(tf.ones([cfg.B,1]),fp);
    });
    gl=gg.value.dataSync()[0]; gOpt.applyGradients(gg.grads);
    Object.values(gg.grads).forEach(g=>g.dispose());
  });
  return {gl,dl};
}

async function epoch() {
  const nb=Math.max(1,Math.floor(cfg.N/cfg.B));
  let tg=0,td=0;
  for(let i=0;i<nb;i++){const{gl,dl}=step();tg+=gl;td+=dl;}
  ep++; gL.push(tg/nb); dL.push(td/nb);
  updDisp(); updPlots();
  if(ep%10===0) await tf.nextFrame();
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
  return { txt:dk?'#f1f5f9':'#1e293b', ax:dk?'#e2e8f0':'#475569', gr:dk?'rgba(255,255,255,0.12)':'rgba(0,0,0,0.08)' };
}

function updPlots() {
  const s=style(), n=800;
  tf.tidy(()=>{
    const real=sample(n,cfg.dist), rV=Array.from(real.dataSync());
    const z=latent(n), fake=G.predict(z), fV=Array.from(fake.dataSync());

    Plotly.react('distribution-plot',[
      {x:rV,type:'histogram',name:'Real',opacity:0.7,marker:{color:'#3b82f6'},histnorm:'probability density',nbinsx:40},
      {x:fV,type:'histogram',name:'Gen',opacity:0.7,marker:{color:'#10b981'},histnorm:'probability density',nbinsx:40}
    ],{
      title:{text:`Epoch ${ep}`,font:{color:s.txt,size:13}},barmode:'overlay',
      xaxis:{title:'x',range:[-1,10],color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      yaxis:{title:'Density',color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:s.txt},
      legend:{font:{color:s.txt}},margin:{t:35,b:35,l:45,r:15}
    },{responsive:true});

    if(gL.length){
      const e=Array.from({length:gL.length},(_,i)=>i);
      Plotly.react('loss-plot',[
        {x:e,y:gL,type:'scatter',name:'G',line:{color:'#10b981',width:1.5}},
        {x:e,y:dL,type:'scatter',name:'D',line:{color:'#3b82f6',width:1.5}}
      ],{
        title:{text:'Loss',font:{color:s.txt,size:12}},
        xaxis:{title:'Epoch',color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
        yaxis:{color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
        paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:s.txt},
        legend:{x:0.75,y:1,font:{color:s.txt}},margin:{t:30,b:30,l:40,r:10}
      },{responsive:true});
    }

    const xR=tf.linspace(-1,10,80), dR=D.predict(xR.reshape([80,1]));
    const xV=Array.from(xR.dataSync()), dRV=Array.from(dR.dataSync());
    const zS=latent(60), gS=G.predict(zS), dF=D.predict(gS);
    const gSV=Array.from(gS.dataSync()), dFV=Array.from(dF.dataSync());

    Plotly.react('discriminator-plot',[
      {x:xV,y:dRV,type:'scatter',name:'D(x)',line:{color:'#3b82f6',width:1.5}},
      {x:gSV,y:dFV,type:'scatter',mode:'markers',name:'D(G(z))',marker:{color:'#10b981',size:4,opacity:0.7}}
    ],{
      title:{text:'Discriminator',font:{color:s.txt,size:12}},
      xaxis:{title:'x',range:[-1,10],color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      yaxis:{title:'D(·)',range:[0,1],color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:s.txt},
      legend:{x:0.7,y:1,font:{color:s.txt}},margin:{t:30,b:30,l:40,r:10}
    },{responsive:true});
  });
}

function initPlots() {
  const s=style();
  tf.tidy(()=>{
    const r=sample(800,cfg.dist);
    Plotly.newPlot('distribution-plot',[
      {x:Array.from(r.dataSync()),type:'histogram',name:'Target',opacity:0.7,marker:{color:'#3b82f6'},histnorm:'probability density',nbinsx:40}
    ],{
      title:{text:'Click Start',font:{color:s.txt,size:13}},
      xaxis:{title:'x',range:[-1,10],color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      yaxis:{title:'Density',color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
      paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:s.txt},margin:{t:35,b:35,l:45,r:15}
    },{responsive:true});
  });
  const lay={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:s.txt},
    xaxis:{color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},
    yaxis:{color:s.ax,tickcolor:s.ax,gridcolor:s.gr,linecolor:s.ax,tickfont:{color:s.ax}},margin:{t:30,b:30,l:40,r:10}};
  Plotly.newPlot('loss-plot',[],{...lay,title:{text:'Loss',font:{color:s.txt,size:12}}},{responsive:true});
  Plotly.newPlot('discriminator-plot',[],{...lay,title:{text:'Discriminator',font:{color:s.txt,size:12}},yaxis:{...lay.yaxis,range:[0,1]}},{responsive:true});
}

function init() {
  if(G) G.dispose(); if(D) D.dispose();
  G=makeG(); D=makeD();
  gOpt=tf.train.adam(cfg.lr); dOpt=tf.train.adam(cfg.lr);
  genData(); ep=0; gL=[]; dL=[]; updDisp();
}

function reset() { training=false; document.getElementById('train-btn').textContent='Start Training'; init(); initPlots(); }

document.addEventListener('DOMContentLoaded',()=>{
  const wait=setInterval(()=>{
    if(typeof tf!=='undefined'){
      clearInterval(wait); init(); initPlots();

      document.getElementById('train-btn').onclick=async()=>{
        if(training){training=false;document.getElementById('train-btn').textContent='Start Training';}
        else{training=true;document.getElementById('train-btn').textContent='Stop';loop();}
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
        el.oninput=e=>{cfg[keys[i]]=+e.target.value;document.getElementById(id+'-display').textContent=e.target.value;};
        el.onchange=()=>{if(!training)reset();};
      });
    }
  },100);
});
</script>
