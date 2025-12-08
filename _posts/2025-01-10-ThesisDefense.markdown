---
layout: post
title:  "Now I am become Doctor"
date:   2025-01-10 18:00:00
categories: main
---

This is just a quick post to announce that I've successfully defended my Ph.D. thesis this past December 5th, titled 
***Guiding AI Attention for Driving and Creative Generation***. The final grade I obtained was of *Excel·lent* (A), which
allowed me to also obtain a [Cum Laude](https://www.uab.cat/web/study/phds/online-thesis-deposit/qualification-of-the-thesis-and-teseo-1345799090979.html).

This was a long road, but I'm glad that I took it. I'll be forever grateful to my advisor [Antonio López](https://www.cvc.uab.es/people/antonio/), 
especially for seeing the value in adding the creative usage of Neural Networks to different fields in Machine Learning. 
Likewise, a special thanks to the thesis examination panel [Prof. Jose María Armingol](https://lsi.uc3m.es/personal-jose-maria-armingol/), 
[José Manuel Álvarez](https://alvarezlopezjosem.github.io/), and [Fernando Vilariño](http://vi.cvc.uab.es/fernando-vilarino/).
Their feedback and questions were both reaffirming and encouraging to the work done during my Ph.D., as well as the 
work still to be done in the fields of autonomous driving and creative applications of ML to the arts.

All in all, a Ph.D. was a much different path than the one I expected [coming in](2019-09-23-Threads.markdown), in particular regarding
personal news: we had two kids with my wife Samantha, Santiago and Olivia, which on one hand made it harder to stick to
paper deadlines, but on the other made all the effort far more rewarding. Indeed, a Ph.D. is a hard path for interpersonal relationships,
but having my wife and kids kept me sane most of the time. As Joseph J. Rotman puts it in his dedication in 
*An Introduction to Algebraic Topology* (jokingly I hope):

> To my wife Marganit and my children Ella Rose and Daniel Adam without whom this book would have been completed two years earlier

<a name="OnTheThesisManuscript"></a>
# On the Thesis Manuscript

The main lesson I've learnt is that during your Ph.D., you will have many ideas that will lead you nowhere; others that 
cannot be pursued due to a lack of compute or knowledge or manpower/time. If you by some chance are thinking of doing a 
Ph.D., I must recommend you to both learn from *and* work with those surrounding you, as you can only get so far by yourself. 
Likewise, try to read more and do crazy things, as science is mostly asking and answering the *how*, with the *why* coming 
after discovering a phenomena and delving deep into it.

A key aspect here is to not only read scientific papers, which is a trap I see most students fall in: they only read the 
end product (the papers in prestigious conferences), without realizing that it's the culmination of a lot of iteration, work, and 
storytelling that were mostly burrowed not just from other scientific work, but from life in general: literature, visual arts, 
music, architecture, etc. Storytelling is especially important, as your audience needs to be sold on your idea in an effective 
matter. No amount of mathematical jargon will cure a poorly-written paper, so ***read, read, and read!***

You can find the recording of the thesis defense [here](https://www.cvc.uab.es/blog/2024/12/05/guiding-ai-attention-for-driving-and-creative-generation/). 
While the official link to access the PDF file of the thesis is [this one](https://www.tdx.cat/handle/10803/693252), it is under an embargo until December 2026 
due to unpublished work. Thus, I can only share the recording above and the accompanying slides here:

<div style="position: relative; width: 100%; height: 0; padding-top: 56.2500%;
 padding-bottom: 0; box-shadow: 0 2px 8px 0 rgba(63,69,81,0.16); margin-top: 1.6em; margin-bottom: 0.9em; overflow: hidden;
 border-radius: 8px; will-change: transform;">
  <iframe loading="lazy" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; border: none; padding: 0;margin: 0;"
    src="https://www.canva.com/design/DAGTqdmyOg8/ezrBsiY6pPaxuFamkRmt6A/view?embed" allowfullscreen="allowfullscreen" allow="fullscreen">
  </iframe>
</div>
<a href="https:&#x2F;&#x2F;www.canva.com&#x2F;design&#x2F;DAGTqdmyOg8&#x2F;ezrBsiY6pPaxuFamkRmt6A&#x2F;view?utm_content=DAGTqdmyOg8&amp;utm_campaign=designshare&amp;utm_medium=embeds&amp;utm_source=link" target="_blank" rel="noopener">Guiding AI Attention for Driving and Creative Generation - Thesis Defense</a> by Diego Porres

In any case, I quickly summarize the main chapters as follows, with Chapters 1 and 5 being the Introduction and 
Conclusions/Future Work, respectively. Note that each chapter is meant to be self-contained:

* **Chapter 2: AI as a Creative Collaborator**:
  This chapter consists of the following three publications:
  * [**_Discriminator Synthesis: On reusing the other half of Generative Adversarial Networks_**](https://arxiv.org/abs/2111.02175), [*NeurIPS 2021 Workshop on Machine Learning for Creativity and Design*](https://neuripscreativityworkshop.github.io/2021)
    * When training a Generative Adversarial Network (GAN), we mostly discard the Discriminator and focus on the Generator.
    We find that the Discriminator has learned some pretty unique patterns that we can generate unique artwork. We name
    this first approach *Discriminator Dreaming* in reference to [DeepDream](https://research.google/blog/inceptionism-going-deeper-into-neural-networks/), but this paper is mostly a call-to-action
    to use the Discriminator more in the creative process.
  * [**_At the Edge of a Generative Cultural Precipice_**](http://arxiv.org/abs/2406.08739), [*CVPR 2024 Fourth Workshop on Ethical Considerations in Creative applications of Computer Vision*](https://sites.google.com/view/cvfad2024/ec3v)
    * Many proponents of generative models assert that, in the near future, we won't need *real* data to train models,
    just *synthetic* or *generated* data. On the one hand, recent work has shown that models will quickly collapse if trained
    in this manner, with the only solution to avoid this is by using fresh new data or aggregate old data with new one. For this, we show that
    artists are already no longer sharing as much their art online for fear of others stealing it, which will only push
    models and *new*, incoming artists to collapse to specific styles. A true *cultural precipice*.
  * [**_Towards Kinetic Manipulation of the Latent Space_**](https://arxiv.org/abs/2409.09867), [*NeurIPS 2024 Creative AI Track*](https://neurips-creative-ai.github.io/2024/)
    * When interacting with a generative model such as a GAN, we have relinquished our interaction with it with just a GUI or
    a mouse and keyboard. We propose to use a cheap camera and pre-trained models (such as [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide)),
    to extract body keypoints that we can use to manipulate the latent vectors or parameters of the generative model. We
    tested this with StyleGAN2 and StyleGAN3, and plan to move to diffusion models (so long as the interaction is in real time).
* **Chapter 3: Guiding Attention in End-to-End Driving Models**:
  * [**_Guiding Attention in End-to-End Driving Models_**](https://ieeexplore.ieee.org/document/10588598), [*IEEE Intelligent Vehicles Symposium (IV) 2024*](https://ieee-iv.org/2024/)
    * What are the effects on an end-to-end driving model if we optimize its ***attention weights***? We find the following:
    the model's interpretability is increased, we increase the sample efficiency (needing less data to train), and we don't
    need to modify the original architecture. Indeed, we have found that we have a $\approx \times 4$ data efficiency when
    using this *Attention Loss* and the model learns to weakly segment the objects of interest.
* **Chapter 4: Attention Dichotomy: Towards Stabilizing End-to-End Autonomous Driving** 
  * This chapter deals with yet to be released work regarding updating the loss in Chapter 3 to handle individual classes.
  We do this by letting individual heads segment specific classes of interest, instead of doing a general *average* attention.
  Likewise, we analyze the metrics used to assess and compare the driving quality of autonomous driving models in the 
  [CARLA](https://carla.org/) simulator. Spoiler: the current metrics hide a lot of variance of the models, so some may appear to perform 
  better, but these improvements may not be as consistent as one thinks. We append a sample of a model trained with the
  CAT Loss here:

<div class="google-slides-container">
<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/BMt9r8Kj8Pk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<a name="FutureWork"></a>
## Future Work

**Chapter 5** summarized the conclusions of the thesis, as well as proposing future work:

* **A stronger correlation between validation loss and driving quality needs to be achieved**. This is high in my list 
because current ways to train end-to-end driving models are a bit insane: you need to validate each checkpoint not by using
a static dataset, but by deploying your model into a real car or into simulation. This means it must complete specific
routes and you can then assess which is better not by how much it adheres to the "ground truth" action, but by how many
of these routes it can complete, how many infractions it does, etc. Indeed, a static validation dataset would be cheap, but
the main difference is that [small errors may compound when deploying these models](https://arxiv.org/abs/1904.08980). As such, a lot of compute is 
necessary to both train and validate every checkpoint of the model correctly, which is not feasible for most researchers 
due to the computational cost, so we need to find a way to estimate the driving quality of a model without
*actually* deploying it.
* **Actions aren't symmetric, so why are loss functions symmetric?** Not all end-to-end driving models operate in the action
space. That is, most of them work with predicting the future waypoints and then another module is in charge of planning
the correct actions to take in order to follow this route. In our case, we directly predict the actions: say we normalize 
the action space $\mathcal{A}=[-1, 1]\times[-1,1]$, so that the steering angle goes from $-1$ (left) to $1$ (right) and 
the acceleration goes from $-1$ (brake) to $1$ (full throttle). The *Action Loss* we use is the L1 loss of the difference 
between predicted and ground truth action, meaning we give equal weight when the model predicts $-1$ instead of $1$ and 
vice versa. What I mean by the question above is that reality is *asymmetric*, as the consequences of accelerating when 
the correct action is to brake is not the same as the opposite, as the former may lead to a collision with another object
in an urban scenario. As such, we must either look at asymmetric losses (such as Quantile Regression Loss) or find a way
to make the loss function adaptive in function of the previous batch, i.e.: 
\[
  \tau_{\text{AQR}} = f \left( \mathbb{E}_{a\in\mathcal{B}} [a-\hat{a}] \right)
\]

with $\tau_{\text{AQR}}$ being the quantile to use for the next batch, $a$ the ground truth action, $\hat{a}$ the predicted
action, and $\mathcal{B}$ the batch of actions.
* **What should an end-to-end driving car look like?** This one has been something that has bugged me for a while now:
most work dealing with trust in autonomous vehicles has failed to take into account that these things are basically black
boxes from the perspective of third parties ***outside*** of the vehicle: if you're crossing a road and a car is coming, 
you can infer whether you can cross the road safely merely by looking at the driver. Without the driver, you simply cannot
know what's going to happen! Thus, interpretability of the model should also be seen from the POV of pedestrians, cyclists, 
and other drivers. Perhaps generative models will be useful here in order to synthesize visuals or 3D sounds for the 
specific actors that are influencing its current actions, aimed to let them know the model has seen them.

<a name="Artwork"></a>
## Artwork

Not surprisingly, I've managed to produce some artwork that has been accepted in various exhibitions and galleries. The main
ones became the cover of each chapter, namely:

* **_Threaded History_** (2020), exhibited in the [NeurIPS 2020 Workshop on Machine Learning for Creativity and Design
  Art Gallery](https://www.aiartonline.com/art-2020/diego-porres-2/)

<div class="google-slides-container">
<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/GzHKtcPTKR4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

* **_Regal Space_** (2021), exhibited in the [NeurIPS 2021 Workshop on Machine Learning for Creativity and Design
  Art Gallery](https://neuripscreativityworkshop.github.io/2021/)

<div class="google-slides-container">
<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/DNfocO1IOUE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

* **_Hidden Clergy_** (2022), exhibited in the [NeurIPS 2022 Workshop on Machine Learning for Creativity and Design
  Art Gallery](https://neuripscreativityworkshop.github.io/2022/)

![Hidden Clergy - Wikiart](/img/hidden_clergy/wikiart.png)

![Hidden Clergy - MetFaces](/img/hidden_clergy/metfaces.png)

* **_The Unknown_** (2023), exhibited in the [NeurIPS 2023 Workshop on Machine Learning for Creativity and Design
  Art Gallery](https://neuripscreativityworkshop.github.io/2023/) and the [CVPR 2024 AI Art Gallery](https://thecvf-art.com/project/diego-porres/)

<div class="google-slides-container">
<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/vn-Ih1tDex0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<a name="NextSteps"></a>
# Next Steps

I'll continue to be in the Computer Vision Center ([CVC](https://www.cvc.uab.es/)) where I'll join the [BERTHA](https://berthaproject.eu/)
project as a Postdoctoral Researcher. The main aim of this project is to develop a safer and more human-like autonomous 
vehicle. Specifically, we want to *imbue* certain characteristics that current models are missing, such as object 
permanence/memory, key object selection/attention, and data efficiency. I'll be working on the last two parts in particular,
as the Attention Loss we proposed in Chapter 3 of my thesis has these characteristics: by guiding *where* an end-to-end
driving model should look at, you can also let it know what objects are important and use less data to train them than
usual. We're 14 research centers in total, so expect interesting results from this project in the coming months, both in 
the form of datasets and new models (and of course research papers).

In the meantime, I'll be uploading here a lot of works, tricks, and small studies that I've learned throughout these past
five years of my Ph.D. journey. Some will be generally useful, others more geared towards specific fields such as 
autonomous driving and creative practices regarding AI, the arts, and how we interact with these models.

Cheers!

{% include disqus.html %}