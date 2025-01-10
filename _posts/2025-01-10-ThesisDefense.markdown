---
layout: post
title:  "Now I am become Doctor"
date:   2025-01-10 18:00:00
categories: main
---

This is just a quick post to announce that I've successfully defended my Ph.D. thesis! This was a long road, but I'm 
glad that I took it. I'll be forever grateful to my advisor [Antonio López](https://www.cvc.uab.es/people/antonio/), 
especially for seeing the value in adding the creative usage of Neural Networks to different fields in Machine Learning. 
This was a much different path than the one I expected [coming in](2019-09-23-Threads.markdown), in particular regarding
personal news (e.g., we had two kids with my wife, Santiago and Olivia). A Ph.D. is a hard path for interpersonal relationships,
but they kept me sane most of the time. As Joseph J. Rotman (jokingly I hope) puts it in his dedication in 
*An Introduction to Algebraic Topology*:

> To my wife Marganit and my children Ella Rose and Daniel Adam without whom this book would have been completed two years earlier

There are many ideas that you have that lead nowhere, and many ideas that you cannot pursue due to a lack of compute,
specific knowledge, or manpower/hours available. If you by some chance are thinking of doing a Ph.D., I must recommend you
to work and learn from those surrounding you, as you can only get so far by yourself. Likewise, try to read more and do
crazy things, as science is mostly asking and answering the *how*, not the *why*. A key aspect here is to not only read
scientific papers, which is a trap I see most students fall in: they only read the end product (a scientific paper), without
realizing that it's the culmination of a lot of iteration, work, and mainly storytelling. The latter is especially important,
as the reader/your audience needs to be sold on your story/idea in an effective matter. No amount of mathematical jargon
will cure a badly-written paper, so read, read, and read!

You can find the recording of the thesis defense [here](https://www.cvc.uab.es/blog/2024/12/05/guiding-ai-attention-for-driving-and-creative-generation/), 
assuming the communications department has already released it. While the official link to access the PDF file of the
thesis is [this one](https://www.tdx.cat/handle/10803/693252), it is under an embargo until December 2026 due to unpublished work, 
so I hope the summary of the thesis in the form of the presentation above is sufficient. If it isn't, the chapters mainly 
consisted of the following (with Chapter 1 being the Introduction and Chapter 5 the Conclusions and Future Work):

* **Chapter 2**:
  * [**_Discriminator Synthesis: On reusing the other half of Generative Adversarial Networks_**](https://arxiv.org/abs/2111.02175), [*NeurIPS 2021 Workshop on Machine Learning for Creativity and Design*](https://neuripscreativityworkshop.github.io/2021)
  * [**_At the Edge of a Generative Cultural Precipice_**](http://arxiv.org/abs/2406.08739), [*CVPR 2024 Fourth Workshop on Ethical Considerations in Creative applications of Computer Vision*](https://sites.google.com/view/cvfad2024/ec3v)
  * [**_Towards Kinetic Manipulation of the Latent Space_**](https://arxiv.org/abs/2409.09867), [*NeurIPS 2024 Creative AI Track*](https://neurips-creative-ai.github.io/2024/)
* **Chapter 3**:
  * [**_Guiding Attention in End-to-End Driving Models_**](https://ieeexplore.ieee.org/document/10588598), [*IEEE Intelligent Vehicles Symposium (IV) 2024*](https://ieee-iv.org/2024/)
* **Chapter 4**: deals with yet to be released work regarding updating the loss in Chapter 3 to handle individual classes,
    and analyzing the metrics used to assess and compare the driving quality of autonomous driving models in the 
    [CARLA](https://carla.org/) simulator. Spoiler: the current metrics hide a lot of variance of the models, so they may
    appear to perform better, but they actually aren't that reliable or consistent in their driving.

I'll continue to be in the Computer Vision Center ([CVC](https://www.cvc.uab.es/)) where I'll join the [BERTHA](https://berthaproject.eu/)
project as a Postdoctoral Researcher. The main aim of this project is to develop a safer and more human-like autonomous 
vehicle. Specifically, we want to *imbue* certain characteristics that current models are missing, such as object 
permanence/memory, key object selection/attention, and data efficiency. I'll be working on the last two parts in particular,
as the Attention Loss we proposed in Chapter 3 of my thesis has these characteristics: by guiding *where* an end-to-end
driving model should look at, you can also let it know what objects are important and use less data to train them than
usual. We're 14 research centers in total, so expect interesting results from this project in the coming months, both in 
the form of datasets and new models.

In the meantime, I'll be updating on a lot of works, tricks, and small studies that I've learned throughout these past
five years of my Ph.D. journey. Some will be generally useful, others more geared towards specific fields such as 
autonomous driving and creative practices regarding AI, the arts, and how we interact with these models.

Cheers!

{% include disqus.html %}