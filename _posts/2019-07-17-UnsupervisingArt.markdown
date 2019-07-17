---
layout: post
title:  "On Unsupervising Art"
date:   2019-07-17 18:00:00
categories: main
image_sliders:
  - slider2
---

**tl;dr** I will rant about art and will mos tlikely talk about things I do not fully understand about it, but hey, this is *my* blog. At the end, I present how I've applied Machine Learning algorithms, specifically Unsupervised Learning algorithms such as [GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network), [$k-$means](https://en.wikipedia.org/wiki/K-means_clustering) and [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) to aid me in an ongoing art project.

---

# Intro

[Gatekeeping](https://www.reddit.com/r/gatekeeping/) art is, perhaps, one of the things I dislike the most, and one thing I am certainly guilty of doing before (as I am sure most artist have done once in their lives). In a nutshell, this mindset is as follows: nothing is *art* but what I create and enjoy, and as a corollary, there aren't quality artists nowadays, except for me and the artists whose work I enjoy. Indeed, I believe this not only applies to artists, but to almost everyone, as art is an integral part of our lives: form criticizing nowadays music, to not enjoying superhero movies or even looking down at the No. 1 New York Times bestselling book.

Of course, there is a counterpoint to this. [Mario Vargas Llosa](https://www.nobelprize.org/prizes/literature/2010/vargas_llosa/biographical/) writes in his book [Notes on the Death of Culture](https://www.amazon.com/Notes-Death-Culture-Spectacle-Society/dp/1250094747 "I encourage you to buy it"):

> La literatura *light*, como el cine *light* y el arte *light*, da la impresión cómoda al lector y al espectador de ser culto, revolucionario, moderno y de estar a la vanguardia, con un mínimo esfuerzo intelectual.

In other words, culture and art has been shifting towards a *light* verison, one in which works of art aren't meant to be thought-provoking or profound, their only goal is to entertain. In turn, this will forces artists to never leave the known confort zone of art that sells. Indeed, this is notorious in music and cinema, where very few writers venture outside what the studios and producers dictate them to do. But is it necessarily a bad thing?

While it is true that this happens, I think that the only difference between art nowadays and classical art is the sheer number of artists and art being created nowadays. Therefore, we are bound to see repetitveness, the same movie tropes and the same lyrics/beats in music, the same clichés in novels, but this does not mean that you cannot find *quality* art, you just have to look harder to distinguish it from the noise (of course, this is all still very subjective in my opinion).

Thus, the advent of computers will only aid in further expanding this rift separating what many percieve as good or bad art, even what is and isn't art. Indeed, nowadays you don't need a [physical canvas](https://processing.org/) or [musical instruments](https://magenta.tensorflow.org/), or even spend years studying the human form in order to [create your own runway/clothes](https://robbiebarrat.github.io/oth/bale.html). Of course, there have been some controversial works on this subject, perhaps moreso how they are reported by the media (which we will explore further on), but the main point is that computers, machines, and software are here to stay, and will most definitely influence every aspect of our lives, art included. 

# The Future of Art 

But what exactly lies in the future of art? It is of course hard to tell with certainty, but one can speculate and perhaps zone in . In the excellent series [The Future of Art](https://www.artsy.net/series/artsy-editors-future-art/) by [Artsy](https://www.artsy.net/), we are given the perspective of what the future entails for art from the point of view of different members of the art community. The latest video by [Simon Denny](http://simondenny.net/) delves into the [role of technology](https://www.artsy.net/series/artsy-editors-future-art/artsy-editorial-future-art-simon-denny) , and how the curating part of art will most likely be democratized, to much dismay to Mario Vargas Llosa. [Carrie Mae Weems](http://carriemaeweems.net/) sees the positive light of this, as [this democratization of art will give voice to those who haven't been able to participate](https://www.artsy.net/series/artsy-editors-future-art/artsy-editors-future-art-carrie-mae-weems) in something as elite (even elitist) as art. Specifically, marginalized societies as well as marginalized people such as women of color, and she wishes only to be able to be able to see this in the coming years.

[Marcel Dzama](https://www.davidzwirner.com/artists/marcel-dzama) argues that [classical forms of art such as painting and drawing won't disappear](https://www.artsy.net/series/artsy-editors-future-art/artsy-editors-future-art-marcel-dzama), and I agree: while the newest technologies create exciting, perhaps refreshign works of art, they won't compete with the old ways, only complement them as many other forms of art have done in the past. [Elizabeth Diller](https://www.paglen.com/) brings forth the [impact of time in works of art](https://www.artsy.net/series/artsy-editors-future-art/artsy-editors-future-art-elizabeth-diller): while some ideas can make sense in the moment they are thought, when they are actually executed (such as architecture), those ideas might not even make sense at all anymore. Indeed, any work of art is fixed in time, but their meaning is constantly evolving, both for the artist as well as the spectator. Finally, [Trevor Paglen](https://www.paglen.com/) [agrees with all these points](https://www.artsy.net/series/artsy-editors-future-art/artsy-editors-future-art-trevor-paglen), in that the art of the future won't be radically different than the art in the present and past. I would add that the thing that would most likely change is the medium of how the artist communicates his or her idea to the public, and that medium is being disrupted by computers and computer software.

We do, however, have a potential bleak future in art if we rely too much in predictive models. Ashley Rose writes that [the landscape of music](https://www.entrepreneur.com/article/327781) has been transformed already by Machine Learning, and that music labels have noticed this. While the first two points mentioned by the author have some merit (one more than the other), it is the latter one that truly worries me. Indeed, it is possible to identify the next musical star, but we have to remmeber that music, perhaps moreso than any other art form, is an ever changing landscape, and this is most reflected by the [popular music genres each decade](http://www.thepeoplehistory.com/music.html). 

What many executives fail to understand is that ML models can be used to predict next top hits, but they do so only by using past data. It is not an *AI*, it is merely a predictive model that mimics something that is intelligent, as [Prof. Michael Jordan](https://people.eecs.berkeley.edu/~jordan/) (not the basketball player) [eloquently puts it](https://youtu.be/4inIBmY8dQI) (if you don't have the time, the first couple of minutes are worth it and always in the back of my mind whenever I read a news article about *AI*). As such, while some trends may be discovered, there will be a lot of bias introduced into the scenes of music production, as if more bias was warranted.

The world of cinema is no stranger to this. The Verge reports that many startups are offering [*AI* producers](https://www.theverge.com/2019/5/28/18637135/hollywood-ai-film-decision-script-analysis-data-machine-learning), skimming through all the available data in order to obtain the highest grossing movie possible. Changing actors and seeing how this will affect the movie performance in different markets seem like the best playground for the moviemakers to invest their money in, but to me this technology is frivolous at best. Since there aren't many movies produced per year, there is not much historical data to truly create an unbiased predictive model, and so movie producers might end up greenlighting the wrong movies, or even simply remaking old classics (perhaps they have adopted these technologies after all).

Perhaps I should emphasize that, whilst the predictive models are not nigh imperfect, they are a very useful tool. From [recommending music](https://medium.com/s/story/spotifys-discover-weekly-how-machine-learning-finds-your-new-music-19a41ab76efe), to [recommending the next show to binge watch](https://uxplanet.org/netflix-binging-on-the-algorithm-a3a74a6c1f59), I am open for these tools to enhance the audience experience, not only on these scenarios but on other forms of art as well, such as the plastic arts. However, it must be understood and emphasized that they are a tool only, and it is up to us to both use them wisely and in a meaningful way.

# Generative Art

Generative Art warrants another blog post by itself (one I am sure I will make sometime in the future), but is a tool that I believe will open the doors to many marginalized communities. As mentioned earlier, [Processing](https://processing.org/) lets you generate works of art without the need to have a physical canvas, and that is a huge step in the right direction. However, it is only one step, and I hope there are many more to take in the future.

Generative Design comes hand in hand with generative art, so it is worthwile to research what is being done and said in this field. Is Generative Design being used at all, and if so, is it worthwile or just a hype that is bound to die given the high bar that the human mind has for designed objects?

Rain Noe asks the question of where, aesthetically meaning, does [generative design belongs to](https://www.core77.com/posts/89318/Where-Does-Generative-Design-Belong-Designers-Must-Decide). To him, generative design's strength lays not in the aesthetic presentation of it, but on the cost and material savings that should happen behind close courtain, where the courtain should be created by an actual human designer. The teams of Autodesk and Aribus [seem to agree](https://www.autodesk.com/customer-stories/airbus), as they have designed lattice structures for the partition frames in the Airbus concept plane that is both lightweight and strong, basically saving valuable resources, but still tried to keep this design hidden from the passengers somewhat.

<div class="imgcap">
<img src="https://user-images.githubusercontent.com/24496178/61199133-09236600-a69a-11e9-8bae-404eca3a2e95.png" alt="Rims of a car">
<div class="container"><p><b>Two rims found in Rain Noe's article, one generated and the other desgined by a human. Which would you prefer?</b></p></div>
</div>

There is nothing wrong with this stance, but I believe that more can be done with Generative Design than simply limiting it to *small* objects. What if we let a generative algorithm design, with the appropriate constraints, the highways between cities or even a whole city layout? We don't actually need to hold the object being designed with our own hands in order to appreciate its beauty or usefullness, so perhaps we need to set this paradigm loose. 

As a final note, I feel that there is frustration in Generative Art and Design on how this field is being reported, especially on the topic of who is the actual artist (the programmer or the machine), or even if it is [more appealing](https://hyperallergic.com/391059/humans-prefer-computer-generated-paintings-to-those-at-art-basel/) than human generated art. This is folly, but only if nothing is known about this specific area, or perhaps there is no interest in learning. Time will tell whether or not the effort was worth it, but my prediction will be that it was.

## Machine Learning in Art

Machine Learning has certainly forever changed the panorama of every field it has been introduced to, and art is no exception. Perhaps the ones I enjoy the most, besides image generation, is text and character generation. Projects such as [textgenrnn](https://github.com/minimaxir/textgenrnn) and [kanji-rnn](http://otoro.net/kanji-rnn/) show me the power and potential that language has and how we can leverage Machine Learning to let more and more cultures to express themselves with these tools.

<div class="imgcap">
<img src="https://user-images.githubusercontent.com/24496178/61256685-d83f4180-a72a-11e9-8245-cc83d3f00dd4.png" alt="kanji-rnn">
<div class="container"><p><b>Fake kanji created by kanji-rnn. The model predicts what the next stroke of the *pen* will be, and the results are different as the user advances with their drawing.</b></p></div>
</div>

While these tools are classical Supervised Learning techniques, I believe a greater potential resides in Unsupervised Learning algorithms. This brings me to the main title of this blog post (a clever wordplay if you will): we need to democratize the techniques found in Unsupervised Learning amongst the art community, nay, amongst the general audience, and quit trying to form clubs where only the elites can reside. Otherwise, we are back in the Renaissance, where certainly excellent art was being produced, but only by those who had both the talent and access to the right tools. Hence, we must be open to start *Unsupervising Art*.

Perhaps, in order to illustrate what can be achieved with these techniques, I will tell the tale of one of my latest endeavours, but for that I will need to delve a bit into a short summary of what techniques have been and are being used in the artworld.

## GANs

We begin our mini summary with Generative Adversarial Networks or [GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network), which are perhaps one of the most applied machine larning paradigms to the world of art. We will look at them from the point of view of images, but it can be applied wherever you have a distribution of data that you wish to mimic. The idea is as follows: 

  * There are two competing neural networks, a **generator** $G$ and a **discriminator** $D$, and they are playing a simple zero-sum game.
  * The generator must produce images $\mathbf{x} = G(\mathbf{z}; \mathbf{\theta}_{(g)})$ and the discriminator will emit a probability $D(\mathbf{x}; \mathbf{\theta}_{(d)})$ which will be high (close to $1$) if it believes $\mathbf{x}$ is a real image from our training set or low (close to $0$) if it believes it to be fake. 
  * We will represent both the generator and discriminator as neural networks, hence the parameters $\mathbf{\theta^{(g)}}$ and $\mathbf{\theta^{(d)}}$, but we will use either notation interchangeably. 

This game will be played until the fake images are indistinguishable from the real ones, or in other words, when $D(\mathbf{x}) = 1/2$ always. The fake images will be generated via $G(\mathbf{z})$, where $\mathbf{z}$ will be a random vector drawn from a **latent space** (a vector space) of representations where any point in said space can be mapped to a realistic-looking image $\mathbf{x}$.

Since this is a zero-sum game, we define $v (\mathbf{\theta}_{(g)}, \mathbf{\theta}_{(d)})$ as the reward we give to the discriminator for correctly classifying the fake data from the real one, while we give $-v (\mathbf{\theta}_{(g)}, \mathbf{\theta}_{(d)})$ as a reward to the generator. At convergence, we will have:

$$ G^{\star} = \arg \min_{G} \max_{D} v(G, D)$$

Thus, in function of the parameters of the respective neural networks $\mathbf{\theta^{(g)}}$ and $\mathbf{\theta^{(d)}}$, $v$ should be:

$$ v (\mathbf{\theta}_{(g)}, \mathbf{\theta}_{(d)}) = \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}(\mathbf{x})} \log{(D(\mathbf{x}))} + \mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}(\mathbf{z})} \log{(1-D(G(\mathbf{z})))} $$

This equation can be read as follows: 

### Latent Fabrics

[Latent Fabrics](http://www.aiartonline.com/community/diego-porres/ "Why yes, I will reference myself") began as a project to understand what constitutes a [huipil](https://en.wikipedia.org/wiki/Huipil): is there something more profound than the general shape and colors? Are the patterns indicative of something that links the cultures within Mexico and Central America, even if there is no shared history? As such, the first idea I had was to simply build a classifier, born from the [constant plagiarism](https://lawstreetmedia.com/blogs/ip-copyright/maya-women-fight-protect-indigenous-textiles-appropriation/) that indigenous textile design [suffers from](https://www.huffpost.com/entry/mexico-prevents-indigenous-designs-from-being-culturally-appropriated-again_n_56e87879e4b0b25c9183afc4) (even [outside of the textile world](https://mexiconewsdaily.com/news/mexico-accuses-louis-vuitton-of-copying-indigenous-designs/)).

Indeed, it is a pressing issue where Machine Learning can (and should) be part of the solution. However, the difficulty in using a classifier is the sheer amount of data needed for it to train and converge, as well as to have each individual image correctly labeled. As such, this project shifted to try to generate new huipils without the need to label them, and what better candidate for the job than GANs?

For the data, I used Karen Elwell's [Flickr collection](https://github.com/carpedm20/DCGAN-tensorflow) (with permission) and for the code implementation I used Taehoon Jim's implementation of [DCGAN](https://arxiv.org/pdf/1511.06434.pdf "Deep Convolutional GANs") in [tensorflow](https://github.com/carpedm20/DCGAN-tensorflow). While I started with over 2000 photographs of huipils, many where duplicates, and some others could not be used. As such, I ended up with 641 pictures of huipils, specifically from Mexico and Guatemala. Nowadays, Karen's collection has far more pictures, and I hope to both use the new pictures of huipils, as well as to combine them with other sources such as the [Museo Ixchel's](https://museoixchel.org/) and the [Minneapolis Institute of Art (MIA)](https://new.artsmia.org/), where I have found on the latter [81 new huipils to use](https://collections.artsmia.org/search/huipil/filters/image:valid?size=130).

The following are the end results: after circa 2 days of the networks trainig on my [GeForce 980M](https://www.geforce.com/hardware/notebook-gpus/geforce-gtx-980m "Hey, it does the job"), I obtained 221 generated huipils of size $256 \times 256$ pixels and on another separte run of circa 2 days I obtained 330 generated huipils of size $64 \times 64$ pixels. 

<div class="imgcap">
<img src="https://live.staticflickr.com/4858/32403944098_a649c62e5e_o.png" alt="256x256">
<div class="container"><p><b>The 221 generated huipils of size $256 \times 256$ pixels arranged randomly. Pardon the sheer size of the file.</b></p></div>
</div>

<div class="imgcap">
<img src="https://user-images.githubusercontent.com/24496178/61109804-0549d680-a443-11e9-9e0c-3a7685aee4e9.png" alt="64x64">
<div class="container"><p><b>The 330 generated huipils of size $64\times 64$ pixels arranged randomly.</b></p></div>
</div>

All in all, this was the work of weeks, as some hyperparameters were tuned, some pictures were modified to better work for the dataset, and also there were lots of failures. Of course, there are far more huipils that could be generated, but these where the ones that, to my criteria, where most pleasing to the eye. Indeed, this is after all the job of the artist, otherwise we would let randomness govern us, and the human mind gets easily bored of the random world, [unless we get to participate in its creation process and curate it](http://www.random-art.org/online/). 

Now that we have these generated huipils, what could be the next step? The random arrangement can only get us so far, so why not separate the huipils by likeness to each other? For this task, there is another algorithm that can come in handy...


### $k$-means

The final step (thus far) has been to apply the well-known unsupervised learning algirthm [$k$-means clustering](https://en.wikipedia.org/wiki/K-means_clustering). As with any Unsupervised Learning algirhtm, there is no correct answer, so the different results I would obtain would depend largely on how I  



```python
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from PIL import Image
import glob
import os

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
```



```python
filepath = os.getcwd() + "\\images_to_merge"
image_list = []

for filename in glob.glob(filepath+'\\*.png'):
    im = np.float64(Image.open(filename))
    image_list.append(im)
    
image_list = np.array(image_list)
num_images = len(image_list)
```



```python
>>> image_list.shape
(221, 256, 256, 3)
```



```python
scaler = StandardScaler()

scaler.fit(image_list.reshape(num_images, -1))

X = scaler.transform(image_list.reshape(num_images, -1))
```



```python
sil_score = []

for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_
    sil_score.append(metrics.silhouette_score(X, labels, metric='euclidean'))
```



```python
plt.figure(figsize=(12,9))
plt.plot([k for k in range(2, 20)], sil_score)
plt.xlabel('Numero de clusters')
plt.ylabel('Score de silueta')
plt.xticks([k for k in range(2, 20)])
plt.show()
```

<div class="imgcap">
<img src="https://user-images.githubusercontent.com/24496178/61026073-b2531f00-a36f-11e9-8101-d42fa27da507.png" alt="Silhouette score">
<div class="container"><p><b>Silhouette score for different values of k.</b></p></div>
</div>


```python
kmeans = KMeans(n_clusters=7, random_state=42).fit(X)
labels = kmeans.labels_
```

```python
>>> labels
array([2, 3, 0, 1, 0, 2, 1, 2, 4, 4, 3, 5, 2, 0, 1, 6, 0, 0, 3, 1, 6, 0,
       4, 5, 6, 2, 0, 5, 6, 6, 0, 1, 6, 5, 0, 1, 4, 6, 5, 1, 3, 3, 2, 5,
       5, 5, 2, 3, 5, 6, 0, 6, 0, 1, 6, 0, 5, 5, 6, 4, 5, 2, 6, 5, 4, 0,
       5, 4, 4, 0, 4, 5, 6, 4, 6, 1, 4, 0, 6, 2, 2, 6, 4, 2, 2, 5, 6, 5,
       6, 1, 5, 4, 3, 5, 5, 3, 4, 3, 3, 4, 3, 4, 0, 5, 4, 5, 3, 1, 5, 5,
       4, 5, 6, 1, 2, 5, 0, 5, 6, 6, 4, 3, 3, 4, 5, 6, 6, 0, 1, 3, 4, 4,
       0, 4, 5, 5, 2, 4, 6, 4, 3, 0, 6, 3, 4, 2, 3, 3, 6, 6, 1, 0, 1, 5,
       6, 0, 6, 6, 0, 6, 6, 4, 2, 4, 3, 1, 2, 3, 1, 5, 5, 0, 6, 0, 1, 5,
       0, 2, 5, 3, 5, 0, 4, 2, 1, 5, 4, 0, 6, 6, 4, 6, 3, 5, 4, 4, 3, 4,
       6, 5, 4, 6, 5, 4, 1, 6, 2, 4, 5, 6, 6, 5, 4, 5, 6, 5, 2, 1, 0, 5,
       5])
```



```python
>>> from collections import Counter
>>> Counter(labels)
Counter({0: 29, 1: 21, 2: 21, 3: 24, 4: 38, 5: 46, 6: 42})
```

### PCA

Principal Component Analysis or [PCA)(https://en.wikipedia.org/wiki/Principal_component_analysis)

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
projected = pca.fit_transform(X)
```



```python
plt.figure(figsize=(12,9))
plt.scatter(projected[:, 0], projected[:, 1], c=labels, 
            edgecolor='none', alpha=0.9, cmap=plt.cm.get_cmap('Spectral', 7))
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.colorbar()
plt.show()
```


<div class="imgcap">
<img src="https://user-images.githubusercontent.com/24496178/61026876-ea5b6180-a371-11e9-8e54-6b1a4ed7f838.png" alt="PC plotting">
<div class="container"><p><b>Plotting each generated huipil in a reduced dimension space using Principal Components. Note there are natural clusters appearing, but others are more intertwined.</b></p></div>
</div>



<div class="imgcap">
<img src="https://user-images.githubusercontent.com/24496178/61085752-4368db00-a3ee-11e9-94a2-487abaa9dc89.png" alt="PC 3D plotting">
<div class="container"><p><b>Same as before, but now we add a third dimension.</b></p></div>
</div>



<div class="imgcap">
<img src="https://user-images.githubusercontent.com/24496178/61087744-c50f3780-a3f3-11e9-908f-6b78f546f744.png" alt="EVR plot">
<div class="container"><p><b>The explained variance in function of the number of principal components. There is a steep increase at the beginning, so we know that the first components will hold most of the information.</b></p></div>
</div>

We see that, to explain $70\%$ of the variance, we need 20 principal components:

```python
>>> next(i for i, v in enumerate(np.cumsum(pca.explained_variance_ratio_)) if v>0.7)
20
```

This might seem a lot



#### Separating the images

Before moving them with `Python`, we first create the new directories. If you're using a `Jupyter` notebook, you can also run this code by adding a `!` at the beginning of the line of code:

```
>>> FOR /l %x in (0, 1, 6) DO mkdir .\\Project_name\\%x
```

We basically create 7 directoriess/folders with the names of the grupos we have created with the $k$-means algorithm. Now we just proceed to copy, into each folder, the corresponding huipil using the `labels` array:

```python
import shutil

for folders, subfolders, filename in os.walk(filepath):
    i=0
    while i<num_images:
        shutil.copy(filepath+'\\'+filename[i], os.getcwd()+'\\Project_name\\'+str(labels[i]))
        i+=1
```

We will then end up with 7 different folders, each containing *similar* generated huipils (to the algorithm's criteria). Afterwards, I decided to arrange the images of each folder/label in canvases of $11\times11$ inches, particularly since they were commissioned by my brother. These are the final results, with the names indicating the label of the image (if there were too many images and they didn't fit, I separated into two):

{% include slider.html selector="slider2" %}

#### Next Steps

It is not so obvious what should be the next steps for this project, but I can at least enumerate some. These are, in no particular order:

  * [Remove the checkerboard artefacts](https://distill.pub/2016/deconv-checkerboard/) due to the Deconvolution (this can be more easily seen in the $256\times 256$ generated huipils).
  * Increase the size of the generated huipils, hopefully up to $1024 \times 1024$, should my hardware allow it. This is because I wish to see more patterns emerge within the generated huipils.
  * Train the models for more epochs and with a larger dataset, so basically continue obtaining more images and get better hardware and/or simply train for longer in my laptop. Likewise, clean the original dataset, as I have found both inconsistencies as well as downright bad pictures that slipped my original check.
  * Traverse the latent space that weaves all the huipils together (the *Latent Fabrics* if you will).
  * Explore the possiblity of combining my work with Processing and add another type of generative art.
  * Finally, ponder how to present this work: should digital art remain digital, or does it make sense to bring it to the physical world via another technique, such as actual weaving?
  
  
I hope you have enjoyed this blog post. Please leave any comment below!

{% include disqus.html %}
