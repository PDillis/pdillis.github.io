---
layout: post
title:  "On Unsupervising Art"
date:   2019-07-12 18:00:00
categories: main

---

[Gatekeeping](https://www.reddit.com/r/gatekeeping/) art is, perhaps, one of the things I dislike the most, and one thing I am certainly guilty of doing before (as I am sure most artist have done once in their lives). In a nutshell, this mindset is as follows: nothing is *art* but what I create and enjoy, and as a corollary, there aren't quality artists nowadays, except for me and the artists whose work I enjoy. Indeed, I believe this not only applies to artists, but to almost everyone, as art is an integral part of our lives: form criticizing nowadays music, to not enjoying superhero movies or even looking down at the No. 1 New York Times bestselling book.

Of course, there is a counterpoint to this. [Mario Vargas Llosa](https://www.nobelprize.org/prizes/literature/2010/vargas_llosa/biographical/) writes in his book [Notes on the Death of Culture](https://www.amazon.com/Notes-Death-Culture-Spectacle-Society/dp/1250094747):

> La literatura *light*, como el cine *light* y el arte *light*, da la impresión cómoda al lector y al espectador de ser culto, revolucionario, moderno y de estar a la vanguardia, con un mínimo esfuerzo intelectual.

In other words, culture and art has been shifting towards a *light* verison, one in which works of art aren't meant to be thought-provoking or profound, their only goal is to entertain. In turn, this will forces artists to never leave the known confort zone of art that sells. Indeed, this is notorious in music and cinema, where very few writers venture outside what the studios and producers dictate them to do. But is it necessarily a bad thing?

While it is true that this happens, I think that the only difference between art nowadays and classical art is that the sheer number of artists and generated art nowadays is vastly greater in numbers than before. Therefore, we are bound to see repetitveness, the same movie tropes and the same lyrics/beats in music, but this does not mean that you cannot find *quality* art, you just have to look harder to distinguish it from the noise (of course, this is all still very subjective in my opinion).

Thus, the advent of computers will only aid in further expanding this rift separating what many percieve as good or bad art, even what is and isn't art. Of course, there have been some controversial works on this subject, perhaps moreso how they are reported by the media (which we will explore further on), but the main point is that computers, machines, and software are here to stay, and will most definitely influence every aspect of our lives, art included.

# The Future of Art 

But what exactly lies in the futur of art? It is of course hard to tell with certainty, but one can speculate. In the excellent series by [Artsy](https://www.artsy.net/), [The Future of Art](https://www.artsy.net/series/artsy-editors-future-art/) gives us the perspective of what the future entails for art from different artists. [Simon Denny](https://www.artsy.net/series/artsy-editors-future-art/artsy-editorial-future-art-simon-denny) 

Ashley Rose writes that the landscape of [music](https://www.entrepreneur.com/article/327781) has been transformed already by Machine Learning, and that music labels have noticed this.

# Generative Art



Rain Noe asks the question of where, aesthetically meaning, does [generative design belongs to](https://www.core77.com/posts/89318/Where-Does-Generative-Design-Belong-Designers-Must-Decide). To him, generative design's strength lays not in the aesthetic presentation of it, but on the cost and material savings that should happen behind close courtain, where the courtain should be created by an actual human designer. Of course, this is from the point of view of 

[perhaps moreso how they are reported](https://hyperallergic.com/391059/humans-prefer-computer-generated-paintings-to-those-at-art-basel/)

## Unsupervised Learning in Art




## GANs

$$ v (\mathbf{\theta}^{(g)}, \mathbf{\theta}^{(d)}) = \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}} \log{(d(\mathbf{x}))} + \mathbb{E}_{\mathbf{x}\sim p_{\text{model}}} \log{(1-d(\mathbf{x}))} $$


### Latent Fabrics

[Latent Fabrics](http://www.aiartonline.com/community/diego-porres/) 


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




