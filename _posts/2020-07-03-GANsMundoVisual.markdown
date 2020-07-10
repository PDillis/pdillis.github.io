---
layout: post
title:  "GANs - En el Mundo Visual"
date:   2020-07-03 12:00:00
categories: main

---

<link rel="stylesheet" href="/assets/css/BeerSlider.css">

<a name="TechCommunityDay"></a>
# Tech Community Day

Hace unas semanas me pidieron si podría dar una charla acerca de un tema de Inteligencia Artificial en el [Tech Community Day](https://techcommunityday.com/). Es la primera vez que he dado una charla de este tipo, tanto en línea como a un público en general, por lo que claramente estaba nervioso. Sin embargo, estaba aún más emocionado de poder compartir un poco de lo que he estado realizando durante mi doctorado en el [CVC](http://www.cvc.uab.es/) desde finales del año pasado, por lo que acepté darla. Asimismo, estaba honrado de que los organizadores me hayan considerado para dar una charla en este evento, por lo que les estaré eternamente agradecido.

El tema que he escogido fue de [GANs](https://blog.diegoporres.com/main/2019/07/17/UnsupervisingArt/) y decidí realizarla en español, ya que la información que se puede conseguir del tema en inglés abunda, mas no en español y no se diga en otros idiomas. El título de la charla fue **GANs - En el Mundo Visual** y pueden encontrar la grabación de la misma en [YouTube](https://youtu.be/DdD39y8rJQ8).

Si bien condensar tanta información en una charla de 1 hora es difícil, tuve que dejar fuera ciertos puntos y aplicaciones interesantes de las GANs. Al mismo tiempo, esto me permitió ver lo que el público le interesa saber más (para la próxima vez):

* ¿Qué aplicaciones concretas se ha realizado de las GANs en medicina, arquitectura o diseño? ¿Qué tal en otras áreas que no se han explorado mucho?
* ¿Cómo podemos saber que se tiene que detener el entrenamiento de una GAN?
    * Por aparte, esto lleva a explicar los problemas típicos que se enfrenta uno al entrenar una GAN (colapso, colapso total o inclusive no converger, por ejemplo).
* ¿Puede una GAN llegar a [sobreajustar](https://developers.google.com/machine-learning/glossary#sobreajuste-overfitting) (overfit) los datos de un dataset? ¿Cuál de las dos redes hace esto?
* ¿Debemos de hacer públicos todos los modelos generativos, incluidos los hechos con una GAN?

Sobre el último punto, mi respuesta al momento sería ***No***, que es justamente lo que he querido comunicar con los modelos que he entrenado para la [generación de huipiles](https://blog.diegoporres.com/main/2019/09/23/Threads/): compartir dichos modelos puede llevar fácilmente al abuso de la recreación o apropiación de los diseños y patrones de los huipiles, por lo que no los he hecho públicos, aunque me lo han pedido en [varias ocasiones](https://twitter.com/PDillis/status/1270365318599278593?s=20). Sin embargo, si compartir los modelos reinicia un interés en el tema de los tejidos (más allá de la apropiación de los mismos), quizá valga la pena compartirlos. Claramente, siempre y cuando se tengan algunas medidas precautivas para no afectar a las comunidades de donde son originarios los tejidos, pero aún no lo tengo del todo claro.

En concreto, las diapositivas que he utilizado se encuentran a continuación (editadas para no incluir los archivos originales de algunas piezas de arte):

<div class="google-slides-container">
<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTGilWQpywgIhU1kCfir3zZwusptKPkvVYPdH1Qdga4hF_6Sz38gZerCVchykZHZqD9MzplXZWWNm5H/embed?start=false&loop=false&delayms=5000" frameborder="0" width="800" height="466" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
</div>

<a name="TransferenciaEstilo"></a>
# Transferencia de Estilo

Respecto al ultimo proyecto mencionado en la presentación y diapositivas donde he usando [StyleGAN2](https://github.com/NVlabs/stylegan2), [*Threads*](https://www.youtube.com/watch?v=t9fv4AAt6lw), podemos ver el resultado de mezclar a los vectores latentes generados con un listado de semillas específicas. Más concretamente, las semillas que han generado las imágenes de la primera columna son las **semillas fuente**, mientras que las semillas de que han generado la primera fila son las **semillas destino**.

En el [código de StyleGAN2 que uso](https://github.com/PDillis/stylegan2-fun), especificamos a estos como: `--row-seeds=17,32,78,300` y `--col-seeds=23,50,200,512`, respectivamente. Asimismo, el estilo que más me ha gustado traducir de una imagen a otra es el estilo **fino**, es decir, solamente copiamos los detalles finos (colores) de las semillas destino a las semillas fuente, ya que esto da combinaciones de huipiles conocidos que no hayamos visto antes. En el código, hacemos esto mediante `--col-styles=8-15` (i.e., de $64^2$ a $512^2$).

En el [código de StyleGAN2 que uso](https://github.com/PDillis/stylegan2-fun), especificamos a estos como: `--row-seeds=17,32,78,300` y `--col-seeds=23,50,200,512`, respectivamente. Asimismo, el estilo que más me ha gustado traducir de una imagen a otra es el estilo **fino**, es decir, solamente copiamos los detalles finos (colores) de las semillas destino a las semillas fuente, ya que esto da combinaciones de huipiles conocidos que no hayamos visto antes. En el código, hacemos esto mediante `--col-styles=8-15` (i.e., de $64^2$ a $512^2$).

El comando utilizado fue el siguiente:

```
python run_generator.py \
       style-mixing-example \
       --network=/path/to/network.pkl \
       --row-seeds=17,32,78,300 \
       --col-seeds=23,50,200,512 \
       --col-styles=8-15 \
       --truncation-psi=1.0
```

Obtenemos entonces al siguiente grid:

<div class="container">
<img src="/img/sgan2/style-transfer/grid.png" style="width: 100%;" alt="Grid con todas las semillas">
</div>

Si queremos apreciar los detalles, podemos entonces comparar a dos imágenes mediante un paquete de comparación de imágenes llamado [BeerSlider](https://pepsized.com/wp-content/uploads/2018/09/beerslider/demo/index.html). Por ejemplo, para ver el resultado de mezclar las semillas `17` con `23`, en el primer slider podemos ver a la derecha, la imagen generada con la semilla `17` y a la izquierda el resultado de la mezcla con la semilla `23`.

<div class="beer-slider-row">
  <div class="beer-slider-column">
    <div class="beer-slider beer-ready" id="beer-slider1" data-beer-label="Semilla 17">
      <img src="/img/sgan2/style-transfer/17-17.png" alt="Seed 17">
      <div class="beer-reveal" data-beer-label="23">
        <img src="/img/sgan2/style-transfer/17-23.png" alt="Mezcla con Semilla 23">
      </div>
    </div>
  </div>

  <div class="beer-slider-column">
    <div class="beer-slider beer-ready" id="beer-slider2" data-beer-label="Semilla 78">
      <img src="/img/sgan2/style-transfer/78-78.png" alt="Seed 78">
      <div class="beer-reveal" data-beer-label="50">
        <img src="/img/sgan2/style-transfer/78-50.png" alt="Mezcla con Semilla 50">
      </div>
    </div>
  </div>
</div>

<div class="beer-slider-container">
  <div class="beer-slider beer-ready" id="beer-slider3" data-beer-label="Semilla 300">
    <img src="/img/sgan2/style-transfer/300-300.png" alt="Seed 300">
    <div class="beer-reveal" data-beer-label="512">
      <img src="/img/sgan2/style-transfer/300-512.png" alt="Mezcla con Semilla 512">
    </div>
  </div>
</div>

He seleccionado tres que me han gustado bastante, pero por supuesto, existen muchos mas que aún no he explorado. Espero que en el próximo blog post (cuando suceda) pueda explorar más a fondo los modelos que he usado para este proyecto: [ProGAN](https://github.com/tkarras/progressive_growing_of_gans)[^progan], [StyleGAN](https://github.com/NVlabs/stylegan)[^sgan] y el ya mencionado StyleGAN2[^sgan2].

Hasta la próxima vez ~~que termine el último blog post~~.

[^sgan]: T. Karras, S. Laine & T. Aila, [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948). In *CVPR*, 2019.
[^sgan2]: T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen & T. Aila, [Analyzing and Improving the Image Quality of StyleGAN](http://arxiv.org/abs/1912.04958). In *CoRR*, 2020.
[^progan]: T. Karras, T. Aila, S. Laine & J. Lehtinen, [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196). In *ICLR*, 2018.

<script src="/assets/js/BeerSlider.js"></script>
<script>
  new BeerSlider( document.getElementById( "beer-slider1" ), { start: 50 } );
  new BeerSlider( document.getElementById( "beer-slider2" ), { start: 50 } );
  new BeerSlider( document.getElementById( "beer-slider3" ), { start: 50 } );
</script>

{% include disqus.html %}
