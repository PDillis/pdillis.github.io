---
layout: post
title:  "GANs - En el Mundo Visual"
date:   2020-06-15 18:00:00
categories: main

---

# Tech Community Day

Hace unas semanas me pidieron si podría dar una charla acerca de un tema de Inteligencia Artificial en el [Tech Community Day](https://techcommunityday.com/). Es la primera vez que he dado una charla de este tipo, por lo que claramente estaba totalmente nervioso de aceptar, pero emocionado al mismo tiempo de poder compartir un poco de lo que he estado realizando durante mi doctorado. 

El tema que he escogido fue de [GANs](https://blog.diegoporres.com/main/2019/07/17/UnsupervisingArt/) y decidí realizarla en español, ya que la información que se puede conseguir del tema en inglés abunda, mas tristemente no es tan fácil en español. Si bien condensar tanta información en una charla de 1 hora es difícil, tuve que dejar fuera ciertos puntos y aplicaciones interesantes de las GANs fuera, pero también me ayudó a saber qué más era de interés del público en general (para la próxima):

* ¿Qué aplicaciones concretas se ha realizado de las GANs en medicina, arquitectura o diseño?
* ¿Cómo podemos saber que hay que parar de entrenar a una GAN?
    * Por aparte, esto lleva a los problemas típicos que se enfrenta uno al entrenar una GAN (colapso, colapso total o inclusive no converger, por ejemplo).
* ¿Puede una GAN llegar a sobreajustar (overfit) los datos de un dataset?
* ¿Debemos de hacer públicos todos los modelos generativos o hechos con una GAN?

Sobre el último punto, mi respuesta sería 'No', que es justamente lo que he querido comunicar con los modelos que he entrenado para la [generación de huipiles](https://blog.diegoporres.com/main/2020/06/21/OnStyle/): compartir dichos modelos puede llevar fácilmente al abuso de la recreación o apropiación de los diseños y patrones de los huipiles, por lo que no los he hecho públicos, aunque me lo han pedido en [varias ocasiones](https://twitter.com/PDillis/status/1270365318599278593?s=20).

Pueden encontrar la grabación de la charla [aquí](https://www.youtube.com/watch?v=DdD39y8rJQ8) y las slides utilizadas (editadas para no incluir todo mi arte generado al final) a continuación:

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTGilWQpywgIhU1kCfir3zZwusptKPkvVYPdH1Qdga4hF_6Sz38gZerCVchykZHZqD9MzplXZWWNm5H/embed?start=false&loop=false&delayms=5000" frameborder="0" width="800" height="466" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

# Transferencia de Estilo

Respecto al ultimo proyecto mencionado usando [StyleGAN2](https://github.com/NVlabs/stylegan2), [*Threads*](https://www.youtube.com/watch?v=t9fv4AAt6lw), quiero probar un paquete de comparación de imágenes llamado [twentytwenty]

<script src="assets/js/twentytwenty/jquery-3.2.1.min.js" type="text/javascript"></script>
<script src="assets/js/twentytwenty/jquery.event.move.js" type="text/javascript"></script>
<script src="assets/js/twentytwenty/jquery.twentytwenty.js" type="text/javascript"></script>
<link rel="stylesheet" href="assets/css/twentytwenty/twentytwenty.css" type="text/css" media="screen" />

<div id="container1" class="twentytwenty-container">
 <!-- The before image is first -->
 <img src="/img/sgan2/style-transfer/32-32.png" />
 <!-- The after image is last -->
 <img src="/img/sgan2/style-transfer/32-64.png" />
</div>


Hasta la próxima vez ~~que termine el último blog post~~.

{% include disqus.html %}
