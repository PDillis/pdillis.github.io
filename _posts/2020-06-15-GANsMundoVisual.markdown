---
layout: post
title:  "GANs - En el Mundo Visual"
date:   2020-06-15 18:00:00
categories: main

---

# Tech Community Day

Hace unas semanas me pidieron si podría dar una charla acerca de un tema de Inteligencia Artificial en el [Tech Community Day](https://techcommunityday.com/). Es la primera vez que he dado una charla de este tipo, por lo que claramente estaba totalmente nervioso de aceptar, pero emocionado al mismo tiempo de poder compartir un poco de lo que he estado realizando durante mi doctorado. 

El tema que he escogido fue de [GANs](https://blog.diegoporres.com/main/2019/07/17/UnsupervisingArt/) y decidí realizarla en español, ya que la información que se puede conseguir del tema en inglés abunda, mas tristemente no es tan fácil en español. Asimismo, el título de la charla fue **GANSs - En el Mundo Visual** y pueden encontrar la grabación de la misma en [YouTube](https://youtu.be/DdD39y8rJQ8).

Si bien condensar tanta información en una charla de 1 hora es difícil, tuve que dejar fuera ciertos puntos y aplicaciones interesantes de las GANs fuera, pero también me ayudó a saber qué más era de interés del público en general (para la próxima):

* ¿Qué aplicaciones concretas se ha realizado de las GANs en medicina, arquitectura o diseño?
* ¿Cómo podemos saber que hay que parar de entrenar a una GAN?
    * Por aparte, esto lleva a los problemas típicos que se enfrenta uno al entrenar una GAN (colapso, colapso total o inclusive no converger, por ejemplo).
* ¿Puede una GAN llegar a sobreajustar (overfit) los datos de un dataset?
* ¿Debemos de hacer públicos todos los modelos generativos o hechos con una GAN?

Sobre el último punto, mi respuesta sería 'No', que es justamente lo que he querido comunicar con los modelos que he entrenado para la [generación de huipiles](https://blog.diegoporres.com/main/2019/09/23/Threads/): compartir dichos modelos puede llevar fácilmente al abuso de la recreación o apropiación de los diseños y patrones de los huipiles, por lo que no los he hecho públicos, aunque me lo han pedido en [varias ocasiones](https://twitter.com/PDillis/status/1270365318599278593?s=20).

Pueden encontrar la grabación de la charla [aquí](https://www.youtube.com/watch?v=DdD39y8rJQ8) y las slides utilizadas (editadas para no incluir todo mi arte generado al final) a continuación:

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTGilWQpywgIhU1kCfir3zZwusptKPkvVYPdH1Qdga4hF_6Sz38gZerCVchykZHZqD9MzplXZWWNm5H/embed?start=false&loop=false&delayms=5000" frameborder="0" width="800" height="466" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

# Transferencia de Estilo

Me cago en todo Respecto al ultimo proyecto mencionado usando [StyleGAN2](https://github.com/NVlabs/stylegan2), [*Threads*](https://www.youtube.com/watch?v=t9fv4AAt6lw), quiero probar un paquete de comparación de imágenes llamado [twentytwenty](https://zurb.com/playground/twentytwenty). 

<script src="https://cdnjs.cloudfare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>

<script src="/assets/js/twentytwenty/jquery.event.move.js" type="text/javascript"></script>
<script src="/assets/js/twentytwenty/jquery.twentytwenty.js" type="text/javascript"></script>
<link rel="stylesheet" href="/assets/css/twentytwenty/twentytwenty.css" type="text/css" media="screen" />

<!-- <script src="{{ "/twentytwenty/js/jquery.event.move.js" | relative_url }}" type="text/javascript"></script>
<script src="{{ "/twentytwenty/js/jquery.twentytwenty.js" | relative_url }}" type="text/javascript"></script>
<link rel="stylesheet" href="{{ "/twentytwenty/css/twentytwenty.css" | relative_url }}" type="text/css" media="screen"> -->

<script src="https://cdnjs.cloudfare.com/ajax/libs/jquery/3.2.1/jquery.min.js">
$(window).load(function() {
  $(".twentytwenty-container").twentytwenty({
      before_label: '',
      after_label: '',
      click_to_move: true
  });
});
</script>

<div class="twentytwenty-container" id="container1" >
 <!-- The before image is first -->
 <img src="/img/sgan2/style-transfer/32-32.png" />
 <!-- The after image is last -->
 <img src="/img/sgan2/style-transfer/32-64.png" />
</div>

## Gemfiles

Sobre como correr [GitHub pages](https://pages.github.com/) [localmente](https://help.github.com/en/github/working-with-github-pages/testing-your-github-pages-site-locally-with-jekyll), primero se debe de [instalar a Jekyll](https://jekyllrb.com/docs/installation/). Luego, lo mas sencillo es hacer un `git clone` al repositorio original de Cayman o bien bajar el archivo `.zip` que se menciona en [mi repositorio](https://github.com/PDillis/pdillis.github.io#how-to-use-it), descomprimirlo y luego correr `cd jekyll-cayman-theme-master`.

Debido a que se [requiere usar Bundler 1.12](https://github.com/PDillis/pdillis.github.io/blob/ac44808e7d7bef62281d9646c573e96eebce20e3/jekyll-cayman-theme.gemspec#L16) en el Cayman Theme que he seleccionado, debemos de primero instalarlo mediante:

```
gem install bundler -v 1.12
```

y luego usarlo:

```
bundle _1.12_ install
```

de lo contrario, lo normal en las instrucciones es de simplemente usar `bundle install`. Es normal que hayan errores de con los paquetes necesarios/instalados/activados, así que de envés de simplemente usar `jekyll serve`, [lo correcto](https://stackoverflow.com/a/6393129) será usar:

```
bundle exec jekyll serve
```

Así, podemos dirigirnos a `http://127.0.0.1:4000` y ver la pagina en 'vivo'. 

Hasta la próxima vez ~~que termine el último blog post~~.

{% include disqus.html %}
