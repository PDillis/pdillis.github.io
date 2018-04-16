---
layout: post
title: "42 - March 2018 Piscine"
date: 2018-04-16 10:00:00
categories: main
image_sliders:
  - slider1
---

After a month of blood, sweat and tears (usually all three in the form of code), I have emerged victorious from the [42 Piscine!](https://www.42.us.org/program/intensive-basic-training/). Victorious from meeting amazingly talented and kind people, making new friends, learning more than I have ever learned in a month, and gaining much-needed confidence in my coding skills. I must also add one last piece of excitement: I got accepted!

![42acceptedemail](https://user-images.githubusercontent.com/24496178/38800989-7f5322d6-411d-11e8-82e2-15a94b7eef3e.PNG)

The Piscine requires a lot of planning: given that the projects, daily exercises, and sometimes even exams overlap the majority of the time, it is up to you to decide what to give priority to, not only because you don't have the time to do them all, but because they're not worth the same amount of points (of course, do the exams!). This taught me a lot on how to plan ahead, as well as to rely on the previous knowledge of my fellow pisciners. The conjecture we arrived as to what the final grade received per project is as follows:

\[ \small
final\_grade(MGrade, AVGrade)=
\begin{cases}
\max(MGrade, AVGrade), \text{ if } |MGrade - AVGrade| <= 10\\
\min(MGrade, AVGrade), \text{ otherwise} 
\end{cases}
]\

where $MGrade$ is the grade given by Moulinette, a computer program that automatically graded your work, and $AVGrade$ is the average grade given to you by your fellow correctors (some projects required 1 corrector, others up to 3 correctors).

While not my original end goal, there are many different paths to take during life as a 42 Cadet, which makes their offer very hard to say no to. The constant flow of speakers like [Deon Nicholas](https://angel.co/deon-nicholas) from [Forethought AI](https://www.forethought.ai/) and [Jane Herriman](https://twitter.com/janeherriman) presenting [Julia](https://www.juliabox.com/), combined with the presence of students pushing themselves forward, is the perfect recipe for success. It is not to be taken lightly, and now I do wish to become a part of the 42 school.

If you are curious, you can find (almost) all the code I generated, along with a bit more information of the Piscine in [my GitHub](https://github.com/PDillis/42SiliconValley). If you plan to attend any future Piscine, ignore my code and any you may find on GitHub and just go with a clean mind; it's for the best. Else, know that most of the coding was done in [C](https://en.wikipedia.org/wiki/C_(programming_language)), but on the first days there was some [Shell](https://en.wikipedia.org/wiki/Shell_(computing)) scripting, which I found to be extremely insightful.

Without further ado, I leave you with some pictures I managed to take during this month of coding (some pictures have secret hidden links):

{% include slider.html selector="slider1" %}

In the next post, I plan to show how to create a chatbot with Python, so stay tuned!
