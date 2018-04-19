---
layout: post
title:  "Making a Chatbot in Python - Part 0"
date:   2018-04-18 18:00:00
categories: main
---

As of last year, thanks to my internship at the [Computer Vision Center](http://www.cvc.uab.es) and the [work](https://github.com/PDillis/DQN-CarRacing) [I did](https://github.com/PDillis/CartPole-Cases) [there](https://github.com/PDillis/Experiment-CarRacing) using [Reinforcement Learning](https://deeplearning4j.org/deepreinforcementlearning), I have become extremely interested in both the current state and applications of AI, as well as its origins. Without much reservation, there was an area I did not hold particularly dear: that of [Natural Language Processing](https://en.wikipedia.org/wiki/Natural-language_processing), or NLP. I believe this was due to what was presented to me as being *the* application of NLP: chatbots, in particular, chatbots for companies (I will admit another popular area in NLP is that of predictive text keyboards, which most of the times get [hilarious results](https://twitter.com/botnikstudios/status/986295833451692032) and I wish to experiment with this later on).

I have used chatbots in the past, and they have never been engaging or, in the majority of the cases, even useful in answering my questions or solving my problems; there was always a need for a human representative for help. In retrospective, I should immediately note that this was due to bias: chatbots are key in [answering routine questions](https://www.ibm.com/blogs/watson/2017/10/how-chatbots-reduce-customer-service-costs-by-30-percent/) (and saving company money), of which I never ask. Indeed, I prefer to find the answer for myself (or by other user's questions), and if no answer satisfies my need, then I would contact the company. In the cases my questions arrived a chatbot, it would always end with a non-answer or waiting time for a customer representative to get back to me. Thus, perhaps I was being too harsh on both NLP, but especially chatbots.

There were other key turning points that made me look at chatbots in a different light. One of these was reading [Greg Brockman's path to AI](https://blog.gregbrockman.com/my-path-to-openai). Was I truly missing something by not looking at chatbots in a more positive way? After all, language is one of the key pillars in society, indeed in intelligence. Should I not then explore them further? 

Perhaps chatbots had far more to offer than what I had initially realized or was willing to give credit for. This was reinforced when, whilst attending [2018's State of Latino Entrepeneurship Forum](https://www.gsb.stanford.edu/events/2018-state-latino-entrepreneurship-forum), I met [Conchita Franco Serri](https://www.linkedin.com/in/conchitaserri), Founder and CEO of [Caroami](http://caroami.com/). What Caroami plans to do (as it hasn't yet to launched), is to resolve conflicts between two parties via a mediator, the chatbot. They will start with small scale conflicts, like disputes between roomates, or between couples, and then scale as time progresses. Perhaps they are overselling what their bot can do *now*, but it has great potential. Indeed, chatbots, like any other machine learning paradigm, will get better with more data.

This was further cemented when, whilst completing [Coursera's Deep Learning Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/M94FBCS34JG5), [Prof. Andrew Ng](http://www.andrewng.org/) showed us a new chatbot for mental health: [Woebot](https://woebot.io/). Whilst not a complete replacement for therapists, Woebot offers a far more powerful use for chatbots than I had ever seen before, and I loved it (and have even started to use it). After that, I found [spaCy](https://spacy.io/), which offer industrial-strength NLP in different languages, and I basically just started to play with it. From analyzing and visualizing novel snippets, to comparing sentences via [word vectors](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/) (more on that on a later blog post).  

By this point I have decided to build my own chatbot, and this is where we are now. Thus, I will write a bit of what I have learned (which is also a bit), and then we will proceed to both make our chatbot, as well as to gradually improve it, using more advanced tools like spaCy or [TensorFlow](https://www.tensorflow.org/).

## A bit of history

Since we are trying to start making our own chatbots, it is natural for us to look back in time and study (albeit superficially) the first chatbot ever made. This was [ELIZA](https://en.wikipedia.org/wiki/ELIZA), created by [Joseph Weizenbaum](http://history.computer.org/pioneers/weizenbaum.html) from 1964 to 1966 at the MIT Artificial Intelligence Laboratory. ELIZA was born https://www.cse.buffalo.edu//~rapaport/572/S02/weizenbaum.eliza.1966.pdf


To Weizenbaum, we should never allow computers to make any decisions, as they lack compassion and wisdom, purely human emotions. Humans have judgement, which in turn allows us to compare apples with oranges. 

This is even more apparent in the following quote (from an [excerpt found online](https://web.archive.org/web/20050508173416/http://www.smeed.org/1735)):

> I want them [teachers of computer science] to have heard me affirm that the computer is a powerful new metaphor for helping us understand many aspects of the world, but that it enslaves the mind that has no other metaphors and few other resources to call on. The world is many things, and no single framework is large enough to contain them all, neither that of man's science nor of his poetry, neither that of calculating reason nor that of pure intuition. And just as the love of music does not suffice to enable one to play the violin - one must also master the craft of the instrument and the music itself - so it is not enough to love humanity in order to help it survive. The teacher's calling to his craft is therefore an honorable one. But he must do more than that: he must teach more than one metaphor, and he must teach more by the example of his conduct than by what he writes on the blackboard. He must teach the limitations of his tools as well as their power. 

A recreation of ELIZA can be found [here](http://deixilabs.com/eliza.html) (in spanish)

### Building ELIZA

The chatbot we will build will be greatly influenced by the following: 

- Noah Moroze's [simpleChatBot](https://github.com/nmoroze/SimpleChatBot)
- Jezz Higgin's [recreation of ELIZA](https://github.com/jezhiggins/eliza.py)
- DataCamp's course on [building chatbots in Python](https://www.datacamp.com/courses/building-chatbots-in-python) given by Alan Nichol, co-founder and CTO of [Rasa](http://rasa.com/). 

The latter will set the tone and direction we will take, as it helped me realize how to scale this endeavor. So, in order for the user and chatbot to engage in a conversation, we have the following template (to be filled) by the algorithm we will develop later on:

```python
user_template = "USER : {}"
bot_template = "ELIZA : {}"
```

As a first step,`ELIZA` will simply repeat back the message the user inputs:

```python
def respond(message):
	# Concatenate the user's message to the end of a standard bot response
	bot_message = "I can hear you! You said: " + message
	return bot_message
```

For the bot to receive the message sent by the user, we must define another function, `send_message()`, which will process the `message` that the user inputs, whilst printing it and the `ELIZA`'s response:

```python
def send_message(message):
	# Print user_template including user_message
	print(user_template.format(message))
	# Get ELIZA's response to the message:
	response = respond(message)
	# Print the bot template including ELIZA's response:
	print(bot_template.format(response))
```

For example, we have the following:

```python
>>> send_message("Hello there!")
'USER: Hello there!'
'ELIZA: I can hear you! You said: Hello there!'
```

This might seem impressive at first, but it can trick a user only so far. This version of `ELIZA` is neither expressive, memorable or charismatic, some of the main points by which we judge a conversation with another human, perhaps even moreso a machine. 

Personality is essential to any chatbot, indeed to any human! This is subconciously expected by the user: if it does not meet our expectations, then we shy away from it (like I do nowadays), and this is why I particularly despise call centers. 

<div class="polaroid">
	<img src="https://images.pexels.com/photos/41280/agent-business-call-center-41280.jpeg" alt="Your worst nightmare" width="324" height="487" class = "center">
	<div class="container"><p><b>Hi! This is the T-1000...I mean, Jenny, I will be your representative!</b> The majority of the cases, I cannot tell the difference.</p>
		</div>
</div>
