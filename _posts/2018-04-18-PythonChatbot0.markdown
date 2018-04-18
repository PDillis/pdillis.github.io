---
layout: post
title:  "Making a Chatbot in Python - Part 0"
date:   2018-04-18 18:00:00
categories: main
---

As of last year, thanks to my internship at the [Computer Vision Center](http://www.cvc.uab.es) and the [work](https://github.com/PDillis/DQN-CarRacing) [I did](https://github.com/PDillis/CartPole-Cases) [there](https://github.com/PDillis/Experiment-CarRacing) using [Reinforcement Learning](https://deeplearning4j.org/deepreinforcementlearning), I have become extremely interested in both the current state and applications of AI, as well as its origins. Without much reservation, there was an area I did not hold particularly dear: that of [Natural Language Processing](https://en.wikipedia.org/wiki/Natural-language_processing), or NLP. I believe this was due to what was presented to me as being *the* application of NLP: chatbots, in particular, chatbots for companies (I will admit another popular area in NLP is that of predictive text keyboards, which most of the times get [hilarious results](https://twitter.com/botnikstudios/status/986295833451692032)).

I have used chatbots in the past, and they have never been engaging or, in the majority of the cases, even useful in answering my questions or solving my problems; there was always a need for a human representative for help. In retrospective, I should immediately note that this was due to bias: chatbots are key in [answering routine questions](https://www.ibm.com/blogs/watson/2017/10/how-chatbots-reduce-customer-service-costs-by-30-percent/) (and saving company money), of which I never ask. Indeed, I prefer to find the answer for myself (or by other user's questions), and if no answer satisfies my need, then I would contact the company. In the cases my questions arrived a chatbot, it would always end with a non-answer or waiting time for a customer representative to get back to me. Thus, perhaps I was being too harsh on both NLP, but especially chatbots.

There were other key turning points that made me look at chatbots in a different light. One of these was reading [Greg Brockman's path to AI](https://blog.gregbrockman.com/my-path-to-openai). Was I truly missing something by not looking at chatbots in a more positive way? This was further cemented when, whilst completing [Coursera's Deep Learning Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/M94FBCS34JG5), [Prof. Andrew Ng](http://www.andrewng.org/) showed us a new chatbot for mental health: [Woebot](https://woebot.io/). Whilst not a complete replacement for therapists, Woebot offers a far more powerful use for chatbots than I had ever seen before, and I loved it.

<script src="https://gist.github.com/PDillis/8536d28861252a31de8db48c591651b6.js"></script>

DataCamp's course on [building chatbots in Python](https://www.datacamp.com/courses/building-chatbots-in-python) given by Alan Nichol, co-founder and CTO of [Rasa](http://rasa.com/)

## A bit of history

Since we are trying to start making our own chatbots, it is natural for us to look back in time and study (albeit superficially) the first chatbot ever made. This was [ELIZA](https://en.wikipedia.org/wiki/ELIZA), created by [Joseph Weizenbaum](http://history.computer.org/pioneers/weizenbaum.html) from 1964 to 1966 at the MIT Artificial Intelligence Laboratory. ELIZA was born https://www.cse.buffalo.edu//~rapaport/572/S02/weizenbaum.eliza.1966.pdf


To Weizenbaum, we should never allow computers to make any decisions, as they lack compassion and wisdom, purely human emotions. Humans have judgement, which in turn allows us to compare apples with oranges. 

This is even more apparent in the following quote (from an [excerpt found online](https://web.archive.org/web/20050508173416/http://www.smeed.org/1735)):

> I want them [teachers of computer science] to have heard me affirm that the computer is a powerful new metaphor for helping us understand many aspects of the world, but that it enslaves the mind that has no other metaphors and few other resources to call on. The world is many things, and no single framework is large enough to contain them all, neither that of man's science nor of his poetry, neither that of calculating reason nor that of pure intuition. And just as the love of music does not suffice to enable one to play the violin - one must also master the craft of the instrument and the music itself - so it is not enough to love humanity in order to help it survive. The teacher's calling to his craft is therefore an honorable one. But he must do more than that: he must teach more than one metaphor, and he must teach more by the example of his conduct than by what he writes on the blackboard. He must teach the limitations of his tools as well as their power. 



### Step 1: Building EchoBot

As a first step, we will build `EchoBot`, which will simply repeat back the message the user 

```python
# EchoBot will simply respond by replying with the same message it receives.

bot_template = "BOT : {0}"
user_template = "USER : {0}"

# Define a function that responds to a user's message:
def respond(message):
	# Concatenate the user's message to the end of a standard bot response
	bot_message = "I can hear you! You said: " + message
	return bot_message

# Define a function that sends a message to the bot. This will log the message
# and the bot's response.
def send_message(message):
	# Print user_template including user_message
	print(user_template.format(message))
	# Get the bot's response to the message:
	response = respond(message)
	# Print the bot template including the bot's response:
	print(bot_template.format(response))
```

This bot is neither expressive, memorable or charismatic, some of the main points by which we judge a conversation with another human, much moreso a machine. Personality is essential to any chatbot, indeed to any human! This is subconciously expected by the user: if it does not meet our expectations, then we shy away from it (like I do nowadays), and this is why 
