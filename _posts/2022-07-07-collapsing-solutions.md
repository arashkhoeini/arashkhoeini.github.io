---
layout: post
title: Self-supervision and Collapsing Solutions
author: Arash Khoeini
date: 2022-07-07 22:22:00 -0500
categories: [Tutorial]
tags: [deep learning, self-supervised learning]
image: ssl-header.jpg
---


# Collapsing Solutions in Self-supervised Learning

[In the previous post I explained how self-supervised learning has been established as a decent method for unsupervised representation learning. I dicussed pre-text task learning, contrastive learning, and touched upon a few non-contrastive learning methods.]({% post_url 2022-10-24-self-supervision %})

In this post, I aim to dig deeper into similarity learning. Similarity learning is a more general self-supervised learning approach and includes both contrastive and non-contrastive methods. In particular, I talk about collapsing solutions, what they are, and how different methods use different strategies to avoid these collapsing solutions.

## Similarity Learning

SSimilarity learning is a simple concept that most of the self-supervised methods are built based on. In similarity learning, a feature extractor f learns to map similar input close to each other in the representation space. This is very intuitive and could be looked at from various levels: two photos of a same cat but with different pose should be mapped close to each other in the representation space, two photos of different cats should be mapped relatively close to each other, compared to other classes like dogs. To learn such similarities, most of the current methods use some version of Siamese networks. Siamese networks are weight-sharing networks and this makes them a natural tool for comparing two or more inputs. In practive, Siamese network is just a fancy name for feeding two different inputs through the same network twice, and computing the desired loss function using two extracted representations. The two different input samples are usually two different augmented versions of the same sample.

![alt text]({{ site.baseurl }}/assets/img/collapsing-solutions.jpeg){: style="max-width: 90%" } 

[image source](https://scitechdaily.com/covid-19-has-triggered-a-global-financial-crisis-and-called-into-question-the-us-dollars-hegemony-whats-next/)

## Collapsing Solutions
There exists one major problem with similarity learning using Siamese networks and that’s called collapsing solutions! Forcing the network to map two different images very close to each other might end up in the network learning to cheat, and map EVERY input into a one same representation. This simply happens because our network learns to create a shortcut for minimizing loss: by just mapping everything to the same input.

Various strategies have been introduced in the self-supervised learning literature to avoid such collapsing shortcuts. We will discuss some of them briefly here.

### Contrastive Learning
The core idea behind contrastive learning is to attract the positive sample pairs and repulse the negative sample pairs. The idea is straight forward: if we want to avoid our network to map every sample into a single point (to avoid finding a collapsing solution) we need to also define negative pairs and force our model to map them far apart. Different methods have different methodologies in order to define and use negative pairs. One well-known contrastive learning method is called SimCLR and it was introduced by Geoffrey Hinton team at Google Brain. SimCLR is short for Simple Contrasting LeaRning and its simplicity truly fits its name! In SimCLR, we create an augmentation for each data point in our batch. So we will have 2N data points for batch size N. Then for each input x_i, we take its augmentations as the positive sample for x_i and train the the network to minimize the distance between their latent representations. Meanwhile, we take all 2N-2 remaining samples as the negative samples for x_i and maximize the distance between their latent representation and x_i’s. This distance maximization between each input and its negative pairs is the key in avoiding collapsing solutions.

### Clustering Methods
Clustering methods use one distorted sample to compute ‘targets’ for the loss and another distorted version of the sample to predict these targets. This is followed by an alternate optimization (e.g. similar to k-means method) in [DeepCluster](https://arxiv.org/pdf/1807.05520.pdf) or non-differentiable operators in [SwAV](https://arxiv.org/pdf/2006.09882.pdf).

### Asymmetry In Siamese Networks
When using Siamese networks for similarity learning, the main reason for collapsing is actually the symmetric architecture of the network. This symmetry exists because of the weight sharing in Siamese network. Although weight sharing is very intuitive for learning similarities, it is the driving force for collapsing. Therefore in another recent like of work new methods with asymmetric network architecture are introduced. In one method, named [BYOL (Bootstrap Your Own Latent representations)](https://arxiv.org/pdf/2006.07733.pdf) two sub-networks of the Siamese network perform very different roles. Both of these sub-networks have a representation component and a projection component which projects the input image into its latent representations. But the top network (named as the online network) has one extra component compared to the bottom network (named as the target network). This extra component, which is called the prediction component, receives the latent representation of the online network as its input and tries to predict the latent representation of the target network. The other difference between these two networks is how their parameters are updated. The online network is trained through stochastic gradient descent. However, the target network is trained using the slow moving average of the online network. In another work, SimSiam, authors show that the only thing we need to prevent collapsing is a stop-grad operation. In SimSiam, the network architecture is modified to be asymmetric using a special ‘predictor’ network and the parameter updates are asymmetric such that the model parameters are only updated using one distorted version of the input, while the representations from another distorted version are used as a fixed target. Authors of SimSiam conclude that the asymmetry of the learning update, ‘stop-gradient’, is critical to preventing trivial solutions.

In this blog post I explained collapsing solutions and briefly introduced three lines of research targeting to avoid collapsing solutions. Please let me know in the comments if you know any other interesting method, or if there is any particular related paper that you would like me to summarize or explain in my Medium. This is actually what I think I am going to focus more on in this blog: to explain the core idea of the interesting papers I read with a simple language. Thank you for reading :)

