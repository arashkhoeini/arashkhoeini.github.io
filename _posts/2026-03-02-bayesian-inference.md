---
layout: post
title: "From Prior to Posterior: A Guide to Bayesian Inference"
date: 2026-03-02
categories: [machine-learning, bayesian-inference]
math: true
---

Bayesian inference is one of the most principled frameworks for reasoning under uncertainty. At its core, it is about updating your beliefs when you see new data. In this post, we will build up the full picture: starting from the intuition of a prior, working through how the posterior is computed, why we care about it, and then exploring how modern algorithms like MCMC and Variational Inference approximate it when exact computation is impossible.

---

## The Prior: Your Belief Before Seeing Data

Suppose you are trying to learn something about a latent variable $t$; this could be the weights of a neural network, the parameters of a regression model, or anything else you want to infer. Before you observe any data, you already have *some* belief about what $t$ might look like. Maybe you think small weights are more likely than large ones, or that a coin is probably close to fair.

This initial belief is captured by the **prior distribution** $p(t)$. It is a probability distribution over all possible values of $t$, encoding your assumptions before any evidence is considered.

The prior can be:
- **Informative**: you have strong domain knowledge and encode it explicitly
- **Weakly informative**: you have mild constraints, like "weights should be small"
- **Uninformative**: you have little prior knowledge and try to stay neutral

The prior is not a guess pulled from thin air. It is a formal expression of what you know (or don't know) before looking at your data.

---

## The Posterior: Updating Your Belief With Data

Once you observe a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$, you want to *update* your belief about $t$ in light of the evidence. The result is the **posterior distribution** $p(t \mid \mathcal{D})$, and it is computed using **Bayes' Theorem**:

$$p(t \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid t) \cdot p(t)}{p(\mathcal{D})}$$

Let's unpack each term.

### The Likelihood: $p(\mathcal{D} \mid t)$

This asks: *if $t$ were the true value, how probable would the observed data be?* For example, if $t$ parameterises a Gaussian distribution and the data points cluster tightly around the predicted mean, the likelihood is high. If the data looks nothing like what $t$ would predict, the likelihood is low.

The likelihood acts as the **evidence filter**; it up-weights values of $t$ that explain the data well.

### The Prior: $p(t)$

As discussed above, this is your belief about $t$ before seeing data. In the numerator, it acts as a **regulariser**, and values of $t$ that were already implausible under the prior get down-weighted even if they fit the data well.

### The Marginal Likelihood (Evidence): $p(\mathcal{D})$

The denominator $p(\mathcal{D})$ is the probability of observing the data *regardless* of what $t$ is. It is computed by marginalising over all possible values of $t$:

$$p(\mathcal{D}) = \int p(\mathcal{D} \mid t)\, p(t)\, dt$$

Its role is purely to **normalise** the posterior so it integrates to 1, making it a valid probability distribution.

---

## Why Do We Care About the Posterior?

Before discussing how to compute or approximate the posterior, it is worth asking: why do we want it in the first place? There are two closely related answers.

### 1. Making Predictions: The Posterior Predictive Distribution

In supervised learning, the ultimate goal is to predict the output $y$ for a new input $x$. A standard (non-Bayesian) model picks a single best set of parameters $\hat{t}$ and predicts using $p(y \mid x, \hat{t})$. But this ignores a fundamental question: *how confident are you that $\hat{t}$ is the right set of parameters?*

The Bayesian answer is to not commit to a single $t$ at all. Instead, average your predictions over *all* plausible values of $t$, weighted by how probable each is given the data. This gives the **Posterior Predictive Distribution (PPD)**:

$$p(y \mid x, \mathcal{D}) = \int p(y \mid x, t)\, p(t \mid \mathcal{D})\, dt$$

Intuitively, the PPD says: "my prediction is a weighted mixture of what every plausible model would predict." Values of $t$ that are more consistent with the data (higher posterior probability) contribute more to the prediction. This naturally gives you **uncertainty quantification**, if the posterior is spread out over many different values of $t$ that make very different predictions, the PPD will be wide, reflecting genuine uncertainty.

### 2. Bayesian Neural Networks (BNNs)

A **Bayesian Neural Network** is simply a neural network treated in the Bayesian way: instead of learning a single fixed set of weights $t$, we place a prior $p(t)$ over the weights and infer the posterior $p(t \mid \mathcal{D})$ after seeing data.

This might sound like a small change, but the implications are significant. A standard neural network gives you one answer. A BNN gives you a *distribution* over answers, directly reflecting how uncertain the model is. This uncertainty comes in two flavours:

- **Epistemic uncertainty** (model uncertainty): the network has not seen enough data to be sure which weights are correct. This is captured by a spread-out posterior over $t$, and it *decreases* as you collect more data.
- **Aleatoric uncertainty** (data uncertainty): the data itself is noisy, so even with perfect knowledge of $t$ there is irreducible randomness in $y$. No amount of extra data can eliminate this.

Crucially, predictions from a BNN are made *exactly* via the PPD above, you integrate out the weights under the posterior. So BNNs are not a separate concept from the PPD; they are just the PPD applied to a neural network. The posterior is the central object, and making predictions with a BNN *is* computing the PPD. This is why getting the posterior right matters so much: a poor approximation to $p(t \mid \mathcal{D})$ leads to poorly calibrated uncertainty estimates, which can be dangerous in high-stakes applications like medical diagnosis or autonomous driving.

---

## Why is the Posterior Intractable?

The posterior is what we want, and the PPD is what we use it for. So how do we actually compute $p(t \mid \mathcal{D})$?

The difficulty lies entirely in the denominator:

$$p(\mathcal{D}) = \int p(\mathcal{D} \mid t)\, p(t)\, dt$$

This integral requires summing over *every possible configuration of $t$*. In a simple model with one or two parameters, this might be feasible. But in a neural network, $t$ represents millions of weights. Integrating over a space of millions of dimensions is computationally impossible to do exactly.

There is one important exception: when the prior and likelihood are **conjugate**, meaning they are from compatible distribution families, the posterior has a known closed form. The most famous example is Gaussian Processes (GPs): because everything is jointly Gaussian, all the integrals work out analytically and the PPD has a clean closed-form solution. But for most real-world models, and certainly for neural networks, no such shortcut exists.

So we need approximations. The two most prominent classes of methods are **Markov Chain Monte Carlo (MCMC)** and **Variational Inference (VI)**.

---

## Approximation I: Markov Chain Monte Carlo (MCMC)

Rather than computing $p(t \mid \mathcal{D})$ analytically, MCMC takes a different approach: **draw samples from it**. Once you have samples $t^{(1)}, t^{(2)}, \ldots, t^{(S)}$ from the posterior, you can approximate the PPD as a simple Monte Carlo average:

$$p(y \mid x, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^{S} p(y \mid x, t^{(s)})$$

But how do you draw samples from a distribution you cannot compute? This is where Markov chains come in.

### Constructing the Markov Chain

A Markov chain is a rule for **randomly jumping from one value of $t$ to the next**, where each new value depends only on the current one. Starting from some initial $t^{(1)}$, you generate a sequence:

$$t^{(1)} \to t^{(2)} \to t^{(3)} \to \cdots$$

The key insight is that you can carefully **design the jumping rule** so that the chain has a special property: if you run it long enough, the distribution of $t^{(s)}$ converges to $p(t \mid \mathcal{D})$. This is called the **stationary distribution** of the chain.

The most common way to guarantee this is the **detailed balance condition**:

$$p(t)\, T(t \to t') = p(t')\, T(t' \to t)$$

Intuitively: the flow of probability from $t$ to $t'$ must equal the flow back. Any transition rule satisfying this will have $p(t \mid \mathcal{D})$ as its stationary distribution.

The famous **Metropolis-Hastings** algorithm constructs exactly such a rule. And crucially, it only requires evaluating $p(\mathcal{D} \mid t) \cdot p(t)$, the numerator of Bayes' theorem, so you never need to compute the intractable $p(\mathcal{D})$.

### Running the Chain Long Enough

"Running the chain long enough" bundles two distinct concerns:

**1. Burn-in.** The chain starts at an arbitrary $t^{(1)}$. Early samples are heavily influenced by this starting point, not the true posterior. The chain needs time to drift toward high-probability regions and "forget" where it started. The standard practice is to **discard the first $B$ samples**, this warm-up period is called the burn-in.

**2. Mixing.** Even after burn-in, consecutive samples $t^{(s)}$ and $t^{(s+1)}$ are not independent, each is a small perturbation of the previous one. If the posterior has multiple separated modes, the chain can get **stuck** in one region for a very long time before discovering another:

```
Posterior has two modes A and B:

Slow-mixing chain:  AAAAAAAAAAAAAAAA ... BBBBBBBB
Fast-mixing chain:  AABABBAABBAABAB
```

To get reliable estimates, you need enough samples that the chain has visited all regions of the posterior proportional to their true probability mass.

### Multiple Chains and Diagnostics

In practice, people often run several chains in parallel, each starting from a different initialisation. If all chains converge to the same distribution, that is a good sign of mixing. The most widely used diagnostic is the **Gelman-Rubin statistic $\hat{R}$**, which compares variance *within* each chain to variance *between* chains. An $\hat{R} \approx 1$ means the chains agree and have likely converged.

That said, multiple chains are a **diagnostic tool, not a guarantee**, if all chains are trapped in the same mode, they will agree with each other while missing the rest of the posterior entirely.

### NUTS: Making MCMC Faster

Vanilla random-walk proposals are inefficient, they take small, undirected steps and mix slowly in high dimensions. The **No-U-Turn Sampler (NUTS)** improves this dramatically by using **gradient information**, specifically, the gradient of $\log p(t \mid \mathcal{D})$ with respect to $t$, to make larger, informed jumps in the right direction, similar to how gradient descent navigates a loss surface. NUTS is the default sampler in probabilistic programming systems like Stan and PyMC.

**The bottom line on MCMC:** it is asymptotically exact: given enough samples, it converges to the true posterior. But in high dimensions, mixing can be extremely slow, making MCMC impractical for large neural networks.

---

## Approximation II: Variational Inference (VI)

Variational Inference takes a fundamentally different approach. Rather than sampling, it **reframes posterior inference as an optimisation problem**.

The idea: pick a family of simple, tractable distributions $q_\phi(t)$, for example, a fully factorised (mean-field) Gaussian, where each dimension of $t$ is modelled independently. Then find the member of that family that is **closest to the true posterior** $p(t \mid \mathcal{D})$:

$$q^*_\phi(t) = \arg\min_\phi\ \text{KL}\!\left(q_\phi(t) \,\|\, p(t \mid \mathcal{D})\right)$$

Here $\text{KL}$ is the Kullback-Leibler divergence, a measure of how different two distributions are. Minimising it pushes $q_\phi$ to match the shape of the posterior as closely as possible within the chosen family.

Once you have found $q^*_\phi$, you use it as your posterior approximation and plug it into the PPD:

$$p(y \mid x, \mathcal{D}) \approx \int p(y \mid x, t)\, q^*_\phi(t)\, dt$$

### The ELBO

In practice, you cannot directly minimise the KL divergence because it still involves $p(\mathcal{D})$, the same intractable normaliser we started with. Instead, VI maximises a proxy objective called the **Evidence Lower BOund (ELBO)**:

$$\text{ELBO}(\phi) = \mathbb{E}_{q_\phi(t)}\!\left[\log p(\mathcal{D} \mid t)\right] - \text{KL}\!\left(q_\phi(t) \,\|\, p(t)\right)$$

The first term rewards $q_\phi$ for placing mass on values of $t$ that explain the data well. The second term penalises $q_\phi$ for straying too far from the prior. Maximising the ELBO is equivalent to minimising the KL divergence to the posterior, and it can be optimised with standard gradient-based methods.

### The Trade-off

VI is much faster than MCMC; it reduces to gradient descent. But it comes with a fundamental limitation: the quality of the approximation is **bounded by the expressiveness of the chosen family $q_\phi$**. If the true posterior is multimodal or has complex correlations between dimensions, a mean-field Gaussian will fail to capture that structure, no matter how well you optimise it. This bias is baked in by design.

---

## MCMC vs VI: A Summary

|  | **MCMC** | **Variational Inference** |
|---|---|---|
| **Approach** | Sampling | Optimisation |
| **Accuracy** | Asymptotically exact | Biased by choice of $q$ family |
| **Speed** | Slow | Fast |
| **Scales to large models?** | Poorly | Yes |
| **Handles multimodality?** | In principle (slow mixing) | Often fails |

---

## Conclusion

The Bayesian framework gives us a coherent way to reason under uncertainty: encode prior beliefs, observe data, update to a posterior. The posterior is not just a mathematical curiosity, it is the engine behind principled predictions via the PPD, and the foundation of Bayesian Neural Networks, which offer something standard neural networks fundamentally cannot: calibrated, decomposable uncertainty estimates.

The catch is that computing the posterior exactly is almost always intractable. MCMC solves this by sampling, constructing a Markov chain that explores the posterior faithfully, at the cost of being slow. VI solves this by optimising, fitting a simple distribution to approximate the posterior, trading some accuracy for speed and scalability.

Neither is perfect, and much of modern Bayesian deep learning research is focused on finding better, faster, and more accurate ways to approximate the posterior predictive distribution, which is precisely the problem that motivates a great deal of active work in this space.