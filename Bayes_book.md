# Bamojax

[Bamojax](https://doi.org/10.5281/zenodo.15038847) is the Bayesian modelling in JAX toolkit developed by our group. It provides an interface between Bayesian computational modelling and fast inference using the [Blackjax](https://blackjax-devs.github.io/blackjax/) software.


```python
:tags: [hide-input]
%load_ext autoreload
%autoreload 2

import os

SELECTED_DEVICE = '0'
print(f'Setting CUDA visible devices to [{SELECTED_DEVICE}]')
os.environ['CUDA_VISIBLE_DEVICES'] = f'{SELECTED_DEVICE}'
```

    Setting CUDA visible devices to [0]



```python
:tags: [hide-input]
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)

import jax.random as jrnd
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
import distrax as dx
import blackjax

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import sys

import bamojax
from bamojax.base import Model
from bamojax.samplers import gibbs_sampler, mcmc_sampler
from bamojax.inference import SMCInference

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
```

## Example 1: Eight schools

Here is a simple example of how to use **bamojax** for the famous eight-schools dataset. Here's how we set up a simple hierarchical model:


```python
means = jnp.array([28, 8, -3, 7, -1, 1, 18, 12])
stddevs = jnp.array([15, 10, 16, 11, 9, 11, 10, 18])

J = len(means)

ES = Model('eight schools')
mu = ES.add_node('mu', distribution=dx.Normal(loc=0, scale=10))
tau = ES.add_node('tau', distribution=dx.Transformed(dx.Normal(loc=5, scale=1), tfb.Exp()))
theta = ES.add_node('theta', distribution=dx.Normal, parents=dict(loc=mu, scale=tau), shape=(J, ))
y = ES.add_node('y', distribution=dx.Normal, parents=dict(loc=theta, scale=stddevs), observations=means)
```

After defining the model, we can approximate the posterior, for example by using Blackjax' [Sequential Monte Carlo](https://osf.io/preprints/psyarxiv/swjtu_v2).

Because this is a hierarchical model, it is efficient to use Gibbs sampling with SMC. **bamojax** automatically derives the correct sampling densities; we only need to provide it the step functions for the individual parameter updates, like so:


```python
gibbs_params = dict(mu=dict(sigma=6.0),
                    tau=dict(sigma=6.0),
                    theta=dict(sigma=5.0*jnp.eye(J)))
step_fns = dict(mu=blackjax.normal_random_walk, tau=blackjax.normal_random_walk, theta=blackjax.normal_random_walk)
gibbs_kernel = gibbs_sampler(ES, step_fns=step_fns, step_fn_params=gibbs_params)

num_particles = 10_000
num_mutations = 100
num_chains = 4

engine = SMCInference(model=ES, num_chains=num_chains, mcmc_kernel=gibbs_kernel, num_particles=num_particles, num_mutations=num_mutations)
result = engine.run(jrnd.PRNGKey(0))
```


```python
:tags: [hide-input]
fig, axes = plt.subplots(nrows=1, ncols=J, sharex=True, sharey=True, constrained_layout=True, figsize=(16, 2))
fig.set_facecolor((1,1,0,0))
fig.patch.set_alpha(0)

xlim = [-30, 60]
xrange = jnp.linspace(*xlim, 100)

for i, ax in enumerate(axes):    
    pdf = gaussian_kde(result['final_state'].particles['theta'][:, :, i].flatten())
    y = pdf(xrange)
    ax.plot(xrange, y, lw=1, zorder=1, color='k')
    ax.fill_between(xrange, y, jnp.zeros_like(y), color='tab:blue', alpha=0.5, zorder=0)
    ax.axvline(x=means[i], ls='--', color='k', lw=1)
    ax.set_ylim([0.0, 0.06])
    ax.set_xlim(*xlim)
    ax.set_xlabel(fr'$\theta_{i+1}$', fontsize=SMALL_SIZE)
    ax.set_facecolor((1,1,1,0))
    ax.patch.set_alpha(0)

axes[0].set_ylabel(r'$p(\theta\mid \mu, \sigma)$')

plt.suptitle('Eight schools hierarchical estimate', fontsize=MEDIUM_SIZE)
plt.show();
```


    
![png](_static/Bayes_book_files/Bayes_book_7_0.png)
    

