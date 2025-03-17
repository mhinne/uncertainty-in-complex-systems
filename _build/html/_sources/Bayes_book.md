# Bamojax

[Bamojax](https://doi.org/10.5281/zenodo.15038847) is the Bayesian modelling in JAX toolkit developed by our group. It provides an interface between Bayesian computational modelling and fast inference using the [Blackjax](https://blackjax-devs.github.io/blackjax/) software. 

## Quickstart

Here is a simple example of how to use **bamojax** for the famous eight-schools dataset:

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

We can then approximate the posterior, for example by using Blackjax' [Sequential Monte Carlo](https://osf.io/preprints/psyarxiv/swjtu_v2):

```python
# Set up Rosenbluth-Metropolis-Hastings within SMC:
mcmc_params = dict(sigma=4.0*jnp.eye(ES.get_model_size()))
rmh_kernel = mcmc_sampler(ES, mcmc_kernel=blackjax.normal_random_walk, mcmc_parameters=mcmc_params)

num_particles = 10_000
num_mutations = 100
num_chains = 4

engine = SMCInference(model=ES, num_chains=num_chains, mcmc_kernel=rmh_kernel, num_particles=num_particles, num_mutations=num_mutations)
result = engine.run(jrnd.PRNGKey(0))
```

The full worked example [can be found here](https://github.com/UncertaintyInComplexSystems/bamojax/blob/main/bamojax/examples/toy_examples/eight_schools.ipynb)!

More tutorials on **bamojax** will follow here later. For now, feel free to browse the examples at https://github.com/UncertaintyInComplexSystems/bamojax.