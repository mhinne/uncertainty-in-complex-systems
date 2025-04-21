# Bamojax

[Bamojax](https://doi.org/10.5281/zenodo.15038847) is the Bayesian modelling in JAX toolkit developed by our group. It provides an interface between Bayesian computational modelling and fast inference using the [Blackjax](https://blackjax-devs.github.io/blackjax/) software.


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

    
![png](_static/Bayes_book_files/Bayes_book_7_0.png)
    
More examples of **bamojax** will follow here soon. For now, feel free to browse the examples at https://github.com/UncertaintyInComplexSystems/bamojax!
