```{.python .input}
import matplotlib.pyplot as plt    # import matplotlib
import math                        # import basic math functions
import random                      # import basic random number generator functions

fig_w, fig_h = (6, 4)
plt.rcParams.update({'figure.figsize': (fig_w, fig_h)})
```

## Objectives

In this notebook we'll implement a *Leaky Integrate-and-Fire (LIF)* neuron with stochastic pre-synaptic input current, and visualize its key statistical properties.

## Background

This neuron model is defined by a *membrane equation* and a *reset condition*:


\begin{align*}
\\
&\tau_m\,\frac{d}{dt}\,V(t) = E_{L} - V(t) + R\,I(t) &\text{if }\quad V(t) \leq V_{th}\\
\\
&V(t) = V_{r} &\text{otherwise}\\
\\
\end{align*}


where $V(t)$ is the membrane potential, $\tau_m$ is the membrane time constant, $E_{L}$ is the leak potential, $I(t)$ is the pre-synaptic input current, $V_{th}$ is the firing threshold, and $V_{r}$ is the reset voltage.

&nbsp; 

We'll extract and visualize the mean, standard deviation and histogram of the following quantities:

* Pre-synaptic input current $I(t)$
* Membrane potential $V(t)$
* Output firing frequency $\lambda(t)$

The problem will be split in several steps:

* Discrete time integration of $V(t)$ under sinusoidal pre-synaptic input, and without spikes
* Stochastic pre-synaptic input
* Visualizing ensemble statistics
* Introducing output spikes

**EXERCISE 1**

Initialize the main simulation variables.

**Suggestions**
* Modify the code below to print the main simulation parameters

```{.python .input}
# insert your code here

# t_max = 150e-3   # second
# dt = 1e-3        # second
# tau = 20e-3      # second
# el = -60e-3      # volt
# vr = -70e-3      # volt
# vth = -50e-3     # volt
# r = 100e6        # ohm
# i_mean = 25e-11  # ampere

# print(t_max, dt, tau, el, vr, vth, r, i_mean)
```

**EXPECTED OUTPUT**

```
0.15 0.001 0.02 -0.06 -0.07 -0.05 100000000.0 2.5e-10
```

### ODE integration without spikes

The numerical integration of the membrane equation can be performed in discrete time, with a sufficiently small $\Delta t$. We start by writting the membrane equation. without taking the limit $\Delta t \to 0$ in the definition of the time derivative $d/dt\,V(t)$:


\begin{align*}
\\
\tau_m\,\frac{V\left(t+\Delta t\right)-V\left(t\right)}{\Delta t} &= E_{L} - V(t) + R\,I(t)\\
\\
\end{align*}


The value of membrane potential $V\left(t+\Delta t\right)$ can be expressed in terms of its previous value $V(t)$ by simple algebraic manipulation. For *small enough* values of $\Delta t$ this provides a good approximation of the continuous time integration.

**EXERCISE 2**

Compute the values of $V(t)$ between $t=0$ and $t=0.01$ with $V(0)=E_L$ and pre-synaptic input given by:


\begin{align*}
\\
I(t)=I_{mean}\left(1+\sin\left(\frac{2 \pi}{0.01}\,t\right)\right)
\\
\end{align*}


**Suggestions**
* Express $V\left(t+\Delta t\right)$ in terms of $V(t)$
* Initialize the membrane potential variable `v` to `el`
* Loop in the time variable `t` from `t=0` to `t=0.01` with time step `dt`
* At each time step
    * Compute the current value of `i`
    * Update the value of `v`
    * Print `v`
* Use `math.pi` and `math.sin` for evaluating $\pi$ and $\sin(\,)$, respectively

```{.python .input}
# insert your code here
```

**EXPECTED OUTPUT**

```
-0.05875
-0.056827768434634406
-0.05454755936753374
-0.05238136075378811
-0.05077756115073311
-0.049988683093196457
-0.04997398050390223
-0.05041410212407606
-0.0508322176632412
-0.050775338345444725
-0.050775338345444725
```

**EXERCISE 3**

Plot the values of $V(t)$ between $t=0$ and $t=t_{max}$ under the same conditions.

**Suggestions**
* Update end time of loop to `t_max`
* Replace printing command with plotting command (with keyword `'ko'`)

```{.python .input}
# insert your code here

```

**EXPECTED OUTPUT**

![](https://github.com/ccnss/ccnss2018_students/raw/master/module0/lif_neuron/figures/lif_3.png)

## Stochastic pre-synaptic input

From the standpoint of neurons, their pre-synaptic input is random. We'll improve the pre-synaptic input model by introducing random input with similar statistical properties:


\begin{align*}
\\
I(t)=I_{mean}\left(1+0.1\sqrt{\frac{t_{max}}{\Delta t}}\,\xi(t)\right)\qquad\text{with }\xi(t)\sim U(-1,1)\\
\\
\end{align*}


where $U(-1,1)$ is the uniform distribution with support $x\in[-1,1]$.

A random pre-synaptic input $I(t)$ results in a random time course $V(t)$.

**EXERCISE 4**

Plot the values of $V(t)$ between $t=0$ and $t=t_{max}$ with random input $I(t)$.

Initialize the (pseudo) random number generator to a fixed value to obtain the same random input across executions of the code (initial value `0` will reproduce the expected output). The function `random.seed()` initializes the random number generator, and `random.random()` generates samples from the uniform distribution between `0` and `1`.

**Suggestions**
* Use the function `random.seed()`to initialize the random number generator to `0`
* Use the function `random.random()` to generate the input at each timestep
* Repeat the execution several times to verify that $V(t)$ has a random time course

```{.python .input}
# insert your code here

```

**EXPECTED OUTPUT**

![](https://github.com/ccnss/ccnss2018_students/raw/master/module0/lif_neuron/figures/lif_4.png)
