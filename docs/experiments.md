# Experiments

We study how functional-action policies perform as the offline sample grows,
and how critic updates behave at learned actions far from the logged actions.
AdaFNN results appear in the main text; the KRR counterparts are in the appendix.
The coefficient sweep at $`n=8{,}000`$ shows how policy regularization changes return.

This page describes the model and numerical settings. The main configuration
is `configs/paper_simulation.yaml`. See [the running guide](reproducibility.md)
for commands and [the result guide](../results/README.md) for figure inputs.

## Functional-action Pendulum

The state is $`S=(\cos\theta,\sin\theta,\omega)`$. At each decision the policy chooses
a torque function $`a(u)`$, $`u\in[0,1]`$, which acts for 0.5 physical time units. In
normalized action time, the deterministic dynamics are

```math
d\theta/du=0.5\omega,\qquad
d\omega/du=0.5\{15\sin\theta+3a(u)\}.
```

Mass and length are one, gravity is 10, torque is bounded by 2, and angular
velocity by 8. Fourth-order Runge–Kutta uses 128 substeps. The angle and velocity
noise scales are 0.03 and 0.05. The simulator wraps angles and enforces velocity
bounds. The macro-step reward is

```math
R=1-\frac{\int_0^1[\mathrm{wrap}(\theta(u))^2+
0.1\omega(u)^2+0.001a(u)^2]du}{\pi^2+6.4+0.004}.
```

Rewards are clipped to $`[0,1]`$ to handle numerical roundoff. The integration,
transition noise, and action interpolation are implemented in
`src/functional_fitted_q/env/`.

## Offline data and sample sizes

The behavior policy generates smooth torque functions:

```math
A_{i,t}(u)=2\tanh\{0.7 f_{\rm ref}(S_{i,t},u)+0.15G_i(u)+0.35H_{i,t}(u)\}.
```

$`G_i`$ and $`H_{i,t}`$ are unit-variance Matérn Gaussian processes with smoothness $`5/2`$ with lengthscales 0.5 and 0.25. $`G_i`$ is
shared across a subject's decisions; $`H_{i,t}`$ is drawn again at each decision. Actions
are represented on 128 coordinates. The separately calibrated reference
controller is held fixed; its coefficients are in `configs/reference_policy.yaml`.
Here $`f_{\rm ref}`$ is the **latent** reference signal: the implementation uses
$`\mathrm{arctanh}(\mathrm{clip}(a_{\rm ref}/2,-1+10^{-10},1-10^{-10}))`$,
rather than inserting the bounded reference torque directly into the formula.

For each seed 0–19, generate 2,500 subjects with $`T=20`$ decisions. Training uses
nested prefixes of $`N\in\{100,200,400,800,1600\}`$ subjects, giving
$`n=NT\in\{2000,4000,8000,16000,32000\}`$ transitions. Both approximators and the
constant-action comparator share each seed's data. Keep the full generation
pool when reproducing the experiment, since its size affects random-number order.

## Fitted Q-iteration

Each fit runs $`M=20`$ updates with discount $`\gamma=0.95`$. With
$`\widehat Q_0=0`$, the Bellman labels and deployed critic follow Algorithm 1:

```math
\begin{aligned}
Y_{i,t,m}&=R_{i,t}+\gamma\widehat Q_{m-1}
\{S_{i,t+1},\widehat\pi_{m-1}(S_{i,t+1})\},\\
\widehat Q_m&=\mathrm{clip}_{[0,V_{\max}]}(\widetilde Q_m),\\
V_{\max}&=(1-\gamma)^{-1}=20.
\end{aligned}
```

Here $`\widetilde Q_m`$ is the regression fit before prediction clipping. At the
first iteration the label is just the observed reward.

### AdaFNN critic

Four learned action-basis networks have architecture `(1,32,32,1)` with ReLU hidden
layers. Their integrals against the action use 128-point quadrature. The four
action projections and three state coordinates enter a ReLU network with
architecture `(7,256,256,256,1)`. The parameter bound is 1.

Adam uses learning rate 3e-4, batch size 256, weight decay 1e-5, and gradient
clipping at 10. Early stopping uses patience 30 and minimum improvement 1e-5.
Although `max_epochs` is 300, the loop continues when that stopping criterion
has not been met. It initially allows 1,200 epochs, then resumes with caps of
2,400, 4,800, and 9,600 without resetting the optimizer or changing the criterion.
If the criterion is still unmet, the fit stops with an error.

### Nyström KRR critic

The state kernel is Gaussian with coordinate lengthscales (1,1,2). The action
kernel is Gaussian in quadrature-weighted functional $`L^2`$ distance, with bandwidth
equal to the median distance among all unordered pairs of training actions.
The calculation uses disk-backed distances when needed.

Nyström ranks are 512, 1024, 1536, 2304, and 3584 for the five sample sizes.
Factorization and solves use float64, jitter 1e-10, relative eigenvalue tolerance
1e-12, and prediction blocks of 512.

At every FQI iteration, fivefold subject-grouped CV selects the critic ridge
from 81 logarithmically spaced values between 1e-12 and 1e4. The fold seed is
200000. All candidate ridge values use the same Bellman targets; each fold refits
the ridge coefficients. Scores average unclipped squared prediction errors at
the subject level. Policy returns are used to tune policy regularization, not
the critic ridge.

## Policy and curvature penalty

The functional policy uses $`p_u=12`$ cubic B-splines (the code calls this `k`):

```math
\pi_C(s)(u)=B_{p_u}(u)^\top c_C(s),\qquad c_C(s)=2\tanh\{C^\top z(s)\}.
```

The seven state features $`z(s)`$ are defined in
[`policies/bspline_policy.py`](../src/functional_fitted_q/policies/bspline_policy.py).
Writing $`w=\omega/8`$, they are
$`z(s)=(1,\sin\theta,\cos\theta,w,w\sin\theta,w\cos\theta,w^2)^\top`$.
The parameter matrix $`C`$ has shape $`7\times p_u`$; in batch code the row of
`2*tanh(features @ parameter_matrix)` is $`c_C(s)^\top`$.
Bounding the coefficients bounds the torque without clipping the spline
pointwise. A constant-action policy instead chooses one scalar torque per state
and holds it throughout the macro-step.

We penalize uncentered integrated squared curvature:

```math
W_{p_u}=\int_0^1B_{p_u}''(u)B_{p_u}''(u)^\top du,\qquad
\widehat\Omega_{\mathcal D}(\pi_C)
=\frac1n\sum_{j=1}^n c_C(S_j^+)^\top W_{p_u}c_C(S_j^+).
```

Two-node Gauss–Legendre quadrature on each knot span integrates this cubic-spline
curvature exactly up to roundoff. The average uses all $`n`$ training next states, indexed by $`S_j^+`$ after
flattening the subject and time indices. For a fixed subset $`\mathcal I_\Phi`$
of those states, define

```math
\widehat\Phi_m(\pi_C)=\frac{1}{|\mathcal I_\Phi|}
\sum_{j\in\mathcal I_\Phi}\widehat Q_m\{S_j^+,\pi_C(S_j^+)\}.
```

The dimensionless improvement objective in the experimental appendix is

```math
\mathcal J_{\lambda_{\Omega,n},m}(C)
=\frac{\widehat\Phi_m(\pi_C)}{Q_{\rm scale}}
-\frac{\lambda_{\Omega,n}}{s_\Omega}
 \frac{\widehat\Omega_{\mathcal D}(\pi_C)}{\Omega_{\rm scale}(p_u)},
```

```math
\begin{aligned}
Q_{\rm scale}&=(1-\gamma)^{-1}=20,\\
\Omega_{\rm scale}(p_u)&=4p_u\lambda_{\max}(W_{p_u}),\\
s_\Omega&=\frac{Q_{\rm scale}}{\Omega_{\rm scale}(p_u)}.
\end{aligned}
```

The grid is $`\lambda_{\Omega,n}/s_\Omega\in\{10^{-4},10^{-3},10^{-2},10^{-1},1\}`$.
The code and command-line argument `--policy-lambda` use this **dimensionless
ratio**, stored as `lambda_dimensionless`. Its raw curvature multiplier is

```math
\lambda_{\Omega,n}
=\texttt{lambda\_dimensionless}\,s_\Omega
=\texttt{lambda\_dimensionless}\,
  \frac{Q_{\rm scale}}{\Omega_{\rm scale}(p_u)}.
```

AdaFNN scores the dimensionless objective directly. The KRR Adam optimizer
scores $`\widehat\Phi_m-\lambda_{\Omega,n}\widehat\Omega_{\mathcal D}`$, which is
the same objective multiplied by the positive constant $`Q_{\rm scale}`$;
`train_krr.py` converts the coefficient before calling it. Objective values and
numerical stopping tolerances must be read in the respective units.

The experimental curvature penalty is a surrogate for the within-action
regularity controlled by $`\lambda_{\pi,n}\|\pi\|_{\mathcal H}^2`$ in the theory.
These penalties and their coefficients are not identical. The critic ridge and
representation-diagnostic ridge are separate again. See the
[notation guide](notation.md#policy-objective-and-coefficient-units).

AdaFNN policy improvement uses full-covariance CMA-ES with 3 restarts, population
64, objective budget 2,500, initial sigma 0.5, minimum sigma 1e-5, and candidate
batch size 8. The Q objective uses a fixed subset of up to 2,048 states; curvature
still uses all training next states. The optimizer also evaluates a state-dependent
constant policy and zero policy, keeping the best objective. The constant-action
comparator in Figure 1 is a separate fit restricted to constant actions at every
FQI iteration.

KRR policy improvement uses bounded Adam: up to 40,000 steps at rate 0.01,
followed when needed by a fresh-Adam restart of 20,000 steps at rate 0.001.
Stopping checks start after 100 steps, use a 25-step window, objective-span
tolerance 1e-5, action-update tolerance 1e-4, and finite perturbation probes.
The Q-objective subsets contain 1024, 2048, 3072, 4608, and 7168 states; curvature
uses all next states. Details are in `runners/krr_optimizer.py`,
`krr_restart.py`, and `krr_gradient.py`.

## Evaluation and uncertainty

For each approximator, sample size, and seed, train all five coefficients.
Select the largest mean tuning return from 1,000 episodes of horizon 100,
breaking exact ties toward the smaller coefficient. Evaluate the selected
policy on 1,000 independent reporting episodes. At $`n=8{,}000`$, report all five
coefficients as well.

Candidates and approximators share random numbers within each tuning or
reporting stream. The streams are independent, and the constant comparator
shares the reporting stream. We plot unpenalized normalized discounted return:

```math
\widehat J_{\rm normalized}=(1-\gamma)\frac1{1000}\sum_{e=1}^{1000}\sum_{t=0}^{99}\gamma^tR_{e,t}.
```

The curvature penalty affects training but is not subtracted from reported
return. The remaining normalized reward tail is at most $`\gamma^{100}`$, about 0.00592.

Pointwise 95% percentile intervals use 10,000 bootstrap resamples of the 20
training seeds. Value panels summarize means; locality and critic-ratio panels
summarize medians. Paired functional/constant differences resample the whole
five-sample-size trajectory for each seed. Resampling code is in
`analysis/statistics.py` and `analysis/plotting.py`.

## Critic diagnostics

For each fitted model, $`\widehat Q_{19}`$ and $`\widehat Q_{20}`$ are evaluated on an independent cohort of $`N`$
new subjects with $`T=20`$ decisions. The cohort therefore contains $`q=n`$ evaluation
states: 2,000, 4,000, 8,000, 16,000, or 32,000. The learned, previous, observed
behavior, and reference actions are evaluated at each state in this cohort.

At each query state, take the $`k=32`$ nearest logged states under

```math
d_S(s,s')^2=(s_1-s_1')^2+(s_2-s_2')^2+(s_3-s_3')^2/4.
```

Writing $`\mathcal N_k(s)`$ for their record indices, measure

```math
d_k(s,a)=\min_{(i,t)\in\mathcal N_k(s)}
\left[\int_0^1\{a(u)-A_{i,t}(u)\}^2du\right]^{1/2}.
```

Ties in state distance retain training-record order. The integral is computed
on the common action grid with trapezoidal weights. Compare learned-policy and held-out behavior actions.
The plot summarizes each fit's median distance, then the median across seeds.

For $`h=\widehat Q_{20}-\widehat Q_{19}`$, computed with the deployed clipped critics, define

```math
\begin{aligned}
D_h^2&=\frac1n\sum_{i=1}^N\sum_{t=1}^T h(S_{i,t},A_{i,t})^2,\\
G_h^2&=\frac1q\sum_{j=1}^q h\{S_j^{\rm ev},\widehat\pi_{20}(S_j^{\rm ev})\}^2.
\end{aligned}
```

```math
q=n=NT,\qquad \mathsf{ER}_h=\frac{G_h^2}{D_h^2}.
```

$`D_h^2`$ and $`G_h^2`$ name the two empirical energies for the code mapping;
the manuscript names their ratio $`\mathsf{ER}_h`$. Every logged transition enters $`D_h^2`$. We calculate the ratio within each fit and
then summarize across seeds, without adding an epsilon to the denominator.
The appendix also shows $`G_h^2`$ and $`D_h^2`$ separately. The CSV columns are
`G_actual_sq`, `D_actual_sq`, and `actual_graph_design_ratio`. The ratio plotted
is the median of per-fit ratios, not a ratio of the plotted median energies.

Additional diagnostics use AdaFNN's current final hidden features plus bias,
or whitened Nyström features for KRR. Their design second moment is uncentered.
Feature leverage is divided by the median held-out behavior leverage within
the corresponding fit and evaluation cohort. Actual and feature-projected critic
differences are stored separately because these representations need not capture
a nonlinear or clipped critic difference exactly.

AdaFNN restores the two saved critic parameter states directly. KRR restores
the saved $`\widehat Q_{19}`$ and $`\widehat Q_{20}`$ coefficient vectors and deterministically reconstructs the
Nyström feature design needed to evaluate those coefficients; it does not refit
Bellman targets, ridge coefficients, or policies.

Query rows are summarized within a fit before uncertainty is computed across the
20 data/training seeds. These plots describe finite-sample action locality and
realized critic updates. They illustrate the motivation for the theory; they do
not estimate true-Q error or verify a population identification bound.
