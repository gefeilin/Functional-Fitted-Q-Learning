# Experiments

We study how functional-action policies perform as the offline sample grows,
and how critic updates behave at learned actions far from the logged actions.
AdaFNN results appear in the main text; the KRR counterparts are in the appendix.
The coefficient sweep at n=8,000 shows how policy regularization changes return.

This page describes the model and numerical settings. The main configuration
is `configs/paper_simulation.yaml`. See [the running guide](reproducibility.md)
for commands and [the result guide](../results/README.md) for figure inputs.

## Functional-action Pendulum

The state is `(cos(theta), sin(theta), omega)`. At each decision the policy chooses
a torque function a(u), u in [0,1], which acts for 0.5 physical time units. In
normalized action time, the deterministic dynamics are

\[
d\theta/du=0.5\omega,\qquad
d\omega/du=0.5\{15\sin\theta+3a(u)\}.
\]

Mass and length are one, gravity is 10, torque is bounded by 2, and angular
velocity by 8. Fourth-order Runge–Kutta uses 128 substeps. The angle and velocity
noise scales are 0.03 and 0.05. The simulator wraps angles and enforces velocity
bounds. The macro-step reward is

\[
R=1-\frac{\int_0^1[\operatorname{wrap}(\theta(u))^2+
0.1\omega(u)^2+0.001a(u)^2]du}{\pi^2+6.4+0.004}.
\]

Rewards are clipped to [0,1] to handle numerical roundoff. The integration,
transition noise, and action interpolation are implemented in
`src/functional_fitted_q/env/`.

## Offline data and sample sizes

The behavior policy generates smooth torque functions:

\[
a_{it}(u)=2\tanh\{0.7 f_{\rm ref}(S_{it},u)+0.15G_i(u)+0.35H_{it}(u)\}.
\]

G and H are Matérn-5/2 Gaussian processes with lengthscales 0.5 and 0.25. G is
shared across a subject's decisions; H is drawn again at each decision. Actions
are represented on 128 coordinates. The separately calibrated reference
controller is held fixed; its coefficients are in `configs/reference_policy.yaml`.

For each seed 0–19, generate 2,500 subjects with T=20 decisions. Training uses
nested prefixes of N=100, 200, 400, 800, and 1,600 subjects, giving
n=2,000, 4,000, 8,000, 16,000, and 32,000 transitions. Both approximators and the
constant-action comparator share each seed's data. Keep the full generation
pool when reproducing the experiment, since its size affects random-number order.

## Fitted Q-iteration

Each fit runs M=20 updates with discount gamma=0.95. The regression target uses
the previous critic and policy. Deployed Q predictions are clipped to [0,20].

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
kernel is Gaussian in quadrature-weighted functional L2 distance, with bandwidth
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

The functional policy uses K=12 cubic B-splines:

\[
\pi_C(s)(u)=B(u)^Tc_C(s),\qquad c_C(s)=2\tanh\{z(s)^TC\}.
\]

The seven state features z(s) are defined in `policies/bspline_policy.py`.
Bounding the coefficients bounds the torque without clipping the spline
pointwise. A constant-action policy instead chooses one scalar torque per state
and holds it throughout the macro-step.

We penalize uncentered integrated squared curvature:

\[
W=\int_0^1B''(u)B''(u)^Tdu,\qquad
\widehat\Omega_D(\pi)=\frac1n\sum_j c_C(S_j^+)^TWc_C(S_j^+).
\]

Two-node Gauss–Legendre quadrature on each knot span integrates this cubic-spline
curvature exactly up to roundoff. The average uses all n training next states.
The implemented improvement objective is

\[
\widehat\Phi_m(\pi)/20-\ell\widehat\Omega_D(\pi)/\Omega_{\rm scale},
\qquad \Omega_{\rm scale}=4K\lambda_{\max}(W).
\]

The grid is ell in {0.0001, 0.001, 0.01, 0.1, 1}. The code calls ell
`lambda_dimensionless`; the manuscript writes `lambda_{Omega,n}/s_Omega`, where
`s_Omega=20/Omegascale`. The raw curvature multiplier is therefore
`ell*20/Omegascale`. This is separate from the critic ridge and the ridge used
in representation diagnostics.

AdaFNN policy improvement uses full-covariance CMA-ES with 3 restarts, population
64, objective budget 2,500, initial sigma 0.5, minimum sigma 1e-5, and candidate
batch size 8. The Q objective uses a fixed subset of up to 2,048 states; curvature
still uses all next states. The optimizer also evaluates a state-dependent
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
policy on 1,000 independent reporting episodes. At n=8,000, report all five
coefficients as well.

Candidates and approximators share random numbers within each tuning or
reporting stream. The streams are independent, and the constant comparator
shares the reporting stream. We plot unpenalized normalized discounted return:

\[
(1-\gamma)\frac1{1000}\sum_e\sum_{t=0}^{99}\gamma^tR_{e,t}.
\]

The curvature penalty affects training but is not subtracted from reported
return. The remaining normalized reward tail is at most gamma^100, about 0.00592.

Pointwise 95% percentile intervals use 10,000 bootstrap resamples of the 20
training seeds. Value panels summarize means; locality and critic-ratio panels
summarize medians. Paired functional/constant differences resample the whole
five-sample-size trajectory for each seed. Resampling code is in
`analysis/statistics.py` and `analysis/plotting.py`.

## Critic diagnostics

We retain Q19 and Q20 and generate an independent held-out behavior sample of
32 subjects with 17 decisions. Decision indices 1–16 are used. The paper plots
split B, comprising 16 subjects and 256 query states per action class.

At each query state, take the 32 nearest logged states under the scaled state
metric. Measure the minimum functional L2 distance from the query action to
their logged actions. Compare learned-policy and held-out behavior actions.
The plot summarizes each fit's median distance, then the median across seeds.

For h(s,a)=Q20(s,a)-Q19(s,a), computed with the deployed clipped critics, define

\[
D^2=\frac1n\sum_{(s,a)\in D}h(s,a)^2,\qquad
G^2=\frac1{256}\sum_{s\in B}h\{s,\pi_{20}(s)\}^2,\qquad
C_{\rm adj}=G^2/D^2.
\]

Every logged transition enters D². We calculate the ratio within each fit and
then summarize across seeds, without adding an epsilon to the denominator.
The appendix also shows G² and D² separately.

Additional diagnostics use AdaFNN's current final hidden features plus bias,
or whitened Nyström features for KRR. Their design second moment is uncentered.
Feature leverage is divided by the median held-out behavior leverage within
the same fit and split. Actual and feature-projected critic differences are
stored separately because these representations need not capture a nonlinear
or clipped critic difference exactly.

These plots describe finite-sample action locality and realized critic updates.
They illustrate the motivation for the theory; they do not estimate true-Q
error or verify a population identification bound.
