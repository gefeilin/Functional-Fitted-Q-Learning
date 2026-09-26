# Paper-to-code notation

This guide connects the paper's notation to the implementation: logged data,
Bellman updates, policy regularization, value evaluation, and critic diagnostics.
Mathematical symbols follow the paper. The tables map them to Python arguments
and saved-result columns.

## Data and array shapes

The logged data are

```math
\mathcal D=\{(S_{i,t},A_{i,t},R_{i,t},S_{i,t+1}):
1\le i\le N,\ 1\le t\le T\},\qquad n=NT.
```

Python flattens the subject/time indices into rows and stores time indices
starting at zero. $`S_j^+`$ denotes the next state of flattened record $`j`$;
it is not a new independent sample. The diagnostic states $`S_j^{\rm ev}`$ come
from separate behavior trajectories, with $`q=n`$ records in the paper design.

| Paper quantity | Python / saved field | Meaning and shape |
| --- | --- | --- |
| $`N`$ | `n_subjects` | Independent subjects; paper grid: 100, 200, 400, 800, 1600 |
| $`T`$ | `t_per_subject` | Decisions per subject; 20 |
| $`n=NT`$ | `n_transitions`, `len(dataset.states)` | Logged transitions |
| $`S_{i,t}`$ | `states` | Rows of $`(\cos\theta,\sin\theta,\omega)`$, shape `(n, 3)` |
| $`A_{i,t}(u)`$ | `action_values` | Sampled torque functions, shape `(n, 128)` in paper fits |
| $`R_{i,t}`$ | `rewards` | Reward vector, shape `(n,)`, bounded in $`[0,1]`$ |
| $`S_{i,t+1}`$ | `next_states` | Shape `(n, 3)` |
| $`i,t`$ | `subject_ids`, `time_indices` | Keep trajectory membership when splitting or subsampling |
| $`m,M`$ | `iteration`, `max_iterations` | FQI update number and total updates; $`M=20`$ |
| $`\gamma`$ | `gamma` | Discount, 0.95 |
| $`V_{\max}`$ | `value_max` | Prediction clipping bound, $`(1-\gamma)^{-1}=20`$ |
| $`q`$ | `query_count` | Number of independent-cohort diagnostic records; $`q=n`$ |
| $`p_u`$ | `policy.k`, `basis_dimension` | Number of policy B-splines; 12 |
| $`k`$ in $`\mathcal N_k(s)`$ | `state_neighbor_count` | Number of nearby training states; 32 by default |

The 128 action coordinates, 128 RK4 substeps, 12 policy B-splines, four learned
AdaFNN basis functions, and Nyström landmark rank are distinct dimensions.
RK4 samples actions at endpoints and midpoints, using 257 values per macro-step.
See [`data.py`](../src/functional_fitted_q/data.py) and
[`env/batched.py`](../src/functional_fitted_q/env/batched.py).

## Bellman labels and critics

Algorithm 1 uses

```math
Y_{i,t,m}=R_{i,t}+\gamma\widehat Q_{m-1}
\{S_{i,t+1},\widehat\pi_{m-1}(S_{i,t+1})\},\qquad
\widehat Q_0=0,
```

```math
\widehat Q_m=\mathrm{clip}_{[0,V_{\max}]}(\widetilde Q_m).
```

`targets` holds $`Y_{i,t,m}`$. `forward_unclipped` (AdaFNN) and
`predict_torch(..., clip=False)` (KRR) expose the raw fit. Default deployed
predictions include clipping. The first iteration therefore regresses rewards
without a continuation term. The continuing Pendulum has no terminal-mask term.

The KRR appendix denotes critic ridge by $`\varrho`$:

```math
\frac1n\sum_{i,t}\{Y_{i,t,m}-f(S_{i,t},A_{i,t})\}^2
+\varrho\|f\|_{\mathcal F}^2.
```

In the whitened Nyström approximation, `alpha` stores the **primal feature
coefficient vector**, and the normal equation adds $`n\varrho I`$ to the feature
Gram matrix. The code calls the selected ridge `selected_lambda` inside
`gcv_selection`; it is not the policy coefficient. Fivefold CV holds whole
subjects together, freezes the labels across candidates, and uses unclipped
prediction errors. It breaks exact ties toward the **larger critic ridge**;
policy-return selection breaks ties toward the **smaller policy coefficient**.

AdaFNN's action scores approximate the paper's functional inner products
$`\int_0^1a(u)\beta_j(u)\,du`$ using trapezoidal weights. The learned basis
functions change during fitting; they are distinct from the fixed policy
B-splines. See the [critic source guide](code-guide.md#critics).

## Policy objective and coefficient units

For the implemented bounded-coefficient policy,

```math
\pi_C(s)(u)=B_{p_u}(u)^\top c_C(s),\qquad
c_C(s)=2\tanh\{C^\top z(s)\},\qquad C\in\mathbb R^{7\times p_u}.
```

Nonnegative B-splines sum to one, so coefficients in $`[-2,2]`$ keep the entire
action in $`[-2,2]`$. This is the paper path implemented by
`BoundedCoefficientBSplinePolicy`; the separate `BSplinePolicy` clips the
curve pointwise and is not used for the paper's curvature-penalized fits.

The uncentered empirical curvature is

```math
W_{p_u}=\int_0^1 B_{p_u}''(u)B_{p_u}''(u)^\top du,\qquad
\widehat\Omega_{\mathcal D}(\pi_C)
=\frac1n\sum_{j=1}^n c_C(S_j^+)^\top W_{p_u}c_C(S_j^+).
```

The stored `raw_total_curvature_penalty` is this average before multiplication
by a coefficient. Centered covariance curvature is only a separate diagnostic.
The exact per-knot-span matrix in `total_curvature_matrix` is used for training;
the policy's convenience `roughness()` method uses grid quadrature instead.

The fitted-Q average uses a fixed subset $`\mathcal I_\Phi`$ of next states,
whereas the curvature term uses every next state:

```math
\widehat\Phi_m(\pi_C)=\frac1{|\mathcal I_\Phi|}
\sum_{j\in\mathcal I_\Phi}\widehat Q_m\{S_j^+,\pi_C(S_j^+)\}.
```

Define

```math
Q_{\rm scale}=(1-\gamma)^{-1},\qquad
\Omega_{\rm scale}(p_u)=4p_u\lambda_{\max}(W_{p_u}),\qquad
s_\Omega=Q_{\rm scale}/\Omega_{\rm scale}(p_u).
```

The factor $`4p_u`$ bounds $`\|c_C(s)\|_2^2`$. These scales depend on the spline
basis and discount, not on the observed curvature or training seed. Then

```math
\mathcal J_{\lambda_{\Omega,n},m}(C)
=\frac{\widehat\Phi_m(\pi_C)}{Q_{\rm scale}}
-\frac{\lambda_{\Omega,n}}{s_\Omega}
 \frac{\widehat\Omega_{\mathcal D}(\pi_C)}{\Omega_{\rm scale}(p_u)}.
```

| Name | Meaning |
| --- | --- |
| `--policy-lambda`, `lambda_dimensionless` | $`\lambda_{\Omega,n}/s_\Omega`$, the grid value |
| `equivalent_raw_lambda`, `effective_raw_lambda` | $`\lambda_{\Omega,n}=\texttt{lambda\_dimensionless}\,s_\Omega`$ |
| KRR `TheoryTotalCurvatureAdamConfig.policy_lambda` | The converted **raw** coefficient $`\lambda_{\Omega,n}`$ |
| `q_scale`, `omega_scale` | $`Q_{\rm scale}`$ and $`\Omega_{\rm scale}(p_u)`$ |
| Theory $`\lambda_{\pi,n}`$ | Multiplies the complete policy-RKHS norm squared |
| Critic ridge $`\varrho`$ | Regularizes the KRR regression fit |
| Diagnostic `representation_ridge_etas` | Controls regularization of feature-geometry diagnostics |

Multiplying $`\mathcal J`$ by $`Q_{\rm scale}`$ gives
$`\widehat\Phi_m-\lambda_{\Omega,n}\widehat\Omega_{\mathcal D}`$.
AdaFNN scores the dimensionless form; KRR's Adam path scores the latter raw
form. They have the same maximizers, but different objective units. This does
not imply identical finite-step optimizer paths.

The theory uses $`\lambda_{\pi,n}\|\pi\|_{\mathcal H}^2`$.
The experiment substitutes empirical total curvature to control within-action
regularity. The paper provides a spectral-envelope connection under stated
conditions, not an identity between these two penalties.

## Value and empirical diagnostics

The paper defines infinite-horizon value
$`J(\pi)=\mathbb E^\pi\sum_{t\ge0}\gamma^tR_t`$.
`J_normalized_mean` estimates its truncated normalized counterpart:

```math
\widehat J_{\rm normalized}
=\frac{1-\gamma}{1000}\sum_{e=1}^{1000}\sum_{t=0}^{99}\gamma^tR_{e,t}.
```

No curvature penalty is subtracted from reported return. For rewards in $`[0,1]`$,
the omitted normalized tail is at most $`\gamma^{100}`$.

For $`h=\widehat Q_{20}-\widehat Q_{19}`$, let

```math
\begin{aligned}
D_h^2&=\frac1n\sum_{i=1}^N\sum_{t=1}^T h(S_{i,t},A_{i,t})^2,\\
G_h^2&=\frac1q\sum_{j=1}^q
h\{S_j^{\rm ev},\widehat\pi_{20}(S_j^{\rm ev})\}^2.
\end{aligned}
```

The paper's critic-energy ratio is

```math
\mathsf{ER}_h=G_h^2/D_h^2.
```

The two energies are named here to map `D_actual_sq` and `G_actual_sq` to
`actual_graph_design_ratio`. These columns use clipped critic differences.
All selected paper fits have positive logged-design energy; no epsilon is
added to the denominator. Plots summarize per-fit ratios by their median.
A median of ratios generally differs from a ratio of median energies.
Feature-projected ratios have their own regularized denominator and must not
be substituted for $`\mathsf{ER}_h`$.

## Theoretical error transfer is a different object

In the coverage section, $`v=(f,f_y^\circ)`$ indexes fitted critics and their target
approximants in the declared family $`\mathcal V_n`$, and $`h_v=f-f_y^\circ`$.
The paper defines

```math
\begin{aligned}
e_v&=\|h_v\|_{L^2(\nu_X)},\\
\upsilon_v&=\left[\int\sup_{\pi\in\Pi_{\mathcal H}(B_{\mathcal H})}
|h_v\{s,\pi(s)\}|^2\nu_+(ds)\right]^{1/2}.
\end{aligned}
```

The supremum is **inside** the state integral. Under the directional-information
and smooth-evaluation assumptions, spectral transfer gives

```math
\upsilon_v\le B_{\psi,n}c_n^{-\vartheta/2}
\|d_v\|^{1-\vartheta}e_v^\vartheta.
```

The theory's $`\mathbf U_v`$ projects onto the feature span over the full
domain; its $`\mathbf W_v`$ is a positive spectral weight operator. Neither is
the empirical feature ridge matrix, and $`\mathbf W_v`$ is not the spline
curvature matrix $`W_{p_u}`$.

The experimental $`h`$ is one adjacent fitted-critic difference, evaluated at
one learned policy. It is not the fitted-versus-target $`h_v`$, and $`G_h^2`$ has
no policy supremum. Thus $`\mathsf{ER}_h`$ describes empirical transfer for this
one difference; it does not estimate the uniform population coverage constant
or verify the value theorem's assumptions.
