# Reading the Python implementation

Start with [`examples/quickstart_example.py`](../examples/quickstart_example.py)
for a small runnable path. It generates trajectories, alternates critic fitting
and policy improvement, then evaluates the returned policy. The paper settings
come from [`configs/paper_simulation.yaml`](../configs/paper_simulation.yaml).
The sections below map mathematical symbols to Python names and array shapes.

## From mathematical objects to numerical code

| Object | Experimental implementation | Interpretation |
| --- | --- | --- |
| A functional action | Values on a 128-point grid; a learned policy uses 12 cubic B-splines | The simulator and critics work with a finite numerical representation of a function. |
| Integration within a decision | RK4 integration for dynamics; quadrature for action projections and distances | The integration grid is part of the experiment configuration. |
| A fitted critic | AdaFNN training or subject-grouped Nyström KRR | Network optimization, landmark rank, and ridge selection specify the regression procedure. |
| Policy improvement | CMA-ES for AdaFNN; restarted Adam for KRR | Finite optimization budgets and objective floors do not certify a global maximizer. |
| Policy regularity | Empirical integrated squared curvature | Its connection to the theory's policy norm is conditional; the penalties are not identical. |
| Error-transfer diagnostics | One adjacent-critic difference on logged and learned-policy designs | These summaries do not estimate the uniform population coverage constant. |

The [experiment description](experiments.md) gives the numerical settings;
its [error-transfer discussion](experiments.md#relation-to-theoretical-error-transfer)
explains the distinction between theoretical errors and the empirical diagnostic.

## Data and environment

1. [`data.py`](../src/functional_fitted_q/data.py) stores the logged tuples and
   selects nested prefixes of whole subjects. A row is a transition, so retain
   `subject_ids` when splitting data.
2. [`policies/behavior_gp_policy.py`](../src/functional_fitted_q/policies/behavior_gp_policy.py)
   draws the logged actions. A subject shares one GP component across decisions;
   a fresh component is drawn per decision. The reference controller enters in
   latent, inverse-tanh coordinates.
3. [`env/dynamics.py`](../src/functional_fitted_q/env/dynamics.py) is the scalar
   Pendulum implementation; [`env/batched.py`](../src/functional_fitted_q/env/batched.py)
   performs the same updates for many subjects. Each MDP step integrates a
   complete torque function over normalized time $`u\in[0,1]`$. Transition noise
   is applied after the deterministic integration and reward calculation.

### Data and array shapes

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

## FQI and paper training entry points

For AdaFNN, read
[`runners/train_adafnn.py`](../src/functional_fitted_q/runners/train_adafnn.py),
then [`algorithms/adafnn_ffqi.py`](../src/functional_fitted_q/algorithms/adafnn_ffqi.py).
Each iteration fixes Bellman labels from the previous clipped critic/policy,
fits the next critic, then improves the policy using training next states.
Checkpoint callbacks preserve partially completed critic and policy updates.

`targets` stores the Bellman labels. `forward_unclipped` (AdaFNN) and
`predict_torch(..., clip=False)` (KRR) expose the raw fit; default deployed
predictions include clipping. See the [Bellman equations](experiments.md#fitted-q-iteration).

The paper's KRR command enters
[`runners/train_krr.py`](../src/functional_fitted_q/runners/train_krr.py), which
contains its training loop, grouped ridge selection, coefficient conversion,
and restart handling. Follow
[`runners/krr_gradient.py`](../src/functional_fitted_q/runners/krr_gradient.py)
for Adam updates and
[`runners/krr_restart.py`](../src/functional_fitted_q/runners/krr_restart.py)
for the restart logic. The reusable
[`algorithms/nystrom_ffqi.py`](../src/functional_fitted_q/algorithms/nystrom_ffqi.py)
provides a separate checkpointed FQI loop; it is not the paper command's main
loop.

## Critics

- [`critics/adafnn_functional_critic.py`](../src/functional_fitted_q/critics/adafnn_functional_critic.py):
  `action_scores` integrates actions against learned basis functions;
  `forward_unclipped` combines the scores and state coordinates;
  `forward` clips the deployed prediction to $`[0,V_{\max}]`$.
- [`critics/nystrom_functional_critic.py`](../src/functional_fitted_q/critics/nystrom_functional_critic.py):
  quadrature-weighted action distances enter a product Gaussian kernel;
  whitening produces a fixed landmark feature map, and `alpha` stores its
  primal regression coefficients. The normal equation adds $`n\varrho I`$
  to the feature Gram matrix.
- [`critics/nystrom_grouped_cv.py`](../src/functional_fitted_q/critics/nystrom_grouped_cv.py):
  entire subjects stay in one fold. All ridge candidates share the same labels
  and feature design. The selected ridge is refit on the full design.
  `gcv_selection.selected_lambda` is the critic ridge, not the policy coefficient.

## Policy and curvature

[`policies/bspline_policy.py`](../src/functional_fitted_q/policies/bspline_policy.py)
defines the seven state features and the $`7\times p_u`$ parameter matrix.
Use `BoundedCoefficientBSplinePolicy` to follow the paper: tanh bounds spline
coefficients before constructing the curve. The separate `BSplinePolicy`
clips the curve pointwise and is not used for the paper's curvature-penalized fits.

[`algorithms/theory_total_curvature.py`](../src/functional_fitted_q/algorithms/theory_total_curvature.py)
constructs $`W_{p_u}`$ by exact two-node quadrature on each cubic-spline knot span.
`torch_uncentered_total_curvature` accepts arrays ending in `(states, p_u)`;
leading dimensions can index multiple policy candidates.
`raw_total_curvature_penalty` stores the uncentered average before applying
a coefficient; centered covariance curvature is a separate diagnostic.
Training uses the exact per-knot-span matrix, while the convenience
`roughness()` method uses grid quadrature.

[`algorithms/dimensionless_total_curvature.py`](../src/functional_fitted_q/algorithms/dimensionless_total_curvature.py)
combines fitted value and curvature for AdaFNN. It evaluates curvature on all
training next states even when the Q objective uses a subset. Its constant and
zero-action candidates provide an objective floor, not a proof of global
optimization. The separately fitted constant-action comparator is a different
experiment.

[`runners/krr_optimizer.py`](../src/functional_fitted_q/runners/krr_optimizer.py)
scores KRR policies in raw Q units. Its `policy_lambda` has already been
converted from the command-line dimensionless coefficient; see
[coefficient units](experiments.md#objective-and-coefficient-units).

### Coefficient names

| Name | Meaning |
| --- | --- |
| `--policy-lambda`, `lambda_dimensionless` | $`\lambda_{\Omega,n}/s_\Omega`$, the grid value |
| `equivalent_raw_lambda`, `effective_raw_lambda` | $`\lambda_{\Omega,n}=\texttt{lambda\_dimensionless}\,s_\Omega`$ |
| KRR `TheoryTotalCurvatureAdamConfig.policy_lambda` | The converted **raw** coefficient $`\lambda_{\Omega,n}`$ |
| `q_scale`, `omega_scale` | $`Q_{\rm scale}`$ and $`\Omega_{\rm scale}(p_u)`$ |
| Theory $`\lambda_{\pi,n}`$ | Multiplies the complete policy-RKHS norm squared |
| Critic ridge $`\varrho`$ | Regularizes the KRR regression fit |
| Diagnostic `representation_ridge_etas` | Controls regularization of feature-geometry diagnostics |

## Evaluation, selection, and diagnostics

[`evaluation.py`](../src/functional_fitted_q/evaluation.py) returns both raw and
normalized discounted returns. The same evaluation seed supplies common
random numbers; callers use different seed streams for tuning and reporting.
[`selection.py`](../src/functional_fitted_q/selection.py) accepts tuning returns
only and selects the best coefficient before independent reporting.

[`identification.py`](../src/functional_fitted_q/identification.py) computes
nearest-action distances, feature leverage, and adjacent-critic energies.
`D_actual_sq` uses every logged transition; `G_actual_sq` uses held-out policy
actions. Their ratio is the paper's $`\mathsf{ER}_h`$. The separate projected
quantities in
[`diagnostics/adafnn_representation.py`](../src/functional_fitted_q/diagnostics/adafnn_representation.py)
use representation ridge and do not replace the actual clipped difference.

[`analysis/statistics.py`](../src/functional_fitted_q/analysis/statistics.py)
resamples the 20 independent data/training seeds. Diagnostic query rows are
first summarized within each fit. The plotting and verification entry points
are documented in the [results guide](../results/README.md).
