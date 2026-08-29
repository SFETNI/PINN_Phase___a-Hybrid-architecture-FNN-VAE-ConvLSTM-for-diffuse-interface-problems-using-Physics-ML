# How PINN-Phase Learns from the Physics

## What you will learn

- how the explicit MPF operator creates a target on the model's own field;
- how the relative and absolute residual terms complement each other;
- how training differs from ordinary autonomous inference;
- why represented horizon and TBPTT window are different quantities.

## Prerequisites

Read Tutorials [02](02_explicit_mpf_operator.md) through
[04](04_one_pinn_phase_step.md). Basic familiarity with optimization is useful.

## The target and the prediction

At autoregressive step \(t\), the reported explicit-MPF training path evaluates
the physical operator on the model's current field. The model-field target
displacement is

$$
q_t=\Delta t_m R(\phi_t),
$$

and the neural displacement is

$$
p_t=\Delta\phi_t.
$$

Thus the model learns a time-integrator update whose direction and scale match
the explicit MPF evolution. It does not learn from a stored post-initial field as
the target for the next state.

## Mixed physical discrepancy

The accepted mixed per-step form is

$$
\ell_t=
w_{\mathrm{rel}}
\frac{\langle(p_t-q_t)^2\rangle}
     {\langle q_t^2\rangle+\varepsilon_r}
+w_{\mathrm{abs}}
\frac{\langle(p_t-q_t)^2\rangle}
     {s_{\mathrm{abs}}^2+\varepsilon_r}.
$$

- The **relative component** normalizes by the physical displacement energy.
- The **absolute component** uses a fixed displacement scale, so weak late-time
  dynamics do not become irrelevant merely because \(q_t\) is small.
- \(\varepsilon_r\) stabilizes both denominators.
- \(s_{\mathrm{abs}}\) is a fixed scale, not a per-step reference statistic.

Across a represented horizon of \(H\) autoregressive steps, the core concept is

$$
L_{\mathrm{res}}=\frac{1}{H}\sum_{t=0}^{H-1}\ell_t.
$$

The physical residual is dominant, but benchmark configurations can add or
reweight terms. The first step can receive additional weight; the provisional
state can receive the optional phase-sum penalty from Tutorial 04; and a branch
magnitude penalty can discourage one recurrent branch from overwhelming the
other. Exact weights belong in the accepted configuration records, not in a
universal tutorial formula.

## Advanced note: interface weighting

Some explicit-MPF configurations emphasize interface and junction pixels using
a detached weight \(w(x)\) derived from the **current model field**. A pixel is
classified as interfacial when its second-largest phase fraction exceeds the
configured threshold. The weighted squared-error numerator is conceptually

$$
\left\langle w(x)(p_t-q_t)^2\right\rangle.
$$

The target-energy denominator \(\langle q_t^2\rangle\) remains unweighted. This
changes spatial emphasis; it does not redefine the MPF operator or introduce a
post-\(t_0\) reference mask.

## Exact step order in the trainer

For each predicted step, the accepted trainer does the following:

```text
current model field phi_t
  -> evaluate q_t = dt_m R(phi_t)
  -> predict neural rate/increment and recurrent state
  -> compute physical discrepancy
  -> form provisional next field
  -> compute any enabled pre-map penalties
  -> apply the configured admissibility map for phi_(t+1)
```

Losses accumulate over a TBPTT window. At a window boundary the trainer applies
backpropagation and the optimizer update, then detaches the next field and
recurrent state before continuing. The state map is therefore part of the
autoregressive trajectory; it is not postponed until after the whole horizon.

## Training versus inference

Training uses physics to construct an objective:

```text
phi_t -> neural increment p_t
   \-> explicit MPF target q_t -> loss -> gradient/optimizer
phi_t + p_t -> admissibility map -> phi_(t+1)
```

Ordinary prediction is shorter:

```text
phi_t -> neural model -> bounded zero-sum increment
      -> state update -> admissibility map -> phi_(t+1)
```

During ordinary autonomous inference there is no physical target, loss,
gradient, or optimizer, and the explicit MPF right-hand side is not evaluated.
The physics has shaped the learned parameters during training; the deployed
step is the neural integrator plus its structural state operations.

## Reference isolation, stated precisely

For the reported explicit-MPF training path:

- the initial condition seeds training;
- the physical residual is evaluated on the model's own evolving field;
- post-\(t_0\) phase-field reference states do not create \(q_t\);
- post-\(t_0\) reference states do not enter the loss, early stopping, or
  checkpoint selection;
- reference trajectories are used later for evaluation.

The machine-readable scope and guard evidence are in
[`TRAINING_PATH_DISCLOSURE.json`](../TRAINING_PATH_DISCLOSURE.json). This wording
is intentionally scoped: generic utilities and the historical scalar feasibility
lineage have separate contracts.

## Represented horizon versus TBPTT

The training horizon \(H\) is the physical number of autoregressive steps whose
losses are represented. The TBPTT window is the number of recurrent steps kept
in one gradient graph before detachment.

**Illustrative schedule - not a benchmark configuration.** For \(H=16\) and a
TBPTT window of 4:

```text
steps:  0  1  2  3 | 4  5  6  7 | 8  9 10 11 | 12 13 14 15
detach:             ^             ^             ^            ^
```

All 16 steps contribute physical training signal, while no gradient graph spans
more than four recurrent steps.

## Scalar-family differences

The scalar family uses a related physics-only training framework but not this
exact explicit-MPF objective. Its accepted source supports energy-monotonicity
penalties, branch-target terms, and physics-only branch warm-up. Some scalar
ablation protocols also differ by grid size. These details are kept separate so
the explicit-MPF loss above is not presented as universal.

## What to remember

- The physical target is evaluated on the model's own evolving explicit-MPF field.
- Relative and fixed-scale absolute terms protect different dynamical regimes.
- Inference uses the learned step and state map, not the physical target machinery.
- Training horizon describes represented physics; TBPTT describes gradient memory.

## Next

Continue to [Symmetry, Phase Labels, and Model Families](06_symmetry_and_model_families.md).
