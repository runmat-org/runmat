# Accelerator placement and CPU fallback

A study requests the accelerator with `backend = auto` (or `accelerator`) and
`accelerators = preferred | required` in its execution policy, or with
`fea.ComputeBackend.gpu()` in an `fea.ExecutionStep`. One shared admission decision
(`runmat_analysis_fea::accelerator_admission`) serves every physics family.

## Default: fall back to the CPU reference path

When the accelerator is requested and the accelerator path cannot run the solve, the family runs
on the CPU reference path and:

- reports `requested_backend = gpu` and `execution_backend = cpu` (`cpu_reference`);
- carries exactly one warning, `FEA_ACCELERATOR_ADMISSION_FALLBACK`, naming the family and the
  reason;
- produces the same numbers as a CPU request.

The fallback applies when:

| Family | Device solve path | Falls back when |
| --- | --- | --- |
| Thermal (linear, steady and transient) | yes | no accelerator provider is assigned to the worker, or the provider declines the system at solve time |
| Structural transient (direct) | yes | as for thermal |
| Thermal with radiation | no | always (the damped Newton solve runs on the shared CPU sparse layer) |
| Structural static, modal, buckling, frequency response, nonlinear | no | always |
| Acoustic, electromagnetic, fluid, conjugate heat transfer, fluid-structure interaction | no | always |

A study planned on the CPU before submission (the `runmat run` command admits one backend for
the whole sequence) prints the warning and records it in the published study's `diagnostics`.

## Strict placement: `accelerators = required`

`accelerators = required` in the execution policy, or `strict = true` in
`fea.ExecutionStep.new(step, backend, solverOptions, strict)` (an optional trailing argument that
defaults to `false`), selects the strict policy. A study
that cannot run on the accelerator fails closed with `RM.FEA.ACCELERATOR_REQUIRED`
(`FEA_ACCELERATOR_REQUIRED` in the solver message):

- before any assembly or solve whenever the fallback is known in advance (a family without a
  device path, radiation, no accelerator provider, or no allocatable accelerator);
- after the solve only for a decline the device reports at solve time; nothing is published.

Remote runs settle the same policy: the submitting host admits placement before submission and
the published result is settled against the policy before anything is written.
