# Parallel and distributed execution

Execution policy is optional. A local study with no resource constraints does not need an
`execution` declaration. Adding or changing placement does not change the physical meaning or
study-definition identity.

## Bind an execution policy

```fea
execution qualification_cluster {
    placement = profile.engineering
    backend = auto

    resources {
        workers = 2..64
        memory_per_worker <= 32 GiB
        accelerators = preferred
    }
}

study bracket_response {
    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.bracket
    execution = execution.qualification_cluster
    analysis static { structural static {} }
}
```

`profile.engineering` resolves through project configuration. Cluster addresses, credentials,
trust material, and queue implementation details do not belong in a study file.

`backend = auto` permits an admitted backend chosen from available capabilities. When a study
requests the accelerator and the accelerator path cannot run it (its solver has no device path,
no accelerator can be allocated, or the device declines the system at solve time), every physics
family runs on the CPU reference path: the result reports `cpu_reference` as its execution
backend and carries exactly one `FEA_ACCELERATOR_ADMISSION_FALLBACK` warning, in the published
study as well as on the command line. `accelerators = required` is the strict option: the study
fails closed with `RM.FEA.ACCELERATOR_REQUIRED` instead, before any assembly or solve whenever the
fallback is known in advance. The result records the actual backend and device evidence rather
than the preference alone. See [accelerator placement](../../fea/accelerator-placement.md).

## Three distinct kinds of scale

RunMat keeps these execution shapes separate:

1. **Local parallelism** runs independent tasks concurrently on one host.
2. **Cluster fan-out** schedules independent sweep cases, frequencies, modes, samples, or recovery
   partitions on separate workers.
3. **Cooperative distribution** partitions one numerical operator across a worker group with an
   explicit partition, halo, collective, failure, and checkpoint contract.

The first two do not constitute a distributed solve. Evidence and user-visible status use these
terms precisely.

## Memory per worker

`memory_per_worker <= 16 GiB` bounds the resident memory of each worker that runs the study.
Planning works within the effective ceiling: the smallest of this bound, the step's own resident
ceiling, and the memory the host admits. A linear static solid solve selects its path within that
ceiling. When the sparse Cholesky factor fits, the factor proves coercivity and solves. When it
doesn't fit, element rigidity proves the operator positive definite and AMG-preconditioned
conjugate gradients solve it, accepted by the true residual. Each published run records the
effective ceiling (`effective_memory_ceiling_bytes`) and the selected path (`static_solve_path`)
in its provenance, and `runmat fea run --json` prints both. Neither value enters the
study-definition identity.

## Fan-out policy

```fea
fanout {
    sweep_cases = 16
    frequency_samples = auto
}
```

The current typed runner consumes sweep-case fan-out and worker bounds. Frequency fan-out is part
of the checked policy surface, but not every procedure currently consumes the authored setting.
When a setting is not connected to an executable procedure, planning must not claim that the run
used it.

## Cooperative policy

```fea
cooperative {
    operator_partitions = 8
    partitioner = graph
    balance = nonzeros
}
```

This expresses permission and constraints for procedures with a qualified cooperative execution
path. It is not a request to split an arbitrary sparse matrix and hope the solver tolerates it.
Admission checks the procedure, formulation, backend, partitioner, communication topology,
resource envelope, and worker cohort before dispatch.

## Resilience policy

```fea
resilience {
    retries = 2
    checkpoint = required
}
```

Retries resume only from canonical safe boundaries. A retry never changes geometry, material,
procedure, convergence, or accuracy intent. Authored resilience fields are checked today, but only
procedures with a durable checkpoint implementation may claim resumed numerical work. Otherwise a
retry restarts from the latest valid generic boundary or fails admission.

## CLI placement

An explicit cluster request uses the same source study:

```sh
runmat fea run bracket.fea \
  --study bracket_response \
  --cluster engineering \
  --project 00000000-0000-0000-0000-000000000000 \
  --trust-identity sha256:EXPECTED_IDENTITY
```

Every explicit `--cluster` request requires `--trust-identity`; `--project` is a UUID. Remote-only
flags are rejected without `--cluster`. A configured placement profile may supply pinned trust and
routing without these CLI flags. An explicit CLI cluster overrides authored placement but cannot
weaken physical or accuracy requirements.

## Accuracy source authority

Portable accuracy tasks for linear structural, modal response-spectrum and random-response
results carry the producing workload contract and its canonical numerical workload and mesh
inputs. Preparation and workers bind those inputs to the numerical receipt before checking the
authored accuracy plan, formulation identity and observation coverage. A result catalog cannot
define a weaker acceptance plan or silently omit authored observations. Thermal tasks bind their
original thermal workload and mesh directly to the thermal receipt.

The binding is to canonical scientific content, not one transport chunk layout. Rechunking the
same payload does not change its producing workload identity. This source validation does not
establish convergence, validate a formulation, or prove measured memory bounds. In particular,
constraint compilation and observation reprojection still need independently bounded workspace
admission; the current workspace multiplier is not a proof for multiplicative expansion.

Constraint-rank classification now borrows the canonical equations and label strings. Its
workspace profile bounds coefficient fill and provenance by connected DOF components, with
checked slot arithmetic and fallible exact reservations. A count-only graph bound supports
admitting connectivity derivation before the tighter numerical workspace profile is available.
The same classifier serves compilation and classification validation; validation does not clone
the complete rows merely to recompute rank. The profile covers requested Rust heap slots, not
allocator overhead, process RSS, diagnostic strings, authored constraint expansion or observation
resolution. Whole-task admission must compose those remaining costs before claiming closure.
