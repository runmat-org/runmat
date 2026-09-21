# Execution, parallel work, and distributed work

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

`backend = auto` permits an admitted backend chosen from available capabilities. A required
accelerator fails if no qualified device is available; `preferred` may admit a CPU attempt. The
result records the actual backend and device evidence rather than the preference alone.

## Three distinct kinds of scale

RunMat keeps these execution shapes separate:

1. **Local parallelism** runs independent tasks concurrently on one host.
2. **Cluster fan-out** schedules independent sweep cases, frequencies, modes, samples, or recovery
   partitions on separate workers.
3. **Cooperative distribution** partitions one numerical operator across a worker group with an
   explicit partition, halo, collective, failure, and checkpoint contract.

The first two do not constitute a distributed solve. Evidence and user-visible status use these
terms precisely.

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
