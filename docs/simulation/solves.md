# Studies, Solves, and Sweeps

A study is the complete placement-neutral definition of one analysis graph. It selects exact geometry, its matching solver mesh, a bound physical model, ordered procedures, and authored outputs. A sweep supplies typed input cases to one study. Execution policy is optional and does not change the physical definition.

## The Execution Pipeline

```text
typed source graph
  -> exact geometry realization and region lineage
  -> deterministic solver mesh and quality evidence
  -> property/constraint/load projection
  -> procedure workload graph
  -> local or cluster scheduling
  -> numerical results and receipts
  -> evaluated outputs and content-addressed publication
```

Every boundary has a canonical identity. Local and remote execution must return the same study definition, realized geometry, mesh, workloads, and numerical closure; placement, worker id, retry attempt, queue, and measured timing remain execution evidence rather than physical identity.

## Check

```sh
runmat check model.fea
runmat check model.fea --study static --json
```

Checking resolves the complete `.fea` module graph, exact `.m` imports, referenced geometry facts, and named execution-profile capabilities. It validates syntax, references, physical types/units, topology roles, model composition, procedure state flow, output expressions, sweep cases, and execution constraints. It neither meshes nor solves and does not write result evidence.

The result is `valid`, `invalid`, or `incomplete`. Incomplete means required project facts were unavailable; it is never executable success.

## Plan

```sh
runmat plan model.fea --study static
```

Planning reports the canonical logical phases and selected analysis definition. It is useful in code review and CI because it exposes what will be materialized and solved without performing the work.

## Run Locally

```sh
runmat run model.fea --study static --json
```

The native runner executes portable geometry, meshing, generated-input, and analysis tasks through the shared process-isolated task substrate. Independent generated-input chunks, sweep cases, and frequency samples may run concurrently. Procedures with state dependencies preserve their declared order and recovery closure.

## Run On A Cluster

```sh
runmat run model.fea \
  --cluster engineering \
  --project 7a7d4d37-f0ae-4a6b-a0f8-013a04e0a9d5 \
  --queue batch \
  --workers 16 \
  --trust-identity sha256:<endpoint-fingerprint> \
  --idempotency-key bracket-qualification-42
```

The client freezes one checked source/project revision, bundles content-addressed inputs once, binds authority once, and submits one study or sweep root. Workers receive exact program, object, resource, and capability contracts. Encrypted transport, retry, replacement, checkpoint/recovery, cancellation, and publication use the same infrastructure as other RunMat distributed work.

`--detach` returns after submission. Every explicit `--cluster` run requires `--trust-identity`, and `--project` is a UUID. `--workers 0` retains automatic or driver-only allocation behavior.

## Procedure State

Dependencies are explicit:

```fea
analysis modes {
    structural modal { modes = 30 }
    retain basis
}
analysis spectrum after analysis.modes {
    structural modal_response_spectrum {
        basis = analysis.modes.state.basis
        spectra = [model.response_spectrum.qualification_x]
    }
}
```

Prestress, modal bases, transient checkpoints, and other semantic states are named products. The compiler validates producer/consumer compatibility and rejects missing, stale, or ambiguous state. Restart and worker recovery restore the canonical numerical closure; an observation history is not treated as integrator state.

## Sweeps

A sweep is one atomic ordered root, not a client loop that submits unrelated jobs. Product, zipped, or explicit cases have stable identities. Independent cases may fan out across local workers or cluster allocations, while terminal results return in authored order with success, failure, cancelled, or indeterminate state retained per case.

Sweep outputs can collect attributed scalar values, retain each case's field, or select a governing case through typed operations. Failed cases cannot silently disappear from an envelope.

## GPU And Cooperative Distribution

Backend admission is explicit. A required accelerator request must match an allocated class and memory envelope; it never silently falls back to a host solve. GPU execution is advertised only for qualified shapes and hardware evidence.

Linear-static structural execution currently admits CPU solves only. A `Gpu` static solve request
fails before assembly, including when accelerator algebra is selected; its native sparse solver and
accelerator-bound resource admission are not yet qualified. A preferred placement policy may still
choose a CPU attempt, but that attempt is a CPU solve, not GPU execution.

Fan-out and cooperative distribution are different:

- fan-out schedules independent cases, frequencies, or generated-input chunks;
- cooperative execution partitions one operator/state across workers and requires exact ownership, communication, reduction, and restart contracts.

The execution policy can express both. The current typed runner consumes worker bounds and sweep-case fan-out; authored frequency fan-out, cooperative partitioning, and resilience fields remain checked contracts until their scheduler wiring and evidence gates are complete. A run is described as distributed only when one numerical problem actually uses the cooperative contract.

## Publication And Partial Access

Successful execution publishes a typed result manifest containing the compiled study, mesh and meshing evidence, exact resolved workloads, numerical receipts, semantic state, authored output bindings, and public run links. Large field data is chunked in the ordinary artifact store and can be paged by exact step, field, and sample identity.

Measured stage time, scheduler allocation, transfers, retries, and provider operations are factual run evidence. They do not alter the deterministic study-definition digest.

See [Study contracts](./study-contracts), [Evidence](./evidence), and [Trust](./trust).
