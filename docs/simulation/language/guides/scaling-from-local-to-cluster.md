# Scale from local to cluster

First run the study locally and preserve its study-definition identity, mesh digest, and numerical
requirements. Placement should be the variable in this comparison, not the model.

For reusable project policy, bind an execution declaration:

```fea
execution qualification_cluster {
    placement = profile.engineering
    resources {
        workers = 2..32
        memory_per_worker <= 32 GiB
        accelerators = preferred
    }
    fanout {
        sweep_cases = 8
    }
}
```

Add `execution = execution.qualification_cluster` to the study. Project configuration maps the
profile to a cluster, queue, provider capabilities, and pinned trust identity.

For an explicit CLI placement:

```sh
runmat fea run bracket.fea \
  --study bracket_response \
  --cluster engineering \
  --project 00000000-0000-0000-0000-000000000000 \
  --trust-identity sha256:EXPECTED_IDENTITY \
  --workers 8
```

Use cluster fan-out for independent cases, frequencies, and result partitions. Request cooperative
operator partitioning only for a procedure with a qualified distributed solver path. The run
evidence must say which shape occurred, how work was partitioned, which backend each task used,
and whether any task restarted.

Compare local and remote semantic results, not wall time alone. Geometry, topology, property-map,
procedure, sample, field, and numerical identities should agree wherever placement is intended to
be neutral. Placement-specific resource and transport receipts should differ.
