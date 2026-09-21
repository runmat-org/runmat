# Studies, Sweeps, and Execution

## Studies And Analyses

```fea
study structural_static_experimental {
    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.loaded_bracket
    require formulation = experimental

    analysis static {
        structural static {}
    }

    output displacement = analysis.static.field.node.displacement
    output von_mises = analysis.static.field.element.von_mises_stress
}
```

A study chooses one geometry instance, its matching mesh, one bound model, an explicit formulation requirement, and an ordered analysis graph. Dependencies use `after analysis.<id>`; retained state is named and consumed explicitly. Modal, transient, harmonic, response-spectrum, random-response, prestress, and buckling analyses therefore expose their state flow instead of hiding it in a template.

Study inputs are strongly typed and may have bounds and defaults:

```fea
input load : force {
    minimum = 0 N
    maximum = 2 kN
}
input damping : real = 0.02
```

## Sweeps

Axes can form a Cartesian product or a zip:

```fea
sweep envelope : study.response {
    axis load {
        half = 0.5 kN
        full = 1 kN
    }
    axis damping {
        low = 0.01
        nominal = 0.02
    }
    combine = product

    output peaks = collect(case.output.peak_displacement.value)
    output governing = max(case.output.peak_displacement.value, over = cases)
    output fields = retain_each(case.output.displacement)
}
```

For named scenarios, use explicit `case` blocks. Axis and case values are checked against the study input type, units, range, and required/default status. Cases have stable identities independent of scheduler completion order.

## Optional Execution Policy

Small local studies need no `execution` declaration. Add one only when the analysis needs placement or resource constraints:

```fea
execution scalable {
    placement = profile.engineering
    backend = accelerator
    resources {
        workers = 2..64
        memory_per_worker <= 32 GiB
        accelerators = preferred
    }
    fanout {
        sweep_cases = 16
        frequency_samples = auto
    }
    cooperative {
        operator_partitions = 8
        partitioner = graph
        balance = nonzeros
    }
    resilience {
        retries = 2
        checkpoint = required
    }
}
```

Bind it with `execution = execution.scalable` inside a study. A sweep inherits its study's policy. Project configuration resolves `profile.engineering` to cluster routing and accelerator facts; credentials, endpoint addresses, and queue details do not enter the study definition.

The current typed runner consumes worker bounds and sweep-case fan-out. Frequency fan-out, cooperative partition, and authored checkpoint/retry declarations are checked contracts but are not yet all wired as scheduler controls; they must not be treated as evidence that a run used those shapes. Fan-out of independent work and cooperative partitioning of one operator are distinct. RunMat never describes independent local tasks as a distributed solve, and it never silently replaces an admitted required accelerator solve with CPU execution.
