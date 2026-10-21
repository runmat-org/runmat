# Publish and read large results

Choose outputs that answer the study question without forcing every field into one process. A
field output publishes a manifest and independently addressable partitions; a reduction publishes
a compact attributed value plus evidence.

For a single peak, request the reduction in source:

```fea
output stress = analysis.static.field.element.von_mises_stress
output peak_stress = max(
    output.stress,
    over = geometry.bracket.region.body,
)
```

For a history, state the retained samples and bound:

```fea
output tip_history {
    value = sample(
        analysis.response.field.node.displacement,
        over = time,
        at = [0 s, 10 ms, 20 ms],
    )
    retention = explicit_samples
    maximum_samples = 3
}
```

After execution, select the result object or a typed field. Page reads are bounded; requesting one
sample does not materialize unrelated samples. Plot the selected field with ordinary `plot(field)`
and inspect the associated mesh with `mesh(field)`.

For each published object, retain the study, analysis, field, sample, mesh, workload, and numerical
receipt identities. A result is not complete merely because its data array exists. Verify terminal
status, formulation identity, convergence and balance, recovery method, partition completeness,
and source attribution before downstream use.
