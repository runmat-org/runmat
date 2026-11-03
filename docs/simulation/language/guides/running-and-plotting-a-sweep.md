# Run and plot a sweep

Start with bounded study inputs, then create a named sweep over that study:

```fea
sweep load_envelope over study.bracket_response {
    axis service_load {
        half = 0.5 kN
        nominal = 1.0 kN
        proof = 1.5 kN
    }
    combine = product

    output peak_by_case = collect(case.output.peak_stress.value)
    output governing = max(case.output.peak_stress.value, over = cases)
}
```

Check the sweep root and inspect its case count:

```sh
runmat fea check bracket.fea --sweep load_envelope
runmat fea plan bracket.fea --sweep load_envelope
runmat fea run bracket.fea --sweep load_envelope
```

Cases may finish in any order; publication restores canonical case order and retains stable case
coordinates. Plot the collected compact values for trends. Retain whole fields only when comparison
needs them:

```fea
output stress_fields = retain_each(case.output.stress)
```

`retain_each` can multiply storage by the case count. Prefer attributed extrema or bounded sampled
outputs when the engineering question does not require every field page.

Local execution runs independent cases concurrently within admitted worker and memory bounds.
Cluster placement can fan them out without changing study or case identity.
