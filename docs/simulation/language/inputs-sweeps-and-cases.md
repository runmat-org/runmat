# Inputs, sweeps and cases

Inputs expose the parameters that a study or reusable model permits callers to change. Sweeps bind
those inputs into named cases; they do not rewrite source declarations or mutate a model.

## Declare an input

```fea
study bracket_response {
    input service_load : force {
        default = 1 kN
        minimum = 0 N
        maximum = 5 kN
    }

    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.loaded_bracket(service_load = input.service_load)
    analysis static { structural static {} }
}
```

`default`, `minimum`, and `maximum` are optional. Without a default, a consuming root or case must
bind the input. Bounds have the same physical type as the input and are enforced before geometry or
solver work begins.

Inputs may belong to geometry, model, or study declarations. A study input can bind a geometry
parameter, a model parameter, or both. A geometry change correctly creates a different geometry and
mesh identity; it is not treated as a load-only solve.

## Product sweeps

```fea
sweep envelope over study.bracket_response {
    axis service_load {
        half = 0.5 kN
        nominal = 1.0 kN
        overload = 1.5 kN
    }
    axis damping {
        low = 0.01
        nominal = 0.02
    }
    combine = product
}
```

`product` creates the Cartesian product of all axis values. Case identities are derived from axis
and value labels in a stable canonical order, not from task completion order.

## Zipped sweeps

With `combine = zip`, every axis must use the same labels. Values with matching labels form one
case:

```fea
sweep paired over study.bracket_response {
    axis service_load {
        low = 0.5 kN
        high = 1.5 kN
    }
    axis damping {
        low = 0.01
        high = 0.04
    }
    combine = zip
}
```

## Explicit cases

Use case blocks when the parameter combinations are irregular or deserve domain names:

```fea
sweep qualification over study.bracket_response {
    case limit_load {
        service_load = 1.25 kN
        damping = 0.02
    }
    case proof_load {
        service_load = 1.5 kN
        damping = 0.03
    }
}
```

A sweep uses axes or explicit cases, never both. Every required study input must be bound after
defaults are applied. Unknown inputs, duplicate case names, out-of-range values, unit mismatches,
and unequal zip labels fail checking.

## Sweep outputs

Sweep outputs operate on each case's declared study outputs:

```fea
output peaks = collect(case.output.peak_displacement.value)
output governing = max(case.output.peak_displacement.value, over = cases)
output fields = retain_each(case.output.displacement)
```

`collect` retains compact values with case coordinates. `min` and `max` retain the governing value
and its case attribution. `retain_each` requests each large output object; use it deliberately
because field retention can dominate storage.

## Parallel execution

Independent sweep cases may execute concurrently locally or as cluster fan-out. That is different
from partitioning one numerical operator across workers. The sweep definition remains the same in
both placements, and results are restored to canonical case order before publication.
