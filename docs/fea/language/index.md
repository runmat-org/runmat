# The `.fea` Language

A `.fea` source file declares engineering-analysis intent. The language has its own lossless parser, formatter, type checker, unit system, module graph, semantic diagnostics, and editor support. RunMat lowers a valid source graph into canonical geometry, meshing, model, procedure, output, sweep, and execution contracts before any expensive work starts.

The main declarations are:

| Declaration | Owns |
| --- | --- |
| `geometry` | imported, constructed, derived, or `.m`-generated exact geometry and named topology regions |
| `mesh` | solver-mesh family and meshing controls for one geometry |
| `material` | typed constitutive properties |
| `model` | material/property assignments, conditions, loads, signals, excitation cases, and reusable composition |
| `study` | one geometry, mesh, model, ordered analyses, inputs, outputs, and optional execution policy |
| `sweep` | an axis product, zipped axes, or explicit cases over one study |
| `execution` | optional placement, resource, fan-out, cooperative partitioning, and resilience preferences |

Imports are explicit and symbol-specific. There is no file-wide include, hidden template, implicit default model, lifecycle switch, or schema version preamble.

## Reference

- [Syntax, names, units, and modules](./syntax-and-modules)
- [Geometry, regions, meshes, materials, and models](./geometry-and-models)
- [Studies, analyses, sweeps, and execution](./studies-sweeps-and-execution)
- [RunMat functions, generated inputs, and outputs](./functions-and-outputs)

The checked examples under [`docs/fea/examples`](../examples) are part of the documentation contract. `runmat fea check` validates them in the release workflow.

## Compiler Stages

`runmat fea check` performs these stages without meshing or solving:

1. Parse the root module and its exact imports.
2. Bind declarations and qualified references.
3. Resolve physical types, units, topology roles, and imported `.m` signatures.
4. Validate geometry lineage, property coverage intent, procedure state flow, output expressions, sweeps, and optional execution policy.
5. Produce a typed root inventory and canonical logical plan.

Unavailable project facts produce an incomplete check, not runnable success. Syntax, type, unit, topology, or contract errors produce an invalid check. Only a valid selected study or sweep can run.
