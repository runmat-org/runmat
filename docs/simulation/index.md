# Finite-Element Analysis

RunMat FEA turns exact geometry, an explicitly authored physical model, and a procedure sequence into a validated solver result. A `.fea` file is a typed declarative program: RunMat parses and type-checks it, resolves imported `.fea` modules and `.m` functions, realizes geometry, generates a solver mesh, compiles region-bound properties and conditions, schedules the numerical work locally or on a cluster, and publishes a content-addressed result.

> [!NOTE]
> FEA is a breaking API while its validation evidence is extended. Execution fails closed when a requested physics family, formulation, backend, or evidence level is not available.

## Start Here

The [linear static](./physics/structural), [steady heat transfer](./guides/steady-heat-transfer),
[modal analysis](./guides/modal-analysis), and [mesh refinement](./guides/mesh-refinement) guides
each include complete example projects you can download or open in RunMat. The linear static guide
is a good first study: it imports a STEP cantilever,
assigns steel, fixes one end, loads the other, and reports displacement and stress. With the
project downloaded, check, plan, and run it:

```sh
runmat check cantilever-static.fea
runmat plan cantilever-static.fea
runmat login
runmat run cantilever-static.fea
```

Meshing and solving consume the simulation allowance described on the [pricing page](https://runmat.com/pricing). Usage is measured in GFLOP under [RunMat Simulation Usage Accounting](./usage-accounting). Sign-in and a connection to RunMat Server are required for admission.

Checking and planning a study do not mesh or solve. Ordinary local `.m` scripts that do not start FEA need no account or connection.

`runmat check file.fea` and `runmat run file.fea` are shorthand for the corresponding `runmat fea` commands. If a root module declares more than one runnable root, select it with `--study <id>` or `--sweep <id>`.

## Choose The Right Surface

| Work | Interface |
| --- | --- |
| Author a repeatable analysis | Typed `.fea` source plus imported `.fea` modules and pure `.m` functions |
| Check without executing | `runmat check model.fea` |
| Inspect the canonical logical plan | `runmat plan model.fea` |
| Run locally | `runmat run model.fea` |
| Run on a configured cluster | `runmat run model.fea --cluster <name> --trust-identity <sha256-fingerprint>` |
| Load and inspect geometry in `.m` code | `geometry.load`, `geometry.inspect`, and geometry query operations |
| Generate and display a mesh in `.m` code | `meshing.generate`, `mesh(meshValue)`, and mesh-quality operations |
| Read a published result in `.m` code | typed FEA result, step, field, and mesh objects; use `plot(fieldValue)` for field rendering |

The `.fea` language is the authoring authority. Runtime objects expose geometry, mesh, and result inspection; source modules are compiled by the typed FEA commands.

## Documentation Map

| Topic | Read |
| --- | --- |
| First complete study, CLI execution, and result access | [Using FEA](./using-fea) |
| Language syntax and declaration reference | [FEA language](./language/) |
| Imported, constructed, derived, and generated geometry | [Geometry](./geometry) |
| Deterministic tetrahedral meshing, quality, evidence, and scale | [Meshing](./meshing) |
| Materials, property assignment, conditions, and model composition | [Models](./models) |
| Procedure sequences, state transfer, observations, and accuracy | [Study contracts](./study-contracts) |
| Studies, sweeps, placement, and publication | [Solves](./solves) |
| How to run each supported physics family | [Physics guides](./physics/) and [Current status](./status) |
| Supports, prescribed values, coupling, contact, and interfaces | [Constraint guides](./constraints/) |
| Loads, excitations, signals, and natural boundaries | [Load guides](./loads/) |
| Result evidence, provenance, and trust | [Evidence](./evidence), [Validation](./validation), and [Trust](./trust) |
| Runtime and host operation contracts | [Operation reference](./operation-reference) |

FEA scheduling uses the same execution substrate as other RunMat workloads. Cluster routing, queue selection, retries, worker allocation, encrypted transport, artifact storage, and cancellation are execution concerns; they do not change the physical study definition or its identity.
