# Operation reference

This page maps the public command, language, runtime-object, and host-operation surfaces. The typed `.fea` compiler is the source-authoring authority; runtime functions operate on already typed objects and published results.

## Command Line

| Command | Purpose |
| --- | --- |
| `runmat fea check <file>` | Resolve and type-check one complete source graph without execution |
| `runmat fea plan <file>` | Print canonical logical phases and the selected study or sweep definition |
| `runmat fea run <file>` | Realize geometry, mesh, compile, solve, and publish one study or sweep |
| `runmat check <file.fea>` | Shorthand for typed FEA check |
| `runmat run <file.fea>` | Shorthand for typed FEA run |
| `runmat mesh <cad-file>` | Produce a canonical solver mesh plus bound meshing evidence |

FEA root-selection options are `--study <id>` and `--sweep <id>`. `check`, `plan`, and `run` accept `--json`; `run` additionally accepts `--cluster`, `--project`, `--queue`, `--trust-identity`, `--idempotency-key`, `--workers`, and `--detach`.

`runmat mesh --help` is authoritative for direct meshing controls. Principal options are `--source-units`, `--target-size`, `--deviation`, `--element-order tet4|tet10`, `--max-elements`, `--seed`, `--output`, `--evidence`, `--force`, and the same remote placement options.

## `.fea` Declarations

| Declaration | Required core |
| --- | --- |
| `geometry <id> { ... }` | exactly one imported, constructed, or generated source plus typed regions as needed |
| `geometry <id> from geometry.<source> { derive { ... } }` | immutable derived feature graph |
| `mesh <id> on geometry.<id> { tetrahedra { ... } }` | `order`, `target_size`, and `maximum_deviation`; optional `maximum_grading_ratio`, `deterministic_seed`, `cells_across_gap` and `refine <name> { on, target_size }` blocks |
| `material <id> { ... }` | one or more physical capability properties required by its consumers |
| `model <id> { ... }` | assignments, conditions, actions/signals, cases, properties, and composition |
| `study <id> { ... }` | geometry, matching mesh, model, at least one analysis, and optional outputs/inputs/execution |
| `sweep <id> over study.<id> { ... }` | axes with `product|zip`, or explicit cases |
| `execution <id> { ... }` | optional placement/resource/fan-out/cooperative/resilience preferences |

See the [language reference](./language/) for exact syntax and operation signatures.

## Geometry And Meshing In `.m`

| Function | Result |
| --- | --- |
| `geometry.inspect(path)` | immutable facts about source format, topology, metadata, parameters, and properties |
| `geometry.load(path, Name, Value, ...)` | typed `geometry.Asset` |
| `geometry.listRegions(asset)` | selectable region inventory |
| `geometry.meshes(asset)` | render-oriented geometry meshes |
| `meshing.generate(asset, Name, Value, ...)` | validated `meshing.Mesh` with bound evidence |
| `meshing.nodes(meshValue)` | node ids and coordinates |
| `meshing.elements(meshValue)` | typed element topology |
| `meshing.boundaryFaces(meshValue)` | boundary-face topology and provenance |
| `meshing.regions(meshValue)` | physical mesh-region inventory |
| `meshing.quality(meshValue)` | per-element and summary quality measures |
| `meshing.evidence(meshValue)` | request, stage, invariant, resource, and provenance evidence |
| `meshing.save(meshValue, meshPath, evidencePath)` | atomically save the jointly validated pair |
| `meshing.load(meshPath, evidencePath)` | load and jointly validate a saved pair |
| `mesh(assetOrMesh, Name, Value, ...)` | display geometry or a solver mesh; numeric `mesh(X,Y,Z)` is unchanged |

Geometry/mesh display options affect rendering only; they do not alter exact geometry or canonical solver-mesh identity.

## Result Access In `.m`

| Function | Contract |
| --- | --- |
| `fea.results(runOrRunId, Name, Value, ...)` | load typed result metadata and selected descriptors from a result object or persisted run id |
| `fea.field(resultsOrRun, selector, sample)` | select one typed field or authored output at an optional typed independent-domain sample |
| `fea.fieldPage(field, start, count)` | load a bounded one-based page, at most 4096 logical values |
| `plot(field, Name, Value, ...)` | render a selected field using ordinary RunMat plotting dispatch |
| `mesh(runOrResultsOrField, Name, Value, ...)` | display the exact solver mesh carried by FEA result context |
| `fea.compare(baselineRunId, candidateRunId)` | compare persisted runs |
| `fea.trends(Name, Value, ...)` | query bounded result trends |

Selectors are typed objects such as `fea.OutputField.named("von_mises")`, structural field selectors, and independent-domain values such as `fea.Frequency.hertz(120)` or `fea.Time.seconds(0.0122)`. A transient state can also be selected by its one-based position in time order: `fea.field(results, fea.StructuralField.displacement(), 123)`. A time selects the retained state within one part in 10^9 of the run's time span, and the returned field reports that state's exact time in `sample`; read its values with `fea.fieldPage`. Field choice belongs to `fea.field`; it is not a string option to `plot`.

## Programmatic Typed Objects

RunMat also exposes typed constructors for materials, assignments, constraints, signals, excitations, sequences, accuracy policies, sampling/storage policies, and result selectors. They are useful when `.m` code creates an in-memory study object. `fea.validate`, `fea.plan`, and `fea.run` accept only typed `fea.Study` or `fea.Sweep` values; they do not accept a `.fea` path. Use the CLI/source compiler to execute a `.fea` module graph.

Introspection and completion are generated from builtin descriptors, so the active runtime remains the signature authority for the full constructor inventory.

## Host Operations

Host integrations use versioned operation envelopes. The main object-oriented operations are:

| Operation | Purpose |
| --- | --- |
| `fea.validate_study/v1` | validate an in-memory study |
| `fea.plan_study/v1` | plan an in-memory study |
| `fea.run_study/v1` | execute an in-memory study through the configured runtime provider |
| `fea.validate_study_sweep/v1` | validate an in-memory sweep |
| `fea.plan_study_sweep/v1` | plan an in-memory sweep |
| `fea.run_study_sweep/v1` | execute an in-memory sweep |
| `fea.results/v1` | query persisted result metadata |
| `fea.results_compare/v1` | compare persisted results |
| `fea.trends/v1` | query result trends |

Typed `.fea` source checking and native root execution use their public source/result contracts directly rather than serializing an authoring document through these object operations.

## Identity And Errors

Source/module revision, exact geometry, mesh request, compiled properties, procedure workloads, generated inputs, outputs, and execution artifacts each have separate canonical identities. Physical paths, process ids, worker ids, queue names, retry attempts, and timings do not enter the placement-neutral study identity.

Errors are structured by phase and stable code. Syntax/type/topology/model errors are distinct from unavailable capabilities, exact-geometry admission, meshing quality/budget failures, scheduler failures, numerical failures, cancelled work, and result-publication or reconstruction failures. Callers should branch on the stable code/status and present the accompanying message and source span or evidence witness.
