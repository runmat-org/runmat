---
description: Run a first RunMat FEA study from the command line, RunMat Desktop, the agent or .m code. Check, plan and run a .fea file, select studies and sweeps, run on a cluster, and read results in .m.
keywords: [get started FEA, runmat check, runmat run, runmat plan, first FEA study, RunMat Desktop, FEA results in MATLAB syntax, fea.results, cluster run]
doc_type: guide
examples: [cantilever-static]
---
# Get started

Every RunMat FEA study is a `.fea` file. The same file runs from the command line, in RunMat Desktop, through the RunMat agent and from `.m` code, and each of them produces the same result. This page runs the cantilever example from the [linear static guide](./guides/linear-static.md) each way.

## Run a study from the command line

Install RunMat ([command-line setup](/docs/runtime/getting-started/cli)), download the cantilever example from the [linear static guide](./guides/linear-static.md), and from its folder run:

```sh
runmat check cantilever-static.fea
runmat plan cantilever-static.fea
runmat login
runmat run cantilever-static.fea
```

Checking and planning don't mesh or solve, and need no account. Meshing and solving need sign-in and a connection to RunMat Server, and they consume the simulation allowance described on the [pricing page](https://runmat.com/pricing). [RunMat Simulation Usage Accounting](./usage-accounting) describes how usage is measured. Local `.m` scripts that don't start FEA need no account or connection.

| Command | Does | Takes |
| --- | --- | --- |
| `runmat check <file>` | Parses the file and its imports, and checks types, units, references, CAD region names and that each family runs each requested procedure. Prints every error with its file, line and column. | seconds |
| `runmat plan <file>` | Prints the selected study's work: geometry, mesh, each analysis step, the state passed between steps, outputs and placement. | seconds |
| `runmat run <file>` | Imports the CAD, meshes, solves each step, evaluates outputs and assessments, and publishes the result. | seconds to hours, by model size |

- **Choose a study or sweep.** A file that declares several studies or sweeps needs `--study <name>` or `--sweep <name>`: `runmat run load-envelope.fea --sweep envelope`.
- **Read the output in tools.** `--json` prints structured output, including the run id, the outputs and the assessment verdicts.
- **Use one workflow for `.m` and `.fea`.** `runmat check` and `runmat run` also take `.m` scripts.

## Run on a cluster

The same file runs on a configured cluster without changes to the study:

```sh
runmat run cantilever-static.fea --cluster engineering --workers 8 --trust-identity sha256:<endpoint-fingerprint> --json
```

`--cluster` names the cluster, `--trust-identity` pins the endpoint's certificate fingerprint, `--queue` and `--project` select a queue and a project, and `--detach` returns once the run is submitted. Credentials and routing come from project configuration, not from the study. To state placement, resources, fan-out and checkpointing in the study itself, add an `execution` declaration; see [execution, parallel work and distributed work](./language/execution-parallel-and-distributed.md).

## Run a study in RunMat Desktop

Open the example project in RunMat Desktop from the example panel of any guide. Desktop shows the `.fea` source beside the model and its results, checks the source with the same rules and messages as `runmat check`, and runs the study on the local machine. The published fields open in the 3D viewer.

## Ask the agent

The RunMat agent reads and writes `.fea` files. Give it the question and the criterion, for example: "Set up a linear static study of `bracket.step`: fix the faces named Mounting, apply 2 kN in −z to the face named Hook, and check the peak displacement against 0.1 mm." The agent drafts the study, checks it, runs it and reports the result against the criterion. Its edits to `.fea` files are staged as ordinary text changes, which you review before they are applied.

## Read results in `.m` code

After a run, `.m` code loads the result by its run id (printed by `runmat run --json`) and selects fields for plotting or post-processing:

```matlab
results = fea.results(runId);
stress = fea.field(results, fea.StructuralField.vonMises());
plot(stress);

tip = fea.field(results, fea.OutputField.named("tip_deflection"));
```

`fea.field` selects a field by its physical meaning (`fea.StructuralField.vonMises()`) or by the name of a study output (`fea.OutputField.named`). `plot` and `mesh` are the ordinary RunMat graphics functions; they take the field or mesh object. For a multi-step run, select a step first with `fea.stepResult`. Studies can also be built entirely in `.m` with `fea.study` and solved with `fea.run`; see the [function reference](/docs/reference/builtins).

## Inspect CAD before naming regions

Regions are selected by names authored in the CAD file. To see which names, layers and bodies a file contains before writing a study:

```matlab
facts = geometry.inspect("geometry/bracket.step");
part = geometry.load("geometry/bracket.step");
regions = geometry.listRegions(part);
mesh(part);
```

A face the CAD file leaves unnamed can be named in the CAD tool and re-exported. See [geometry, models and assignments](./language/geometry-models-and-assignments.md#name-regions).

## Next steps

- Work through a task: [linear static](./guides/linear-static.md), [steady heat transfer](./guides/steady-heat-transfer.md), [modal analysis](./guides/modal-analysis.md), [mesh refinement](./guides/mesh-refinement.md).
- Learn the file format: [the `.fea` language](./language/index.md).
- New to the method: [what is finite element analysis?](./guides/what-is-finite-element-analysis.md)
