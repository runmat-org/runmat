---
title: "FEA on Geometry"
category: "FEA"
section: "13.0"
last_updated: "June 10, 2026"
---

# FEA: Run Math on Geometry (Beta)

RunMat FEA lets you start with a CAD or mesh file and run repeatable physics studies against it. The FEA subsystem in RunMat allows you to load geometry, mesh it into finite elements, define material properties, loads, constraints, domains, interfaces, and steps, and solve multi-physics ODEs and parametric sweeps.

> [!NOTE] RunMat FEA is in beta and is under active development. The API is subject to change.

## How To Use It

| If you want to... | Use |
| --- | --- |
| Run a repeatable study from the terminal or CI | `runmat check model.fea` and `runmat run model.fea` |
| Build or generate a study from RunMat code | `geometry.load(...)`, `fea.model(...)`, `fea.study(...)`, `fea.validate(...)`, `fea.run(...)` |
| Build or generate a study from a `.fea` file | A `.fea` YAML document can be loaded by `fea.load(...)` or run with `runmat run` |
| Inspect geometry before modeling | `geometry.inspect(...)` and geometry runtime operations |
| Integrate from Rust or a host application | Versioned runtime operations such as `fea.create_model/v1` and `fea.run_study/v1` |

`.fea` is a declarative study format. It is a YAML document with a RunMat-specific extension, similar to how `.m` is the code format. Just like `.m` files, `.fea` files can be passed to `runmat check` or `runmat run` for static analysis and execution, respectively.

## Topics

| Task | Read |
| --- | --- |
| Run studies from the CLI, `.m` code, or Rust host code | [Using FEA](/docs/fea/using-fea) |
| Load, inspect, and prepare CAD or mesh geometry | [Geometry](/docs/fea/geometry) |
| Define solver-ready model data | [Models](/docs/fea/models) |
| Choose the physics family and understand family limits | [Physics Families](/docs/fea/physics) |
| Run direct solves, studies, and sweeps | [Solves, Studies, and Sweeps](/docs/fea/solves) |
| Understand saved artifacts, diagnostics, and provenance | [Evidence & Artifacts](/docs/fea/evidence) |
| Understand how correctness is tested and validated | [Verification & Validation](/docs/fea/validation) |
| Interpret result quality and trust signals | [Results & Trust](/docs/fea/trust) |
| Check current support and known boundaries | [Current Status](/docs/fea/status) |
| Integrate with runtime operation contracts | [Operation Reference](/docs/fea/operation-reference) |

For general runtime execution, see [Execution](/docs/runtime/execution). For host session behavior, see [Session Engine](/docs/runtime/session). For GPU behavior, see [GPU Acceleration & Fusion Engine](/docs/runtime/gpu).
