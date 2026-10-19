## What validation establishes

RunMat checks finite-element analysis at three levels: the mechanics of each formulation, numerical accuracy against physical references, and preservation of the result through the complete execution workflow.

A successful solve establishes that the assembled equations meet their numerical acceptance criteria. Confidence in a physical prediction also depends on the formulation, mesh, boundary conditions, and comparison quantities used for that problem.

## Formulation checks

Element tests check strain, stress, stiffness, integration, and load projection. Their acceptance criteria identify the element orders and material laws they cover.

## Physical references and refinement

Independent reference problems and mesh refinement check displacement, stress, reactions, and other physical quantities. Each assessment must identify its tested configuration, acceptance criteria, decision, and limits. A field plot alone does not establish convergence.

## Complete workflow checks

Execution checks cover authored studies, resource admission, original saved fields, and reopening results with their provenance. They establish that the workflow preserves the numerical answer within the assessed scope.

## Reading the validation records

The domain pages show standing acceptance criteria and their linked checks. Published assessment records, when available, identify the actual tested scope and findings. A linked test is a verification mechanism; its presence is not a current passing assessment.

The current documentation bundle contains no published qualification assessment records. Open criteria and linked checks remain visible without a passing claim.

## Validation by domain

- [Structural mechanics](/docs/simulation/validation/structural)
- [Thermal analysis](/docs/simulation/validation/thermal)
- [Modal analysis](/docs/simulation/validation/modal)
