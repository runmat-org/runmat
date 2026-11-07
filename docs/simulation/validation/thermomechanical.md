---
description: Reference checks for RunMat thermomechanical analyses: each problem, its closed-form, manufactured or published reference, the compared quantity, the tolerance and the latest result.
keywords: [thermomechanical validation, verification, reference solution, mesh convergence, FEA accuracy]
doc_type: validation
generated: node scripts/docs/generate-pages.mjs
---
# Thermomechanical validation

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Each check compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. The tests listed under a check run on every change and in the evidence report; receipts record a public study run with its build, commit and mesh. The [validation methodology](./methodology.md) explains how checks are built and how tolerances are set, and [Status](./status.md) summarizes every family.

## Static

### Thermal stress in a clamped bar

**Problem.** The 1 m steel bar of [thermal-stress-static.fea](../examples/thermal-stress-static.fea), clamped at both ends and held at 373.15 K, 80 K above its reference temperature; Tet10 at 50 mm.

**Reference.** A fully restrained one-dimensional bar: axial stress $E\alpha\Delta T$ = 192 MPa and clamp reaction $EA\alpha\Delta T$ = 1920 kN.

**Compared.** Nodal von Mises stress at the center of the bar; axial reaction on a clamped face.

**Tolerance.** None set; the receipts record each value. The clamps also restrain lateral expansion, which raises the stress above the one-dimensional value.

**Result.** Run receipts dated 2026-10-29.

| Quantity | Mesh | RunMat | Reference | Difference |
| --- | --- | ---: | ---: | ---: |
| Nodal von Mises stress at the center of the bar (x = 500 mm) | Tet10, 50 mm, 4,679 nodes | [197.584 MPa](receipt:heated-bar-thermal-stress#mid-von-mises) | 192 MPa | +2.91 % |
| Axial reaction resultant on the clamped face at x = 0 | Tet10, 50 mm, 4,679 nodes | [1975.84 kN](receipt:heated-bar-thermal-stress#clamp-reaction) | 1920 kN | +2.91 % |

### Thermal eigenstrain in a single element

**Problem.** A single Tet4 or Tet10 element with a uniform and with a linear temperature change, free to expand and fully restrained.

**Reference.** Free expansion: strain $\alpha\Delta T$ and no stress. Full restraint: stress $-E\alpha\Delta T/(1 - 2\nu)$ and the matching strain energy.

**Compared.** Strain, stress and strain-energy density.

**Tolerance.** Free strain within 1e-12 and free stress below 2e-3. Restrained stress within 1e-5 for the uniform change and relative 1e-12 for the linear change; energy within relative 1e-12.

**Result.** 
