# FEA-COUPLING-CHT-001: Thermal/fluid conjugate heat transfer

0 PASS, 0 FAIL, 3 OPEN, 0 TAGGED. Generated; do not edit.

## interface

Evidence kind: PhysicalBalance. Criterion: Conservatively transfer temperature and heat flux on exact solid/fluid sides under documented perfect-transfer, conductance or contact-resistance law and numerical relaxation policy.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| conjugate_heat_transfer | canonical_mixed_topology | public_study | OPEN |  |

## iteration

Evidence kind: AccuracyAcceptance. Criterion: Re-solve both physical domains to interface and global energy closure, retaining per-iteration interface evidence, failure/cutback chronology and validated spatial/temporal/coupling convergence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| conjugate_heat_transfer | canonical_mixed_topology | public_study | OPEN |  |

## dependencies

Evidence kind: RejectionInvariant. Criterion: Reject CHT while either standalone operator, transfer, state/resource admission or mandatory accuracy is unqualified; shared signal/source and boundary energy attribution remain intact.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| conjugate_heat_transfer | canonical_mixed_topology | public_study | OPEN |  |

