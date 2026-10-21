# Matching periodic and Bloch pressure

The canonical Tet4/Tet10 harmonic solver supports bijective matching
exterior-face pairs under translation (or identity), with
`p_secondary = exp(i phase) p_primary`. The convention is peak phasors and
`exp(+i omega t)`. Core declarations and authored language use the existing typed
periodic and phase-shifted-periodic constraints.

Canonical geometry pairing resolves the actual mesh node identities. A shared
unit-phase graph accepts consistent redundant cycles and rejects conflicting cycles
or propagated prescribed pressures. Nonmatching interpolation, rotated pairing,
overlapping paired faces, and overlapping explicit natural boundaries or normal
velocity sources fail closed. Pressure anchors are allowed when phase-consistent.
Inconsistent cycles are rejected, not interpreted as constraints forcing zero.

The shared affine map uses stacked real/imaginary pressure. Helmholtz application is
restricted as `T^H A T`; prescribed lifting is `T^H(f - A p_D)`. The real-block
transpose is the complex conjugate restriction. No separate complex matrix algebra
or structural displacement surrogate is used. Recovery expands the full physical
pressure before existing velocity, intensity, energy and power calculations. Fixed
phase components contribute prescribed-pressure reaction power; free unit-phase
pair reaction work cancels in the global balance.

## Evidence

- Shared-map tests cover free arbitrary/zero/quarter-turn phases, conjugate
  restriction, independent dense congruence, deterministic components, cycles,
  invalid inputs and prescriptions spanning amplitudes `1e-200` to `1e200`.
- An independent dense complex Helmholtz reference covers genuinely free phased
  DOFs, arbitrary complex forcing and nonzero lifting at four phases and three
  amplitudes. It does not derive expected results from the production recovery map.
- Anchored outgoing plane waves use `p=A exp(-ikx)` and `phase=-kL`, both element
  orders, two complex amplitudes and three refinements. Independent pressure,
  element-mean velocity, signed intensity and pair power checks pass. Unphased
  standing-wave checks and invalid pairing/boundary/prescription checks also pass.
- Portable cold workload/mesh/result round trips retain the full pressure field and
  reject substituted phase lineage. Resource planning includes the phase graph,
  affine recovery, lookup and pairing workspace, without raising authored ceilings.

## Open gates

This is not Stage 8 completion or qualification. Arbitrary rotated or
nonmatching pairs, unforced periodic eigenmodes, independent continuum references
with free phase-linked boundary traces, actual authored native-process periodic
publication/restart, measured memory reconciliation, and release evidence remain.
Existing radiation convergence failures remain independent unmet gates. Cold
serialization is not a resumed solve and does not establish restart support.
