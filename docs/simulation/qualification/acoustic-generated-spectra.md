# Generated complex acoustic spectra

This evidence contract covers shared compiled RunMat signal generation feeding
CPU FP64 canonical Tet4/Tet10 acoustic frequency tasks. It does not
replace the independent duct/cavity/interface continuum evidence, qualify authored
CAD studies, or claim a distributed numerical solve or whole-study restart.

The process-conformance fixture compiles and freezes this actual function:

```matlab
function y = pulse(f)
arguments
    f (:,1) double
end
y = complex(1 + 2*f, 0.5 - f);
end
```

Four samples at 1/8, 1/4, 3/8 and 1/2 Hz are generated in two chunks through the
existing inline, isolated-process, durable-process and encrypted-remote execution
paths. An independently constructed typed explicit table has the same complex
ordinates, linear interpolation and rejecting extrapolation. Shared admission,
value decoding, canonicalization and materialization own the resulting table;
the fixture does not fabricate generator outputs or authorize unknown effects.

Prescribed pressure, normal velocity and normally incident pressure each consume
their correct shared physical quantity. Numerical tasks use both element orders,
result chunk sizes 1024/8192, and five frequencies, including an interpolated
3/16 Hz sample. Explicit and generated inputs must produce bit-exact complex
nodal pressure, element-mean velocity, intensity, energy, net input power, passive
boundary loss and algebraic residual. Natural-source projection evidence must be
nonempty and accepted. Both input power and passive loss must be positive and
accepted relative power closure must remain below the unchanged 1e-8 bound.

Every frequency has a distinct governed task intent. Completion deliberately runs
in reverse frequency order; canonical fan-in must restore the declared ordering
and remain invariant under reversing the imported result list. Per-entry objects
are imported after reopening the filesystem store. Exact equality after individual
and aggregate cold canonical decoding preserves original fields, powers and
provenance without recovering them by another solve. Physical equality must not
erase generated provenance: generated and explicit result identities differ,
while generator placement and numerical transport chunk size do not alter the
corresponding generated identity.

The fixture uses a single authenticated canonical tetrahedron, not generated
production topology. Generator placement equivalence is not numerical placement
equivalence: the acoustic tasks here execute through the local native analysis
kernel. It is wired into normal process conformance through `signal::run` and has
a focused entry point:

```sh
cargo test -p runmat-execution-runner-native --test analysis_process_conformance -- --acoustic-signal-only
```

Full authored acoustic studies, general volume-velocity sources, resonance/
uniqueness diagnostics, radiation qualification, measured resource envelopes,
additional numerical placements and blocking release evidence remain required.

## Verified bounded inventory

The focused process-conformance run passed all three drives: 24 real generator
tasks and 300 local numerical frequency entries, with exact field/phase/power,
provenance, fan-in ordering and cold-decoding assertions above. Local encrypted
QUIC transport requires socket permission in a sandbox; permission denial is not
a numerical failure. Shared `complex` catalog admission uses its canonical pure
contract, without changing generator effect policy or its numerical implementation.
This closes this bounded explicit/generated source comparison, not all Stage 8 gates.
