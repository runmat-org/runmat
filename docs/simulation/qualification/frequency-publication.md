# Shared frequency publication contract

This documents the shared publication contract and the governed authored acoustic
sweep cutover. Neither is a qualification of the acoustic formulation.

## Authority and invariants

`FrequencySamplePublicationAuthority` records the independently authenticated
aggregate workload/result, producer, topology, shared frequency plan and every
ordered numerical entry. Each entry retains its original workload and typed
recovery source. The manifest must cover that inventory exactly. Recovery requests,
field definitions, source identities, topology and frequency coordinates must agree;
two definitions cannot advertise the same semantic field ID at one coordinate.
Optional observations must name the exact recovered field source and coordinate.

The contract contains catalogs, not numerical arrays, transport paths, fabricated
receipts or resource evidence. Internal hash convergence is not proof of numerical
origin: consumers must validate against independently authenticated authority.
The generic object decoder requires that authority as an explicit argument.

Selection is exact. A multi-sample publication requires an explicit positive finite
frequency present in the complete inventory. There is no implicit reference,
first/last, nearest-frequency or interpolation selection. Singleton omission is
unambiguous and allowed.

## Verified boundary

Contract tests cover both existing typed numerical-source variants, canonical
round trips, placeholder/foreign/missing/duplicate/unordered inventory rejection,
producer/topology/source/coordinate mismatch, semantic namespace collisions and
optional observation lineage. Synthetic contract fixtures are not physics evidence.

An acoustic consumer derives authority from actual validated aggregate numerical
outputs. A two-frequency test solves the existing canonical acoustic fixture,
recovers the original catalogs and transports this metadata through generic
chunked objects into a separate cold object store. Both chunk sizes preserve exact
inventory and selection; missing chunks, foreign roots and aggregate substitution
reject. This proves cold metadata transport, not partial cold numerical field access.

Verification commands (using an existing target cache):

```sh
cargo test --locked -p runmat-analysis-fea -p runmat-analysis-execution --lib frequency_sample_publication
cargo test --locked -p runmat-analysis-core -p runmat-analysis-execution --lib
cargo clippy --locked -p runmat-analysis-core -p runmat-analysis-fea -p runmat-analysis-execution --lib --tests --no-deps -- -D warnings
```

## Shared cold numerical access

The runtime now has shared typed artifact roots, a complete ordered frequency
sample-root inventory, and an admitted publication handle. Admission compares
external numerical authority and exact metadata payload, validates all immutable
root manifests and the complete declared physical chunk inventory, and rejects
missing, swapped, duplicate, foreign or unreferenced roots/objects. Numerical
chunks remain lazy; metadata admission is not a claim that every chunk has already
been fetched or that full numerical data has been validated.

Scientific field paging uses the existing structural-frequency partition reader,
not an acoustic-private materializer. Exact frequency selection is mandatory for
multi-sample publications. A cold transport-contract fixture proves that a complex
page reads only its selected root and intersecting original partition chunks,
without retained original arrays or a numerical solver. Unselected-root corruption
and missing chunk declarations fail admission; selected missing/corrupt chunks
fail paging. These synthetic transport fixtures are not independent physics evidence.

Existing structural publishers, cold importers and storage use the same shared
root primitives and physical closure checker; the structural publication index
remains typed because its result/receipt/resource/history authority is structural.
Solver results, numerical receipts, semantic state and resource evidence are not
invented or replaced by the generic metadata/field/observation inventory.

Verification of this shared runtime slice also requires the focused artifact-root
tests, frequency-page regressions, runtime/native compilation and strict lint
review. Known unrelated runtime lint/module debts do not earn a whole release PASS.
The isolated candidate passes artifact-root tests6/6, frequency regressions36/36
and structural study storage16/16. Runtime/native library compilation passes.
Strict lint reports the eight existing field_scene/render_tiles/storage findings;
architecture reports the same three runtime module debts. No waiver was added.

## Governed authored acoustic sweep

The coordinator owns one independent entry lifecycle per exact authored frequency:
solve, original numerical receipt, accuracy evaluation, acoustic-complex state,
baseline usage and terminal publication. Native execution submits independent
entry waves through the shared analysis task pool, subject to its admitted capacity.
The same state declaration can appear at each coordinate, but each state retains
its own frequency and original source result. No first/last/reference sample or
aggregate is presented as one acoustic phasor state.

A separately admitted worker task assembles original accepted entries and publishes
their native fields and shared sample metadata. It has its own resource plan and
actual terminal scheduler accounting, not an invented aggregate numerical receipt.
Worker rederivation honors the original authored resource ceiling intersected with
the shared post-processing envelope. Recovery copies the accepted original fields;
neither aggregate publication nor cold reopening reruns the numerical solver.

Portable completion roots retain every entry's ordered phase closures, the aggregate
task closure and recorded graph elapsed time. Cold replay validates original
response/accounting associations rather than manufacturing new scheduler evidence.
Public runtime completion uses the shared study-completion path and exposes states
inside their explicit frequency entries; the flat single-state list is empty.
Scientific queries require exact sample selection and use the shared admitted
partition reader, including singleton omission when unambiguous.

The current authored envelope remains one fresh direct harmonic step with one
complete acoustic-complex state declaration. Dependent sequences, restart,
observations and authored output declarations remain guarded pending their full
implementation and evidence. Numerical and resource limits are not relaxed.
The graph records successful terminal phase attempts supplied by the shared pool,
not exhaustive failed-retry billing. Full completion authentication still decodes
the bounded numerical closure; lazy partition reads are not a measured peak-memory
qualification of that authentication path.

Native worker tests cover Tet4/Tet10, two/three frequencies, reversed original
completion, cold field objects, differing input/worker chunk policies, corrupt or
missing selected partitions, source substitution and low authored-memory rejection.
The authored CAD/process witness passes Tet4/Tet10 at singleton and two-frequency
schedules. After producer exit it verifies all bounded pressure/velocity pages,
exact partial complex reads and per-entry original states; cold completion replay
rejects omitted/reversed entries, a missing state phase and foreign root authority.
Its paging helper traverses the default bounded pages rather than raising their
limit. Aggregate accounting retains full mesh input residency but does not project
the non-emitted mesh as each output; the same original authored ceiling applies.
The original fine Tet10 duct regressions exposed a second overprojection: one
generic maximum payload was charged to all three distinct outputs. Current admission
uses the shared streaming serializer for the borrowed aggregate and sampled native
catalog/partition shape profiles for fields. Publication metadata reserves64KiB per
entry plus every catalog descriptor; that bound is restricted to the current fixed
native definitions and absence of observations. Full input/validation workspace,
canonical materialization, transport and one field manifest per frequency remain
accounted. No authored or local-pool ceiling was raised. Tests compare profiles with
actual encoded/resident contracts, including immediately below/at/above partition
boundaries and several frequency coordinates. This is admission evidence, not a
measurement of peak process memory.
Post-integration checks pass: the execution bridge175/175, native acoustic
selection14/14, shared roots6/6 and structural storage16/16. The selected runtime
frequency regressions pass41/41 in both isolated and integrated trees; this includes
non-FEA builtins and is not41 independent physics cases. Existing bounded evidence
and scalar-metadata reader checks also pass in the integrated worktree.
The original matched, incident-pressure and volume-flow duct matrices each pass
all eight unchanged cases and terminal refinement/source-equivalence, original full/
partial field and cold-completion checks. Terminal durations are2821.73s,2804.98s
and2805.13s respectively. Both element orders and original generated/explicit meshes
are retained under the unchanged authored targets and1GiB pool. These close this
cutover's original regression gate, not the broader acoustic qualification gates.
Final shared core/FEA/bridge strict lint passes; runtime/native lint retains eight
prior findings. Isolated architecture reports two prior rendering debts; mixed-main
architecture additionally reports existing assessment/observation/structural-root
debt and a stale assembly ratchet. No suppression or size-ceiling increase is used.
Broader resonance/uniqueness, original radiation acceptance, measured
resources, all advertised placement shapes and blocking release evidence remain
separate mandatory gates.
No Stage 8, qualification, distributed/GPU, measured-memory, restart or release gate
is promoted by these tests.
