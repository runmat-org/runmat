# Shared frequency publication contract

This is a metadata/publication contract, not a qualified authored frequency-sweep
execution envelope. Existing singleton product guards remain in force.

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

## Remaining whole-chain work

Complete shared durable root inventories and scientific field paging, governed
ordered coordination, per-sample state/sequence closure, resource accounting,
runtime/native authored publication and independent multi-frequency CAD/cold
evidence before removing singleton guards. Preserve existing structural behavior
through that migration; do not introduce an acoustic-private paging lifecycle.
No Stage 8, production, distributed/GPU, measured-memory, restart or release gate
is promoted by this contract.
