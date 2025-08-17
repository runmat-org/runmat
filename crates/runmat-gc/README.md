RunMat GC
=========

Overview
--------
`runmat-gc` implements a production-oriented, generational mark-and-sweep garbage collector for RunMat. 
The design prioritizes correctness and predictable semantics for a dynamic language runtime, while 
keeping performance and simplicity at the forefront. It exposes a stable, handle-based API via 
`GcPtr<T>` to avoid raw-pointer hazards and to keep integration with the interpreter straightforward.

Key properties:
- Non-moving, generational allocation by default (young/old generations)
- Mark-and-sweep collection with promotion based on survival counts
- Explicit root management (stack, globals, persistents, test roots)
- Write barrier and remembered-set tracking for old→young references
- Centralized statistics (`GcStats`) and configurable policies (`GcConfig`)
- Stable handle type (`GcPtr<Value>`) re-exported from `runmat-gc-api`


Architecture
------------
The GC is composed of the following subsystems:

- HighPerformanceGC: Top-level façade that unifies configuration, allocator, collector, roots, 
  and write-barrier state. Provides public API (allocate, collect, stats, configure, root ops).

- GenerationalAllocator: Owns heap memory across generations. Tracks young allocations, survivors, 
  and promotion thresholds. Supports fast young-generation allocation and statistics query.

- MarkSweepCollector: Implements the marking from explicit roots and sweeping of unmarked objects, 
  including promotion decisions for survivors.

- RootScanner and Root Registry: Centralizes explicit roots registered at runtime (e.g., VM stack, 
  globals, persistents, test harness roots). During a collection, the collector queries the root 
  registry to seed the mark phase.

- WriteBarrierManager + CardTable: Records old→young references at mutation sites so that minor 
  collections can scan remembered-set entries instead of all old-gen objects.

- Stats and Configuration: `GcStats` exposes running counters (allocations, collections, promotions, 
  etc.). `GcConfig` governs thresholds (minor trigger ratio, young generation sizing, and more).


Handle model (`GcPtr`)
----------------------
`GcPtr<T>` is a thin, stable handle to GC-managed objects. In the current implementation:

- Allocation is non-moving, so the pointer remains stable for the object’s lifetime.
- `GcPtr<T>` implements `Deref`/`DerefMut` to provide `&T`/`&mut T`. The only `unsafe` required by 
  the system lives inside those trait impls; user code should not write any `unsafe` for dereferencing.
- `GcPtr<T>` is `Copy + Clone`-like via a trivial bitwise copy (we implement `Clone`), so APIs that need 
  to retain a handle should explicitly clone it when passing ownership to a registry. Tests were updated 
  to clone when adding/removing roots to avoid moves.

Note on moving/compaction: If a future configuration introduces object moving/compaction, we will add a 
forwarding/indirection layer so that existing `GcPtr` values remain valid without changing embeddings.


Roots
-----
The collector discovers live objects by scanning registered roots. The system supports:

- Stack roots: The interpreter maintains a vector of live `Value` slots; a VM-specific adapter 
  registers these with the root scanner for the duration of interpretation.
- Global/persistent roots: Long-lived dictionaries for MATLAB `global` and `persistent` variables are 
  registered as global roots for the lifetime of the VM.
- Variable-array roots: For temporary arrays of locals.
- Test roots: Unit tests can call `gc_add_root`/`gc_remove_root` to pin values across collections.

During collection, the GC merges explicit roots with remembered-set derived minor roots.


Write barriers and remembered set
---------------------------------
A write barrier must run whenever an old-generation object gets a reference to a young-generation object. 
The VM calls `gc_record_write(old, new)` at mutation sites (e.g., cell element writes, struct field 
updates, object property updates). The barrier system stores card/slot metadata in `WriteBarrierManager` 
so that minor collections only scan a compact set of remembered locations rather than the entire old gen.


Generational mark-and-sweep
---------------------------
Collections come in two flavors:

- Minor (young-only):
  1) Mark: From explicit roots and remembered-set entries (old→young), traverse and mark reachable young objects.
  2) Sweep: Iterate over young allocations; free unmarked, retain survivors.
  3) Promotion: Survivors are tracked and promoted to old generation based on survival counts.

- Major (all generations): Mark & sweep across all generations; clears remembered-set state.

Promotion policy
----------------
The allocator tracks survivor counts and performs promotion once an object survives a configurable 
number of minor GCs. Promotion stats are reported via `GcStats` to tune policies.


Configuration and statistics
----------------------------
`GcConfig` parameters:
- `young_generation_size`: target size for young generation (bytes)
- `minor_gc_threshold`: utilization ratio to trigger a minor GC
- Room for future knobs (promotion thresholds, card sizes, etc.)

`GcStats` counters:
- `total_allocations`, `minor_collections`, `major_collections`
- `objects_promoted`, `collections_performed`, `total_objects_collected`
- Other allocator/collector internal counters may be surfaced as needed


Public API (selected)
---------------------
- Allocation: `gc_allocate(value: Value) -> Result<GcPtr<Value>>`
- Collection: `gc_collect_minor()`, `gc_collect_major()`
- Roots: `gc_add_root(handle: GcPtr<Value>)`, `gc_remove_root(handle: GcPtr<Value>)`
- Configuration: `gc_configure(config: GcConfig)`, `gc_get_config()`
- Stats: `gc_stats() -> GcStats`
- Barrier: `gc_record_write(old: &Value, new: &Value)`

Example (pinning a value across collections):
```rust
use runmat_builtins::Value;
use runmat_gc::*;

let v = gc_allocate(Value::Num(42.0)).unwrap();
gc_add_root(v.clone()).unwrap();     // pin
let _ = gc_collect_minor().unwrap(); // safe; v is alive
assert_eq!(*v, Value::Num(42.0));
gc_remove_root(v).unwrap();          // unpin when done
```


Safety model
------------
- The only `unsafe` lives inside `GcPtr<T>` deref implementations. All external usage goes through 
  safe APIs.
- Non-moving allocation makes `GcPtr` addresses stable. The VM and runtime can store them without fear 
  of relocation.
- The interpreter is single-threaded with stop-the-world GCs; shared global state (`WriteBarrierManager`, 
  root registry) is synchronized and marked `Send/Sync` where appropriate.
- Barriers must be called at every mutation site that may introduce an old→young edge. The VM includes 
  barrier calls in:
  - Cell element writes
  - Struct field writes
  - Object property writes


Testing
-------
The repository includes a broad test suite:
- Allocator/collector unit tests: allocation, freeing, promotion, stats accuracy
- Stress tests: large allocation cycles, nested cells/structs, interpreter integration under load
- Rooting tests: explicit root add/remove; global/persistent lifetime; remembered-set scan correctness
- Ignition integration tests: N-D gather/scatter loops, cell/struct/object updates under barriered writes

All tests can be run single-threaded for deterministic behavior:
```bash
cargo test --workspace -- --test-threads=1
```


Integration with the interpreter (Ignition)
-------------------------------------------
- The VM uses `GcPtr<Value>` throughout aggregates (e.g., `CellArray` stores `Vec<GcPtr<Value>>`).
- All expansion and slice-assignment paths dereference `GcPtr` to read `Value` and write through handles 
  on mutation, with barrier calls at each write site.
- The ignition runtime registers long-lived maps (globals/persistents) as roots for the duration of 
  execution.


Current status
--------------
Implemented:
- Generational allocator with promotion accounting
- Mark-and-sweep collector over young/all generations
- Root scanner and explicit root API
- Write barriers and remembered set for old→young edges
- Stable handle type re-exported from `runmat-gc-api`
- Integration in the interpreter and runtime, with passing test suites


Future work (optimizations and completeness)
--------------------------------------------
The following items are not required for correctness today but are desirable for performance, 
observability, and long-run robustness:

- Barrier coverage audit and dedicated remembered-set test harness
- Promotion policy tuning and survival-count decay after promotion
- Extended telemetry: per-collection pause time, freed bytes, RS size, survivor/tenured histograms
- Optional compaction (semi-space for young, sliding compaction for old) to reduce fragmentation
- Fuzzing harness combining deep recursion, randomized allocation patterns, and mixed update/write paths
- API polish: consider `gc_add_root_ref(&GcPtr<Value>)` to reduce accidental moves in user code
- Configurable card sizes / RS structures for high-churn workloads


Contributing
------------
Contributions that improve performance, observability, or add new tests are welcome. Please keep changes 
focused and include benchmarks or tests where applicable.


