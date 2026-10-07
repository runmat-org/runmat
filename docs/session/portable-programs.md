# Portable compiler products and session invocation

A V4 executable unit carries the original MIR, analysis, bytecode, VM layout,
function registry and source map. `ExecutableUnit::from_portable_envelope` restores
these through core's executable owner. It validates the supported environment and
component revisions, original primary source identity and canonical component
inventory. It preserves optional extension metadata and rejects required extensions
the consumer does not support.

The primary source is selected by the original bytecode source ID. Source-map order
does not select it. Portable source maps retain logical ownership, paths and text;
they do not authorize reads from the producer's physical filesystem paths.

Import validates component consistency and provenance. It neither recompiles the
source nor proves that arbitrary imported bytecode implements that source.

## Ownership

- The execution host validates the program request and artifact, decodes arguments,
  and encodes the exact requested output inventory.
- Core's `RunMatSession::invoke_portable_unit` imports and invokes the compiler
  product using the session's existing program, execution and host service ports.
- The configured language environment must match the original program. An installed
  frozen project must match its original graph and source revision.
- Functions are selected by their original semantic function identity. A display
  name does not replace that identity.
- Core does not create another scheduler or artifact transport owner.
- The low-level VM request executor rejects complete V4 units. Native assessment
  and signal adapters, and browser artifact execution, enter the core session host.
  They do not reconstruct only bytecode and registry state.
- Program context is installed on native and WASM targets. Its active-unit scope
  restores the previous callable authority after nested invocation and expires
  when invocation ends.
- Browser in-process execution inherits the existing browser execution service.
  This does not establish nested worker allocation or console/project transport.

## Original dependency closure

Each unit binds foreign semantic functions to the immutable compiler products
published when that unit was compiled. Later interactive redefinitions do not
replace these owners. Capture serializes a flat, sorted, deduplicated inventory of
original products; member products carry their local components and cannot contain
nested inventories. The existing 256 MiB envelope limit and a 1,024-product limit
bound the portable inventory.

Cold import restores the complete ownership graph before exposing the unit. Missing,
ambiguous, cyclic, extra or modified products reject. Copied foreign function
bytecode must match its original owner, including instructions, source identities,
spans and ABI. Only caller-specific coverage sites are excluded from that projection
comparison. Descriptor hashes bind transported content; they do not prove source
semantics or numerical correctness.

Foreign calls enter the original owner's program and source scope, inherit caller
services and cancellation, and restore the caller scope on return. Exact owned
semantic functions also support forced generic native invocation. V4 has a distinct
artifact wire discriminant; V3 products cannot be interpreted as dependency-bearing
units.

## Script outputs

A fresh script with zero requested outputs returns an empty output inventory. One
requested output requires an actually assigned `ans`. An allocated but unassigned
slot, an unrelated final variable, and a branch that never assigned `ans` are not
results. Scripts reject multiple output ordinals.

The portable interpreter uses the VM's existing assignment inventory. Native script
tiering currently has a zero-output contract; forcing native execution while
requesting an `ans` output is rejected explicitly.

## Live invocation controls

`InvocationControl` composes the session's cancellation flag, the caller's original
shared flag, inherited parent controls, and the earliest absolute deadline. Entering
a different session captures those parent controls before activating the child
runtime context. Nested invocation cannot extend or clear a parent's deadline;
returning or unwinding restores the previous control inventory.

The shared execution `CancellationToken` owns this composition. Its clones share
one immutable inventory of original flag authorities. Cancelling a token mutates
its original primary flag; observing a deadline does not mutate that flag or turn
it into a user cancellation. Raw shared-flag constructors remain available where
the host supplies only a flag; adapters that receive a complete token preserve it.

The portable interpreter polls the token during execution and reports user
cancellation as `RunMat:ExecutionCancelled` and deadline expiry as
`RunMat:ExecutionDeadline`. FEA, provider and CAD adapters retain the same live
sources and deadline at their existing cooperative checkpoints. The native CAD
registry uses opaque handles and preserves the original inventory after lock
poisoning; polling clones the token before releasing the registry lock.

Native host safepoints, semantic sites, placement snapshots and fused numeric
regions consume the same complete token. Native deadline exits retain their reason
through Core. Numeric regions poll before allocation, at their existing element
checkpoints and before publishing transactional outputs.

These controls are cooperative. Original running-loop witnesses exercise both the
portable interpreter and forced generic native code; native tests warm the exact
compiled function before entering its loop. They do not prove interruption of a
blocked await or preemption inside a kernel without checkpoints. Live flags are
process-local controls, not serialized restart authority or measured resource usage.

## Nested interpreter dispatch

Indexing, arithmetic and instruction families choose typed handlers before
constructing their async work. Nested calls retain the selected handler rather
than a dispatcher future containing unrelated operation families. Shared stack,
assignment, class-access and exception owners remain authoritative.

The public WASM artifact tests cover original nonprimary functions, bound handles,
anonymous captures that call helpers, zero/one/two output ordinals, and assigned
versus absent script `ans`. These run under the existing stack limit; they do not
establish arbitrary recursion depth or whole-task resource qualification.

## Deferred calls

When an execution service requests a portable capture, interpreter and native
future creation ask the active core program service for the complete V4 product
of the exact semantic function. They preserve the original revision, source map,
analysis and compiler components. Missing program authority fails explicitly.
Raw script execution also enters the original core program scope on native and
WASM targets. Its interpreted call uses that product's revision in a scoped runtime
context; the session's outer revision is restored after execution.
Serial in-context services may retain a future without portable capture.

Artifact construction belongs to the shared artifact owner. Deferred materialization
requires canonical V4 bytes, the same original revision and function identity, and
the original output contract. Partial registries and absent revisions reject; the
consumer does not invent a replacement program revision. Native executor/JIT state
no longer retains a second registry-only capture for this purpose. Standalone AOT
manifests retain their existing owner and do not substitute for a V4 capture.

The original native output/cancellation/pending-await/cold-attachment witness now
uses authored complete compiler products. Core tests compare the captured product
exactly and validate its logical source fields after cold import. Portable physical
paths remain absent. This cutover does not establish transport of the caller's full
project, console, host policy or parent allocation, nor measured resource usage.
Browser service tests compile; actual browser scheduling and broader host-context
qualification remain separate gates.

## Qualification boundary

This API provides session invocation of a portable compiler product. It does not
by itself complete the legacy batch/script cutover, console transport, host-policy
admission, nested process allocation, or placement/resource qualification. Those
require their complete original host contracts and independent execution witnesses.

## Invocation environment

Runtime sessions own their environment values. `getenv`, `setenv`, `unsetenv`, and
native subprocess environment construction use the active session; mutation does
not change a different session or the host process. Nested calls and async polls
inherit their invocation context. Standalone compatibility APIs keep their existing
host behavior outside an active session.

Hosts can construct a session with `RunMatSession::with_frozen_environment` without
first collecting ambient variables or the host search path. An empty snapshot means
no visible variables, not ambient inheritance. Snapshot replacement validates sorted,
unique names before changing state and invalidates dynamic source discovery caches.

Bundles carry only explicitly selected public values through
`ExecutionBundleBuilder::with_portable_environment`. The existing secret-name policy
is retained; invalid names, duplicates, more than256 entries, over4096 bytes per value,
and over1MiB total name/value bytes reject before execution. The snapshot participates
in bundle identity and archive validation. Both driver-local and worker program/test
routes install the exact bundle snapshot after cold source materialization; each
attempt starts with its own copy. This is a bounded semantic environment contract,
not measured process memory or a complete host sandbox.

Complete console retention, portable host policy and parent allocation transport
still require their own execution evidence; captured invocation context is described below.
No remote/distributed, accelerator, FEA family or release qualification follows from
these environment checks.

## Captured invocation environment

The compiler compatibility identity and invocation environment are distinct. Deferred
calls capture the original session environment when the future is created. A single
materializer carries it with the callable, arguments and outputs in the portable
schema-V2 request. The required field may be explicitly empty; omission does not mean
inherit the worker process. Context is included in request identity and in the original
scheduled task, attempt and checkpoint. A code artifact alone does not own context.

Local interpreter and native invocation extents isolate mutations from their parent
while retaining the original session services and cancellation. Native and browser
cold hosts initialize from the request snapshot. Bundle admission requires its declared
environment to agree with the original descriptor. Remote worker control admission also
checks bundle, task and request agreement before reserving capacity. Composed child
descriptors retain their original context through cold decoding. Cold diagnostics establish transport
identity and shape, not permission to execute or independent mathematical proof.

Portable admission uses the existing public bundle policy: sorted unique names, bounded
ASCII data, and rejection of credential-like names. Unsupported inventories reject; adapters
do not silently filter them. Name validation does not establish that values are public;
hosts authorize their selected portable inventories. Direct session initialization has
its broader basic shape contract. Hosts should install an explicitly selected portable environment before
creating tasks that require portable execution.

The shared storage profile counts original UTF-8 data, typed entry storage and exact
JSON encoding without allocating a wire copy. Scheduling rejects a reservation smaller
than this component; standalone native/browser requests add it to their base reservation.
This is a minimum component reservation. Allocator overhead, retained copies, runtime
maps, other execution storage and measured process RSS remain distinct obligations.

## Bounded console source contract

`runmat_execution::console` owns ordered stdout, stderr and clear-screen records.
Records retain original chunks and UTF-8 text without newline insertion, timestamp
sorting or mixing console data into return-value ordinals. A transcript states complete,
interrupted or rejected capture. Decoding and validating its ordered inventory does not
authenticate a producer's completion claim. Progress buffers are not transcript authority.

`PortableConsoleHost` implements the existing runtime host port, and sessions install it
with `RunMatSession::install_host_service`. The caller supplies admitted record/text limits;
there is no implicit limit policy or independent reservation. The host owns the normal
console inventory, avoiding a duplicate unbounded runtime buffer. `evalc` remains a separate
language capture and does not appear in the host transcript. Interleaved invocations keep
their respective hosts; nested calls inherit the original session service.

Overflow rejects the write before append, latches failure and preserves the accepted
prefix. Builtin dispatch and the portable invocation boundary check that failure, so a
caught language exception cannot turn lost output into successful completion. The host
seals once and moves its original inventory. It rejects interactive line/key requests,
including requests with an ambient queued response, without changing user cancellation.

The capture storage component counts requested record slots and UTF-8 payload bytes.
It excludes allocator overhead, formatting and `evalc` buffers, warning/diary state,
retained copies and wire encoding; it is not whole-task admission or measured RSS.
Worker streaming, request-bound limits, original attempt fencing, durable/cold transcript
reads and browser/remote transport remain required integration gates. Existing worker
routes do not acquire a complete transcript merely because this source API is available.

Caller-executed await handoffs now move the original deferred call out of serial,
native and browser service state. The await action and native suspension keep one
boxed carrier instead of cloning captured arguments, program bytes and environment.
The original handle and Running/ExecutingInCaller transition still govern completion,
cancellation and rejection of a second concurrent await.
