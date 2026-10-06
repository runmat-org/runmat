# Portable compiler products and session invocation

A V3 executable unit carries the original MIR, analysis, bytecode, VM layout,
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
- The low-level VM request executor rejects complete V3 units. Native assessment
  and signal adapters, and browser artifact execution, enter the core session host.
  They do not reconstruct only bytecode and registry state.
- Program context is installed on native and WASM targets. Its active-unit scope
  restores the previous callable authority after nested invocation and expires
  when invocation ends.
- Browser in-process execution inherits the existing browser execution service.
  This does not establish nested worker allocation or console/project transport.

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
future creation ask the active core program service for the complete V3 product
of the exact semantic function. They preserve the original revision, source map,
analysis and compiler components. Missing program authority fails explicitly.
Raw script execution also enters the original core program scope on native and
WASM targets. Its interpreted call uses that product's revision in a scoped runtime
context; the session's outer revision is restored after execution.
Serial in-context services may retain a future without portable capture.

Artifact construction belongs to the shared artifact owner. Deferred materialization
requires canonical V3 bytes, the same original revision and function identity, and
the original output contract. Partial registries and absent revisions reject; the
consumer does not invent a replacement program revision. Native executor/JIT state
no longer retains a second registry-only capture for this purpose. Standalone AOT
manifests retain their existing owner and do not substitute for a V3 capture.

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
