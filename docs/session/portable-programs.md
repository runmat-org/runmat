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

## Script outputs

A fresh script with zero requested outputs returns an empty output inventory. One
requested output requires an actually assigned `ans`. An allocated but unassigned
slot, an unrelated final variable, and a branch that never assigned `ans` are not
results. Scripts reject multiple output ordinals.

The portable interpreter uses the VM's existing assignment inventory. Native script
tiering currently has a zero-output contract; forcing native execution while
requesting an `ans` output is rejected explicitly.

## Qualification boundary

This API provides session invocation of a portable compiler product. It does not
by itself complete the legacy batch/script cutover, console transport, host-policy
admission, nested process allocation, or placement/resource qualification. Those
require their complete original host contracts and independent execution witnesses.
