# Shared post-mesh signal binding boundary

This is a bounded input-contract verification, not qualification of the complete
authored spatial-field workflow or a qualified physics envelope.

## Retained behavior

Topology-dependent generator preparation requires an exact canonical mesh binding:
mesh digest, field location, row count, component width, and ordered row identity.
The authored topology identifier must already name that mesh; preparation does not
rewrite an opaque identifier or infer topology from generator output dimensions.
Node, cell/element, boundary-face, and boundary-edge inventories use canonical mesh
order. Integration-point, point, and interface fields reject without an authoritative
domain-specific row inventory. Ordinary scalar/vector/tensor signals stay unbound.

Bindings survive prepared-input transport and expected-result serialization. Local
dispatch authenticates them before executing a generator. Canonicalization receives
the bound entity count, and cold exact-study import reauthenticates count/order against
the imported mesh before consuming the immutable realization. This imports retained
input values; it does not rerun their generator or reconstruct missing input values.

Local structural, thermal, and acoustic composition compiles the model, seals inputs,
and then derives numerical work. The thermal/acoustic product helpers retain their
original observation-resource/material admissions and authenticate the compiled model
against the mesh. No numerical solver, quadrature, accuracy threshold, iteration cap,
or qualification gate changes.

The transported contracts now explicitly include the optional binding. Their canonical
bytes/identities change; no old-API or old-object identity compatibility is promised.

## Evidence

An isolated HEAD6f25bb83e plus the selected input-boundary changes verifies:

- Five binding tests: Tet4/Tet10 node rows for scalar temperature and three-component
  body density; all admitted domain orders; canonical/JSON transport; foreign mesh,
  digest, count, width, location, and row-order rejection; unsupported locations,
  overflow/zero cardinality, and unknown fields.
- A transported expectation/canonical realization test across both orders and both
  quantities. Missing/foreign bindings and changed reopened row counts reject.
- Acoustic product equality and independent material/foreign-mesh admission tests.
- Native named thermal-input regression compares checked callable outputs with
  explicit tables and independent source/boundary-energy evidence. This is existing
  scalar-input coverage, not native generated spatial-field qualification.
- FEA/bridge/native strict no-deps library/test lint and CLI test compilation.

The combined previously retained FEA graph finishes with 1,338 passes and seven
failures: load-frame exact floating-point comparison, adaptation recovery estimator,
electrical and modal symbolic proof resource limits, and three transient continuum
gates. This input boundary does not resolve or waive them. Broad CI/release is not green.

## Remaining vertical-chain work

Callable admission must be separable from mesh-dependent chunk planning. Authored
spatial selectors and canonical coordinates, direct immutable field artifacts, checked
preprocessing/coexistence resources, and remote mesh-before-input sequencing remain.
The current remote path rejects topology-dependent inputs before dispatch when no
authoritative mesh is available. Existing cold exact structural workloads also need
post-sealing resource replanning; replacing only their model is not sufficient for a
newly admitted spatial field. The binding checks do not claim to close that gap.

Actual generated-field source/native/remote execution, independent field projection
and conservation, whole-root cold replay/restart, advertised placement/resource
evidence, and blocking release gates are still required by the completion plan.
