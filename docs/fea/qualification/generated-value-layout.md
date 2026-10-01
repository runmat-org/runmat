# Generated tensor and spatial-field value layout

Status: bounded shared input-layout correction. No formulation, stage or production
promotion; authored post-mesh field selection and whole-study preprocessing resource,
publication and restart gates remain open.

Runtime MATLAB dense arrays use column-major storage. Generator output axes are
sample-leading: `[samples, tail...]`; canonical signal values are sample-major with
row-major tensor tails. A nodal vector therefore stores contiguous components of
node zero, then node one, in the authenticated canonical node order. Previously
only the sample axis was reordered, leaving all nodes' first components before
their second components. Equal-valued fixtures concealed the error.

One physics-neutral conversion now handles scalar/vector, arbitrary tensor and
topology-field real/complex values. It checks positive extents, overflow and exact
cardinality before conversion, allocating only the final reordered value buffer.
The PSD path uses the same conversion and removes its separate matrix transpose;
existing signed imaginary cross-phase evidence must stay unchanged.

Frozen signal admission requires the explicit `sample_major_row_major` value-layout
contract, included in the realization-policy identity. Missing/unknown old layout
contracts are rejected; old sealed values are not silently reinterpreted. Already
canonical authored tables are not reordered. There is no old-object compatibility
path. Core canonical signal-value ownership is extracted into its value module;
the existing public type and serialized representation remain unchanged.

The exact native field callable is `reshape(t * scale, numel(t), N, C)`, with the
authenticated fixture mesh supplying N and declared scalar/vector shape supplying
C. Shared `numel` semantic metadata is corrected from unknown to pure: its current
implementation reads resident array/handle metadata without workspace mutation,
device payload gathering or suspension. Generator admission remains unchanged;
unknown/effectful calls still fail closed. This does not claim GPU execution.

Evidence inventory:

- The distinct-value Tet4/Tet10 node-vector regression fails before correction:
  node-zero Y receives node-one X (`10` instead of `1`). After correction it checks
  every sample, node and component through canonical sealing and solver tables.
- Literal non-square real/complex `[2,2,3]` storage and rank-three tensor tails
  independently check coordinates; invalid extents/cardinalities fail closed.
- Existing scalar/vector and Hermitian complex PSD tests guard unchanged paths.
- Process conformance includes existing vector input plus Tet4/Tet10 nodal body-force
  density and temperature fields. Each field has five samples in three chunks and
  compares every canonical value/digest with independently constructed row-major
  tables across inline, isolated, durable and encrypted-remote task placement.

The process fixture starts from an already authenticated canonical mesh, not authored
`.fea` spatial syntax or deferred CAD-derived mesh selection. Durable task completion
is not failure/replacement restart. Remote generator tasks are not remote whole-study
spatial execution or distributed operators. Physical body-field quadrature/dynamics
evidence remains separately governed; these layout tests do not replace it.
