# Geometry features and STEP AP242

## Feature operations

Constructed and derived geometry uses a bounded typed feature vocabulary:

- primitives: box, cylinder, and sphere;
- rigid transform and instance;
- extrude and revolve;
- governed union, intersection, and subtraction;
- hole and pocket;
- qualified fillet and chamfer;
- typed selections derived from feature history.

Every argument has a physical type and each operation returns a typed feature result. A declaration
names one final `result`. Unknown operations, invalid dimensions, unsupported provider
capabilities, an empty Boolean result, a no-op cut, and selection-cardinality failure stop
materialization. There is no raw geometry-kernel opcode in source.

## Persistent topology

Imported and generated geometry publish occurrence-qualified persistent entity identities and
feature history. Regions may select imported persistent ids, named/tagged entities, a feature
result, or boundaries introduced by a feature. The final identity is digest-bound; source ordinals
and kernel-private shape keys do not become public topology identity.

## AP242 inspection

RunMat inventories AP242 capabilities per file. The inventory can independently report product
structure, names, persistent ids, readable parameters, writable parameters, constraints/features,
material candidates, property provenance, units, and qualified regeneration support. Missing
capability is represented as missing; it is not inferred from `FILE_SCHEMA`.

## CAD parameters

A `cad_parameter` declaration names its physical type, selector, cardinality, and required
capabilities. Reading a parameter retains source entity, unit, value, and provenance. A source
parameter update is allowed only when the admitted provider qualifies writing that parameter type
and the resulting geometry passes the same exact validation and identity pipeline.

## Engineering properties

CAD material and property candidates are not active FEA materials. A material declaration imports
one candidate explicitly, lists each adopted capability with `use`, states environmental
requirements, and selects a conflict policy. The resulting material retains source provenance but
is still assigned to geometry through the ordinary model assignment contract.

## Export and round trip

An export claim states exactly which product structure, geometry, parameters, persistent ids, and
properties were preserved, added, changed, or omitted. Round-trip qualification is fixture- and
capability-specific. RunMat does not describe a file as AP242-complete when only its boundary
representation was retained.
