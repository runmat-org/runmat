# Geometry features and AP242

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

## Examples

Constructed geometry forms an immutable feature graph and names one result:

```fea
geometry clevis {
    input pin_radius : length = 6 mm
    construct {
        feature blank = box(width = 80 mm, depth = 30 mm, height = 16 mm)
        feature pin_hole = cylinder(radius = input.pin_radius, height = 16 mm)
        feature finished = subtract(feature.blank, feature.pin_hole)
        result = feature.finished
    }
    region body : solid {
        select result
        expect exactly_one
    }
}
```

Imported CAD regions should use stable names, tags, or persistent ids published by inspection:

```fea
geometry bracket {
    file = "./bracket.step"
    units = mm
    region fixed_pad : face {
        select name "FIXED_PAD"
        expect exactly_one
    }
    region fastener_holes : face {
        select tag "FASTENER_HOLE"
        expect one_or_more
    }
}
```

CAD metadata becomes active only through an explicit declaration:

```fea
material cad_steel from geometry.bracket.cad_material("steel-4340") {
    use mass_density
    use elastic.youngs_modulus
    require operating_temperature = 20 degC
    on_conflict = reject
}
```
