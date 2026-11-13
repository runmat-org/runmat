# Geometry

Geometry is an immutable, typed input to meshing and analysis. A `.fea` declaration may import CAD, construct a feature graph, derive a new exact shape from another declaration, or call one exact pure `.m` geometry function. Every route converges on the same exact-geometry artifact, persistent topology, region, lineage, and provenance contracts.

## Inspect Or Load From `.m`

```matlab
facts = geometry.inspect("geometry/bracket.step");
part = geometry.load("geometry/bracket.step", "Units", "millimeter");
regions = geometry.listRegions(part);
mesh(part);
```

Inspection answers which exact format, units, assemblies, products, parameters, materials, names, tags, and canonical persistent identities the file actually contains. Do this before authoring region selectors.

## Generated CAD Binding Module

```fea
geometry bracket {
    file = "./geometry/bracket.step"
    units = mm

    region body : solid {
        select name "body"
        expect exactly_one
    }
    region mount : face {
        select name "Mounting Face"
        expect one_or_more
    }
}
```

Accepted authored coordinate units are meters, millimeters, and inches. The source path is confined to the frozen project. Exact execution re-reads content-addressed bytes and verifies the inspected topology/material/parameter inventory before geometry can enter a mesh or solve.

## Supported Input Families

| Format | Analysis use |
| --- | --- |
| STEP | Exact CAD topology plus AP242/XCAF product, assembly, label, layer, color, parameter, material, and property evidence when authored by the source tool |
| IGES | Exact CAD topology and available XCAF names/layers/colors |
| BREP | Exact native OCCT topology; normally little exchange metadata |
| STL, OBJ, PLY, glTF | Tessellated or scene geometry for inspection/display and workflows that do not require exact CAD recovery |

Tetrahedral solver meshing requires an admitted closed exact solid. A format being readable does not imply that a particular file contains solid topology, named boundaries, units, materials, or parameters.

Official native CLI and Desktop builds use the OCCT backend for STEP, IGES, and BREP. Metadata-only builds can inspect supported STEP exchange facts without claiming exact face/solid topology. Browser execution remains unavailable until its configured geometry, meshing, and solver host can satisfy the same contracts.

## Engineering source and CAD selections

Keep the geometry declaration, named regions, and physics in one study source:

```fea
geometry bracket {
    file = "./bracket.step"
    units = mm
    region load_face : face { select name "load_face"; expect exactly_one }
}
```

`select name "load_face"` resolves only against names authored inside the CAD file:
STEP/IGES product, label and face names carried through XCAF. A name the file does
not carry is a check error that lists the names the file does carry. There is no
side file of saved selections; to name a face, name it in the CAD tool (or with a
STEP labeling step such as `geometry/name-step-faces.cc`) and re-export.

Every name resolves once, when the study is checked and materialized, to the canonical
`persistent-entity:sha256:` identity of the matching entity. The solver sees only those
canonical identities. Geometry without authored names (for example a native BREP)
selects entities by that canonical identity directly, as returned by geometry
inspection or by a geometry selection tool:

```fea
region support : face { select id "persistent-entity:sha256:8af3…"; expect exactly_one }
```

Source aliases such as `face_000001` are not identities and are rejected.
Do not create `<study>.geometry.fea` files for machine bindings. Ordinary imports remain
available for deliberate engineering reuse.

Replacing a CAD file invalidates its recorded source digest. Reattach selections through
the existing admitted revision map, or review unresolved selections. Never infer identity
from a display label, face ordinal, or geometric proximity.

## Regions And Selectors

A region declaration has three independent facts:

1. a name used by the model;
2. a topology role (`vertex`, `edge`, `face`, `solid`, or `cell`);
3. a selector plus `exactly_one` or `one_or_more` cardinality.

Implemented selectors include:

| Form | Meaning |
| --- | --- |
| `select id "..."` | role-scoped alias published by source inspection |
| `select persistent_id "..."` | canonical persistent entity identity |
| `select name "..."` | imported authored face or solid name (see below) |
| `select tag "..."` | imported presentation layer (STEP `PRESENTATION_LAYER_ASSIGNMENT`) on a face or solid, or on its part |
| `select result` | final result of constructed geometry |
| `select feature.<id>` | exact output of a feature |
| `select boundary_introduced_by feature.<id>` | boundary introduced by one feature operation |
| `select output.region.<id>` | region returned by generated geometry |

CAD label entries, display face numbers, and unqualified topology names are metadata, not alternate identities. Inspection, picking, authored persistent selectors, and execution share the exact importer's `PersistentEntityId`; the string key is derived by that type alone. Copy the actual inspected key in place of the illustrative placeholder above.

Re-importing an unchanged entity preserves its identity where the exact topology and occurrence remain unchanged. Editing a shape or re-exporting an assembly may change those facts. Reattachment uses an admitted `GeometryRevisionMap`, with explicit retain, replace, split, merge, and delete evidence. Region remapping is atomic: a deleted, split, unmapped, or absent target requires review; names, ordinal position, and proximity never choose a replacement. A fresh STEP file does not itself provide CAD feature history, so automatic correspondence after arbitrary edits is not assumed.

Selectors are resolved at exact realization and then projected onto the solver mesh. Wrong topology, empty selection, cardinality mismatch, missing lineage, or changed source identity fails before assembly.

### Authored names

`select name` matches the names a CAD tool wrote into the file, exactly and case-sensitively:

- a face name written on the STEP face (`ADVANCED_FACE('Fixed end', ...)`);
- a solid name written on the STEP solid (`MANIFOLD_SOLID_BREP('Coil', ...)`), or else the name of the part (product) that holds the solid. A `cell` region selects by its solid's name.

`runmat check` rejects a `select name` that the file does not carry and lists the names it does carry for that role. A file without names (such as a native BREP) selects entities by their canonical `persistent-entity:sha256:` ID from geometry inspection. Source aliases such as `face_000001` are rejected.

## Multi-Body Geometry And Shared Interfaces

A STEP assembly of separate parts imports as separate solids, each with its own faces. Parts that touch keep independent faces at the contact, which is what structural contact and the `interface` pairs in [constraints/interfaces.md](./constraints/interfaces.md) need.

For a conformal multi-body mesh (a part inside an air region, a coil and core in air, fluid against a solid wall), join the bodies:

```fea
geometry cell {
    file = "./geometry/induction-gear.step"
    units = mm
    interfaces = shared
    region gear : solid { select name "Gear"; expect exactly_one }
    region air : solid { select name "Air"; expect one_or_more }
    region coil_in : face { select name "Coil terminal in"; expect exactly_one }
}
```

With `interfaces = shared`, every solid of every placed part enters one General Fuse. Where two solids touch, their coincident faces become one face used by both solids, imprinted where a smaller face covers part of a larger one. The mesh then shares nodes across that face, which is classified as a conformal interface between the two volume regions. `interfaces = separate` is the default.

- Bodies may touch but must not interpenetrate. Import fails, naming the bodies, when a solid would be split by another.
- Each solid keeps its own name, or its part's name. A shared face carries the names authored on every face it replaced, the name from the smaller solid first, so `select name` finds it by the part's face name ("Coil terminal in") or by the surrounding body's name for that face if one was written.
- Geometry inspection reports a canonical `persistent-entity:sha256:` ID for every face of the joined geometry.
- A single STEP part whose solids already share faces (a General Fuse result written as one body) imports with those shared faces either way.
- A solid with several cavities (an air box around a gear and a coil) imports as one solid with one void shell per cavity.

From the command line, `runmat mesh part.step --shared-interfaces` meshes the joined geometry.

## Construct Geometry

```fea
geometry bracket {
    input width : length = 80 mm
    input thickness : length = 8 mm
    construct {
        feature stock = box(
            width = input.width,
            depth = input.thickness,
            height = 50 mm,
        )
        feature hole = cylinder(
            radius = 5 mm,
            height = input.thickness,
            axis = global([0, 1, 0]),
            origin = [15, 0, 15] mm,
        )
        feature finished = subtract(feature.stock, feature.hole)
        result = feature.finished
    }
    region body : solid { select result; expect exactly_one }
}
```

The closed feature set includes boxes, cylinders, spheres, transforms, instances, extrusion, revolution, Boolean union/subtraction/intersection, holes, pockets, fillets, chamfers, and qualified regeneration. Values are unit-checked; feature dependencies form an acyclic immutable graph.

## Derive Geometry

```fea
geometry relieved from geometry.bracket {
    derive {
        feature relief = cylinder(
            radius = 8 mm,
            height = 8 mm,
            axis = global([0, 1, 0]),
            origin = [50, 0, 25] mm,
        )
        result = subtract(source.result, feature.relief)
    }
}
```

Derived declarations keep the exact source closure and compose provider-proven created/modified/deleted topology lineage through every feature. They do not mutate the source declaration.

## Generate Geometry With RunMat

```fea
import function makeBracket from "./makeBracket.m"

geometry bracket {
    input width : length = 80 mm
    generated by function.makeBracket {
        arguments { width = input.width }
    }
    region body : solid {
        select output.region.body
        expect exactly_one
    }
}
```

The imported `makeBracket` symbol is defined in `makeBracket.m`:

```matlab
function part = makeBracket(width)
arguments
    width (1,1) double
end

depth = 0.05;
height = 0.008;
stock = geometry.box(width, depth, height);
hole = geometry.cylinder(0.005, height, [0 0 1], ...
    [width / 2, depth / 2, 0]);
body = geometry.subtract(stock, hole);

part = geometry.finish(body);
part = geometry.withRegion(part, "body", geometry.solid(body));
end
```

The `.fea` length input reaches the function as meters. The return value is an immutable typed
geometry value, and the named `body` region is what makes `output.region.body` a valid selector.

The imported function must resolve to the exact frozen `.m` definition, satisfy the typed geometry signature, and be pure and non-suspending. It returns a portable typed geometry value; a worker cannot replace it with arbitrary file or process output.

The same typed geometry value can be built directly in `.m` code:

```matlab
stock = geometry.box(0.08, 0.05, 0.008);
bore = geometry.cylinder(0.005, 0.008, [0 0 1], [0.04 0.025 0]);
body = geometry.subtract(stock, bore);
bodyRegion = geometry.solid(body);
part = geometry.finish(body);
part = geometry.withRegion(part, "body", bodyRegion);
mesh(part);
```

The programmatic operation set includes box, cylinder, sphere, transform, instance, extrude,
revolve, union, intersection, subtraction, hole, pocket, fillet, chamfer, finalization, and typed
solid/face/edge or feature-introduced region outputs. These functions build immutable geometry
values; display through `mesh(part)` does not discretize them for a solver.

## AP242 Parameters And Engineering Properties

AP242 parameters can be named and required as readable or writable facts. Typed parameter values can feed study inputs or derived geometry. Material candidates and engineering properties may be adopted explicitly into an analysis-owned material declaration; visual color or an incomplete elasticity record is not a physical material.

RunMat reports source capabilities honestly. Reading and sidecar change-set generation are distinct from regenerating a STEP file. If the active provider cannot perform a qualified write/regeneration, the request fails closed rather than claiming the source was updated.

## Geometry To Mesh To Model

The geometry declaration owns topology. The mesh declaration owns discretization. The model assigns materials and other properties to typed geometry regions. Exact projection publishes canonical element ownership and persistent CAD provenance so every solver family consumes the same resolved property map instead of repeating selector logic.

See [Meshing](./meshing), [Models](./models), and the [language geometry reference](./language/geometry-models-and-assignments).
