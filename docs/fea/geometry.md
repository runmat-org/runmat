# Geometry

Geometry is an immutable, typed input to meshing and analysis. A `.fea` declaration may import CAD, construct a feature graph, derive a new exact shape from another declaration, or call one exact pure `.m` geometry function. Every route converges on the same exact-geometry artifact, persistent topology, region, lineage, and provenance contracts.

## Inspect Or Load From `.m`

```matlab
facts = geometry.inspect("geometry/bracket.step");
part = geometry.load("geometry/bracket.step", "Units", "millimeter");
regions = geometry.listRegions(part);
mesh(part);
```

Inspection answers which exact format, units, assemblies, products, parameters, materials, topology aliases, names, tags, and persistent identities the file actually contains. Do this before authoring region selectors.

## Import In `.fea`

```fea
geometry bracket {
    file = "./geometry/bracket.step"
    units = mm

    region body : solid {
        select id "solid_000001"
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
| `select name "..."` | imported authored name |
| `select tag "..."` | imported semantic tag |
| `select result` | final result of constructed geometry |
| `select feature.<id>` | exact output of a feature |
| `select boundary_introduced_by feature.<id>` | boundary introduced by one feature operation |
| `select output.region.<id>` | region returned by generated geometry |

Selectors are resolved at exact realization and then projected onto the solver mesh. Wrong topology, empty selection, cardinality mismatch, missing lineage, or changed source identity fails before assembly.

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
    region body : solid { select result expect exactly_one }
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
