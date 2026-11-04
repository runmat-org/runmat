---
description: How .fea geometry declarations import, construct or derive CAD and name regions from CAD-authored names; how meshes, materials and models assign materials, sections, orientations, loads, supports and signals to those regions.
keywords: [.fea geometry, CAD regions, select name, select tag, STEP AP242, interfaces shared, construct, derive, CAD parameter, mesh declaration, tetrahedra, triangles, lines, material assignment, beam section, shell section, laminate layup, signal, excitation, load case]
doc_type: language-reference
examples: [cantilever-static]
---
# Geometry, models and assignments

This page is the syntax reference for geometry, models and assignments. To set up a study, see [Geometry](../geometry) and [Models](../models).

A study needs a shape and a physical description of it. The `geometry` declaration owns the shape: where it comes from, its unit, and its named regions. The `model` declaration owns the physics placed on those regions: materials, sections, orientations, supports, loads and signals. One CAD revision and one mesh can therefore serve several material or loading studies, and a model can move to a revised CAD file as long as the region names stay the same.

This page covers the tasks in the order a study is written: bring in geometry, name its regions, mesh it, define materials, assign them, and apply supports and loads. The exact members, types and defaults of each declaration are in the [declaration reference](./reference/declarations.md).

## Import a CAD file

A geometry imports one STEP (`.step`, `.stp`), IGES (`.iges`, `.igs`) or BREP (`.brep`) file:

```fea source=cantilever-static.fea
geometry beam {
    file = "./geometry/cantilever.step"
    units = mm
    region body : solid { select name "Beam"; expect exactly_one }
    region fixed_end : face { select name "Fixed end"; expect exactly_one }
    region loaded_end : face { select name "Loaded end"; expect exactly_one }
}
```

- **`file`** is a path relative to the `.fea` file.
- **`units`** (`mm`, `m` or `in`) gives the length unit of a file that states none, such as a BREP file. A STEP file carries its own length unit, so `units` is optional there; when present, it must agree with the file.
- **STEP is the preferred format.** Besides exact geometry, a STEP AP242 file can carry face and body names, presentation layers, CAD parameters, materials and persistent entity ids. The checker records which of these the file actually contains, and a study that needs one the file lacks is a check error.

### Join touching bodies into one conformal mesh

A multi-body file, such as a plate with coolant passages or a bolted assembly, imports each body separately by default (`interfaces = separate`). With `interfaces = shared`, touching bodies share their coincident faces, and the mesh is conformal across them:

```fea source=cold-plate.fea
geometry plate {
    file = "./geometry/cold-plate.step"
    units = m
    interfaces = shared
    region coolant : cell { select name "Coolant"; expect one_or_more }
    region plate : solid { select name "Plate"; expect exactly_one }
    region inlet : face { select name "Inlet"; expect one_or_more }
    region outlet : face { select name "Outlet"; expect one_or_more }
    region channel_wall : face { select name "Channel wall"; expect one_or_more }
    region plate_wall : face { select name "Plate wall"; expect one_or_more }
    region heater : face { select name "Heater"; expect exactly_one }
}
```

Use `shared` when the bodies are bonded or exchange heat or load across a common face. Keep them `separate` when they meet through a contact or bonded [interface](../constraints/interfaces.md) declared in the model.

## Name regions

A region names part of the geometry so the model can refer to it. Each region declares one topology role, `vertex`, `edge`, `face`, `solid` or `cell`, and one selector. An optional `expect exactly_one` or `expect one_or_more` states how many entities the selector must match; a selector that matches a different number is a check error, never a silent first match.

| Selector | Matches |
| --- | --- |
| `select name "Fixed end"` | entities with that name authored in the CAD file |
| `select tag "BOLT_HOLES"` | entities on that CAD presentation layer |
| `select persistent_id "persistent-entity:sha256:…"` | one entity by its canonical persistent id |
| `select feature.<f>` | the entity a construction feature produced |
| `select boundary_introduced_by feature.<f>` | the boundary a construction feature introduced, such as a hole's wall |
| `select result` | the result of a `construct` block |
| `select output.region.<r>` | a region returned by an `.m` geometry generator |
| bounded geometric predicates, such as faces within a box [not yet in the current release](pending:geometry.select.predicate) | entities whose position or normal satisfies the predicate |

Region names come only from the CAD file. To select a face the file leaves unnamed, name it in the CAD tool and re-export, or select it by its persistent id. Each name resolves once, at import, to canonical persistent ids; from then on the mesh, the solver and the results identify the region by those ids.

```fea fragment
geometry bracket {
    file = "./geometry/bracket.step"
    region body : solid { select name "Body"; expect exactly_one }
    region mounting_faces : face { select tag "MOUNTING"; expect one_or_more }
}
```

## Construct geometry from features

A `construct` block builds a solid from primitives and Boolean operations. Each feature is immutable, so a construction is a feature graph that regenerates when an input changes:

```fea fragment
geometry stock {
    input width : length = 120 mm

    construct {
        feature body = box(width = input.width, depth = 80 mm, height = 12 mm)
        feature hole = cylinder(
            radius = 5 mm,
            height = 12 mm,
            axis = global([0, 0, 1]),
            origin = [60 mm, 40 mm, 0 mm],
        )
        feature finished = subtract(feature.body, feature.hole)
        result = feature.finished
    }

    region body : solid { select result; expect exactly_one }
    region hole_wall : face { select boundary_introduced_by feature.hole; expect one_or_more }
}
```

A geometry can also come from an `.m` function (`generated by`) that returns the shape and its named regions; see [RunMat function integration](./runmat-function-integration.md).

## Derive a variant from CAD parameters

A derived geometry regenerates another geometry with changed CAD parameters and records its lineage. The source geometry is unchanged, so a study can compare both:

```fea fragment
geometry housing {
    file = "./geometry/housing.step"
    cad_parameter wall : length { select persistent_id "persistent-entity:sha256:…"; require writable }
    region body : solid { select name "Housing"; expect exactly_one }
}

geometry thin_wall from geometry.housing {
    derive {
        set source.parameter.wall = 2 mm
        result = regenerate(source)
    }
}
```

`require readable` exposes the parameter as `geometry.housing.parameter.wall`; `require writable` also allows a `derive` block to set it. A parameter change invalidates only the derived geometry and what depends on it: its mesh, operators and results.

## Mesh a geometry

A `mesh` declaration belongs to one geometry and holds discretization controls only:

```fea source=cantilever-static.fea
mesh beam on geometry.beam {
    tetrahedra {
        order = quadratic
        target_size = 0.15 m
        maximum_deviation = 0.001 m
    }
}
```

| Block | Meshes | Elements |
| --- | --- | --- |
| `tetrahedra` | solids and cells | linear (Tet4) or quadratic (Tet10) tetrahedra |
| `triangles` | sheet faces, as shells | shell triangles; their boundary edges become segments |
| `lines` | edges, as beams | beam segments |

The controls are target size, maximum surface deviation, grading ratio, cells across a gap, a deterministic seed, and local `refine` blocks on regions. Their types and defaults are in the [declaration reference](./reference/declarations.md#mesh), and the [meshing page](../meshing.md) explains how to choose them. The study must bind the geometry the mesh was declared on. Every mesh carries its regions onto the solver's nodes and elements, and publishes its quality metrics.

## Define materials

A material is a set of typed property blocks. An analysis step uses the blocks it needs: a static structural step needs `elastic`, a modal or transient step also needs `mass_density`, a heat-transfer step needs `thermal`.

```fea fragment
material aluminum {
    name = "Aluminum 6061-T6"
    mass_density = 2700 kg/m^3
    elastic {
        youngs_modulus = 68.9 GPa
        poisson_ratio = 0.33
    }
    thermal {
        conductivity = 167 W/m/K
        specific_heat = 896 J/kg/K
        expansion_coefficient = 23.6e-6 K^-1
        reference_temperature = 20 degC
        modulus_temperature_coefficient = 0 K^-1
    }
}
```

Blocks exist for isotropic elastic, orthotropic lamina, J2 plastic, thermal, electrical (conductivity, permittivity and permeability), acoustic, fluid and porous behavior, and for Rayleigh damping. The full member list is in the [declaration reference](./reference/declarations.md#material). A step that needs a block the assigned material lacks is a check error.

## Assign materials, sections and orientations

Assignments live in the model and target regions of the right role:

| Region role | Takes |
| --- | --- |
| `solid`, `cell` | a bulk material |
| `face` meshed with `triangles` | a material with a shell `section`, or a `layup`; and an `orientation` |
| `edge` meshed with `lines` | a material with a beam `section`, and an `orientation` |

A boundary face does not own the material of the solid behind it. Before any coefficient is assembled, RunMat confirms that every element has exactly one owner of each kind it needs. Missing coverage, two owners for one region, a material without the block a step needs, or a target of the wrong role is reported with the line of the assignment.

### Beams

A frame of edges takes a material, a beam section and an orientation on each member:

```fea source=portal-frame.fea
    section member { rectangular_beam {
        width = 100 mm
        height = 100 mm
    } }
    orientation frame_axes { global_cartesian {} }

    assign material.steel to geometry.frame.region.left_column
    assign material.steel to geometry.frame.region.girder
```

### Homogeneous shells

A shell face takes three assignments: material, section and orientation.

```fea source=square-plate.fea
    section plate_section {
        homogeneous_shell {
            thickness = 10 mm
            reference_surface = mid_surface
            shear_correction = 0.8333333333333333
        }
    }
    orientation plate_axes { global_cartesian {} }

    assign material.steel to geometry.plate.region.panel
    assign section.plate_section to geometry.plate.region.panel
    assign orientation.plate_axes to geometry.plate.region.panel
```

### Composite laminates

A composite face takes a layup and an orientation. Each ply names its own material, so the face takes no separate material assignment.

```fea fragment
material carbon_epoxy {
    mass_density = 1600 kg/m^3
    orthotropic_lamina {
        longitudinal_modulus = 135 GPa
        transverse_modulus = 10 GPa
        major_poisson_ratio = 0.30
        in_plane_shear_modulus = 5 GPa
        transverse_shear_modulus_13 = 5 GPa
        transverse_shear_modulus_23 = 3.8 GPa
    }
}

model panel {
    orientation fibers {
        cartesian {
            origin = [0 m, 0 m, 0 m]
            x_axis = [1, 0, 0]
            y_axis = [0, 1, 0]
        }
    }

    layup skin {
        reference_orientation = orientation.fibers
        ply_order = along_surface_normal
        reference_surface = mid_surface
        ply lower {
            material = material.carbon_epoxy
            thickness = 0.2 mm
            angle = 0 deg
        }
        ply upper {
            material = material.carbon_epoxy
            thickness = 0.2 mm
            angle = 90 deg
        }
    }

    assign layup.skin to geometry.panel.region.skin
    assign orientation.fibers to geometry.panel.region.skin
}
```

Ply order, ply angle and reference surface are named choices, not sign conventions. The [models page](../models.md#composite-layups) gives the laminate equations and how mass, damping and ply stresses are computed.

## Apply supports and loads

A support is a `constraint` on a region; a load with a known value uses the direct form `<kind> on <region> = <value>`. A `load_case` groups the loads a step applies.

```fea source=cantilever-static.fea
model loaded_beam {
    assign material.steel to geometry.beam.region.body
    constraint fixed_end {
        structural.fixed { target = geometry.beam.region.fixed_end }
    }
    load end_traction {
        traction on geometry.beam.region.loaded_end = [0, 0, -100000] Pa
    }
    load_case service { include load.end_traction }
}
```

Each family's supports and loads, with their units and meaning, are on the [constraints](../constraints/index.md) and [loads](../loads/index.md) pages. Their exact fields are in [constraints, loads and interfaces](./reference/constraints-loads-and-interfaces.md).

## Drive loads with signals and excitations

A load that varies in time or frequency splits into a `signal`, which owns the quantity, its domain, samples and interpolation, and an `excitation`, which owns the target, direction and frame. One signal can drive several excitations, and RunMat Desktop shows each target with the signal attached to it.

```fea source=structural-transient.fea
    signal pulse : force over time {
        samples = [
            { at = 0 s, value = 0 N },
            { at = 0.01 s, value = 1 kN },
            { at = 0.02 s, value = 0 N },
            { at = 0.1 s, value = 0 N },
        ]
```

```fea source=structural-transient.fea
    excitation impact {
        force {
            target = geometry.bracket.region.load_face
            signal = signal.pulse
            direction = global([0, -1, 0])
        }
    }
    load_case impact { include excitation.impact }
```

Signals can also be standard waveforms (`sine`, `step`, `ramp`, `half_sine_pulse`, `rectangular_pulse`, `triangular_pulse`), tables, or `.m`-generated samples. See [loads](../loads/index.md) and [RunMat function integration](./runmat-function-integration.md).

## Read next

- [Studies, analyses and state](./studies-analyses-and-state.md): bind a geometry, mesh and model into analysis steps.
- [Declaration reference](./reference/declarations.md): every member, type and default.
- [Meshing](../meshing.md) and [geometry](../geometry.md): choosing mesh controls and preparing CAD names.
