---
description: How .fea geometry declarations import, construct or derive CAD, name regions from CAD-authored names, mesh them, and how models assign materials, sections, loads and signals.
keywords: [.fea geometry, CAD regions, select name, STEP AP242, mesh declaration, material assignment, shell layup, signals, excitations]
doc_type: language-reference
---
# Geometry, models and assignments

Geometry defines topology; a model gives that topology physical meaning. Keeping the two apart lets one CAD revision and one mesh serve several material or loading studies.

## Imported geometry and regions

Region names come from the CAD file itself: the labels authored in a STEP or IGES file (`select name`) and its presentation layers (`select tag`). To name a face the CAD file leaves unnamed, name it in the CAD tool and re-export, or select it by its canonical persistent id. Each name resolves once, at import, to canonical persistent ids, and the solver matches regions by id only.

```fea fragment
geometry bracket {
    file = "../examples/geometry/box.brep"
    units = m

    region body : solid {
        select name "body"
        expect exactly_one
    }
    region mounting_faces : face {
        select tag "mounting"
        expect one_or_more
    }
}
```

A region declares one topology role: `vertex`, `edge`, `face`, `solid`, or `cell`. Cardinality is
stated separately with `expect exactly_one` or `expect one_or_more`; plural topology spellings are
not accepted. Selectors use CAD-authored names (`select name`), presentation layers (`select tag`), canonical persistent ids (`select persistent_id "persistent-entity:sha256:…"`), feature lineage (`select feature.<f>`, `select boundary_introduced_by feature.<f>`), the result of a construction (`select result`), regions returned by a generating function (`select output.region.<r>`), and bounded geometric predicates [not yet in the current release](pending:geometry.select.predicate). Ambiguity is an error, not a request to choose the first match. The full selector table is in the [declaration reference](./reference/declarations.md#geometry).

STEP is the preferred interchange format for exact CAD and product metadata. The compiler records
which AP242 capabilities were actually found; a STEP schema label alone does not imply that
parameters, materials, persistent ids, or writable regeneration are available.

## Constructed and derived geometry

The construction syntax defines an immutable feature graph:

```fea fragment
geometry stock {
    input width : length = 120 mm

    construct {
        feature body = box(width = input.width, depth = 80 mm, height = 12 mm)
        feature hole = cylinder(
            radius = 5 mm,
            height = 12 mm,
            axis = global([0, 0, 1]),
            origin = [60, 40, 0] mm,
        )
        feature finished = subtract(feature.body, feature.hole)
        result = feature.finished
    }

    region body : solid {
        select result
        expect exactly_one
    }
}
```

A derived geometry regenerates another geometry with changed CAD parameters, and records its lineage. It does not change its source:

```fea fragment
geometry thin_wall from geometry.housing {
    derive {
        set source.parameter.wall = 2 mm
        result = regenerate(source)
    }
}
```

The parameter is declared on the source geometry with `cad_parameter wall : length { select persistent_id "…"; require writable }`. Parameter changes invalidate only the affected geometry and its
downstream mesh, projection, operator, and result identities.

## Mesh declarations

```fea fragment
mesh bracket on geometry.bracket {
    tetrahedra {
        order = quadratic
        target_size = 5 mm
        maximum_deviation = 0.05 mm
    }
}
```

`tetrahedra` meshes solids (Tet4 or Tet10), `triangles` meshes sheet faces as shells with their edges as segments, and `lines` meshes edges as beams. A mesh declaration contains discretization controls, not materials; the controls and their defaults are in the [declaration reference](./reference/declarations.md#mesh). The study must bind the same
geometry for which the mesh was declared. Meshing resolves region provenance onto solver entities
and publishes quality and construction evidence.

## Materials and capability blocks

```fea fragment
material aluminum {
    name = "Aluminum 6061-T6"
    mass_density = 2700 kg/m^3
    elastic {
        youngs_modulus = 68.9 GPa
        poisson_ratio = 0.33
    }
}
```

Material blocks are capability aggregates. A consuming analysis determines which capabilities are
required. Static elasticity needs compatible stiffness; structural dynamics also needs mass.
Thermal, electrical, magnetic, acoustic, and fluid properties remain typed capabilities on the
material rather than entries in an unstructured property map.

## Assign physical ownership

```fea fragment
model bracket {
    assign material.aluminum to geometry.bracket.region.body

    constraint fixed_support {
        structural.fixed { target = geometry.bracket.region.mounting_faces }
    }
}
```

Bulk materials belong to solid or cell regions. Beam sections belong to edge regions; shell
sections and layups belong to face regions. Orientations have their own assignment. A boundary face
does not own the bulk material of the solid behind it.

Before assembly, RunMat compiles these declarations into one canonical property map bound to the
exact geometry revision, topology digest, and solver elements. Missing coverage, overlapping
owners, an incompatible material capability, a stale topology reference, or a target with the
wrong role fails before a solver chooses coefficients.

### Shell section and laminate ownership

A homogeneous shell face receives three independent assignments: material, section, and
orientation. A composite face receives a layup and orientation; each ply names its own material,
so a second face-level material assignment would be competing constitutive ownership.

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

    assign layup.skin to geometry.bracket.region.skin
    assign orientation.fibers to geometry.bracket.region.skin
}
```

Ply order, ply angle, and reference surface are typed declarations, not signed numeric
conventions. See [models](../models.md#composite-layups) for the complete surface choices, laminate
equations, mass and damping treatment, and recovery behavior.

## Loads, constraints, and excitations

Known static loads may use the direct form:

```fea fragment
load service_force {
    force on geometry.bracket.region.mounting_faces = [0, -1, 0] kN
}
```

Time- or frequency-dependent input separates the physical signal from its application. Both declarations live in a model:

```fea fragment
signal pulse : force over time {
    samples = [
        { at = 0 s, value = 0 N },
        { at = 10 ms, value = 1 kN },
        { at = 20 ms, value = 0 N },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation impact {
    force {
        target = geometry.bracket.region.mounting_faces
        signal = signal.pulse
        direction = global([0, -1, 0])
        kinematics = fixed_frame
    }
}
```

The signal owns quantity, independent domain, samples, and interpolation. The excitation owns the
target, direction, frame, distribution, and kinematics. This split lets one signal drive several loads, and lets RunMat Desktop show each target with the function or magnitude attached to it. Standard waveforms (`sine`, `step`, `ramp`, `half_sine_pulse`, `rectangular_pulse`, `triangular_pulse`), tables and `.m`-generated signals are described in [loads](../loads/index.md) and [RunMat function integration](./runmat-function-integration.md).
