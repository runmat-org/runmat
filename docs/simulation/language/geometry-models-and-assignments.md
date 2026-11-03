# Geometry, models and assignments

This page is the syntax reference for geometry, models and assignments. To set up a study, see [Geometry](../geometry) and [Models](../models).

Geometry defines topology. A model assigns physical meaning to that topology. Keeping those roles
separate allows one admitted CAD revision and mesh to support several material or loading studies.

## Imported geometry and regions

Keep the CAD declaration and readable region names in the study. Geometry selection tools maintain their exact persistent bindings in CAD-asset metadata.

```fea
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
not accepted. Selectors may use exact inspected ids, persistent ids, imported names or tags,
feature lineage, and governed geometric predicates. Ambiguity is an error, not a request to choose
the first match.

STEP is the preferred interchange format for exact CAD and product metadata. The compiler records
which AP242 capabilities were actually found; a STEP schema label alone does not imply that
parameters, materials, persistent ids, or writable regeneration are available.

## Constructed and derived geometry

The construction syntax defines an immutable feature graph:

```fea
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

`geometry derived from geometry.source` creates another geometry revision with explicit lineage.
It does not mutate its source. Parameter changes invalidate only the affected geometry and its
downstream mesh, projection, operator, and result identities.

## Mesh declarations

```fea
mesh bracket on geometry.bracket {
    tetrahedra {
        order = quadratic
        target_size = 5 mm
        maximum_deviation = 0.05 mm
    }
}
```

A mesh declaration contains discretization controls, not materials. The study must bind the same
geometry for which the mesh was declared. Meshing resolves region provenance onto solver entities
and publishes quality and construction evidence.

## Materials and capability blocks

```fea
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

```fea
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

```fea
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
            origin = [0, 0, 0] m
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
conventions. See [Models](../models#composite-layups) for the complete surface choices, laminate
equations, mass and damping treatment, and recovery behavior.

## Loads, constraints, and excitations

Known static loads may use the direct form:

```fea
load service_force {
    force on geometry.bracket.region.mounting_faces = [0, -1, 0] kN
}
```

Time- or frequency-dependent input separates the physical signal from its application:

```fea
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
target, direction, frame, distribution, and kinematics. This split is also what lets the Desktop
preview show the exact target and function or magnitude attached to it.
