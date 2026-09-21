# Geometry, Meshes, and Models

## Imported Geometry

```fea
geometry bracket {
    file = "./geometry/bracket.step"
    units = mm

    region body : solid {
        select persistent_id "cad-solid-id"
        expect exactly_one
    }
    region support : face {
        select id "face_000001"
        expect exactly_one
    }
}
```

`select id` addresses an inspected source alias within the declared topology role. `select persistent_id` addresses a canonical persistent entity identity. Other typed selectors cover imported names/tags, feature results, Boolean lineage, and bounded geometric predicates. Every region declares its topology role and cardinality expectation; a face cannot silently substitute for a solid.

## Constructed And Derived Geometry

```fea
geometry stock {
    construct {
        feature body = box(width = 120 mm, depth = 80 mm, height = 12 mm)
        result = feature.body
    }
    region body : solid { select result expect exactly_one }
}

geometry drilled from geometry.stock {
    derive {
        feature bore = cylinder(
            radius = 5 mm,
            height = 12 mm,
            axis = global([0, 0, 1]),
            origin = [60, 40, 0] mm,
        )
        result = subtract(source.result, feature.bore)
    }
}
```

Feature operations are typed and immutable. Derived geometry retains exact source lineage; changed parameters invalidate only the affected geometry and downstream mesh/solve identities.

## Solver Mesh

```fea
mesh bracket on geometry.bracket {
    tetrahedra {
        order = quadratic
        target_size = 5 mm
        maximum_deviation = 0.05 mm
        deterministic_seed = 7
    }
}
```

The mesh declaration carries physical controls, not materials. Materials belong to models and are assigned to typed geometry regions. Exact meshing later projects those assignments onto canonical solver elements while retaining CAD provenance.

## Materials And Assignments

```fea
material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.30
    }
}

model loaded_bracket {
    assign material.steel to geometry.bracket.region.body

    constraint fixed_support {
        structural.fixed { target = geometry.bracket.region.support }
    }
    load service_force {
        force on geometry.bracket.region.load_face = [0, -1, 0] kN
    }
    load_case service { include load.service_force }
}
```

Bulk constitutive ownership is assigned to solid or cell regions; shell and beam formulations use face or edge regions plus explicit section/layup/orientation properties. Boundary faces of a solid do not own the solid's bulk material. Compilation rejects missing coverage, overlapping owners, incompatible material capabilities, stale topology, and wrong-dimensional targets.

Models may accept typed region or material parameters and include other models with explicit bindings. Composition retains the defining module of every declaration and rejects cycles or conflicting ownership; it never relies on source order or implicit override.
