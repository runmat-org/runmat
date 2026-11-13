# Materials

A material declares the physical properties a study needs: density, elasticity, damping, thermal,
acoustic, electrical, magnetic and plasticity data. Defining a material and assigning it to a region
are separate steps, so one material can be reused across parts and studies.

## Material Is Separate From Assignment

```fea
material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.30
    }
}

model bracket {
    assign material.steel to geometry.part.region.body
}
```

Defining `steel` does not apply it everywhere. The assignment names the exact geometry region that owns the constitutive law. For a tetrahedral volume model that target is a solid or cell region—not one of its boundary faces. Edge and face material ownership is used for beam and shell formulations together with the required section, layup, and orientation properties.

Before assembly, RunMat projects every assignment through persistent geometry-to-mesh lineage into one immutable element property map. It rejects gaps, overlaps, wrong topology, incompatible capabilities, stale geometry/mesh revisions, and competing homogeneous/composite ownership. Solver implementations consume that compiled map and cannot choose a fallback material independently.

## Material Capabilities

Material declarations may provide the capabilities required by the selected physics:

- mass density;
- linear elasticity;
- Rayleigh damping;
- thermal reference data, conductivity, heat capacity, expansion, and temperature dependence;
- acoustic density/sound-speed damping data;
- electrical and magnetic properties;
- typed plasticity parameters.

The checker validates physical dimensions and admissible ranges. A study cannot satisfy a mechanical formulation with only thermal conductivity, or invent a missing density for dynamics.

AP242 material/property candidates can be adopted explicitly from imported geometry. The declaration identifies each property to use, its evidence environment, and conflict policy; source confidence remains provenance and never scales a coefficient.
