# Build and run a static structural study

This guide uses the checked example at
[`docs/simulation/examples/structural-static.fea`](../../examples/structural-static.fea).
It uses a two-part STEP assembly so the complete exact import, mesh, multi-solid material coverage,
model, solve, and result path can run in a release check.

## 1. Inspect the geometry

Before writing selectors, inspect the CAD file:

```sh
runmat run inspect_geometry.m
```

In RunMat code, `geometry.inspect(path)` reports units, topology roles, persistent ids, imported
names and tags, assemblies, and AP242 capabilities. Choose selectors from that inventory. Do not
guess a face ordinal from a viewport.

## 2. Declare geometry and regions

The example declares both solid occurrences, a support on each disconnected part, and a load face
on the left part. Each region states its topology role and expected cardinality. Run `runmat fea
check` after this step; a stale or ambiguous selector should fail before meshing.

The included STEP file carries the authored body and end-face names the study selects; `select name` resolves only names authored in the CAD file. The study stays in one readable source file; exporting the project preserves its CAD names.

```fea
geometry bracket {
    file = "./geometry/two-box-assembly.step"
    units = m

    region left_body : solid { select name "left_body"; expect exactly_one }
    region right_body : solid { select name "right_body"; expect exactly_one }
    region left_support : face { select name "left_support"; expect exactly_one }
    region left_load_face : face { select name "left_load_face"; expect exactly_one }
    region right_support : face { select name "right_support"; expect exactly_one }
}
```

## 3. Declare the mesh

The mesh targets the same geometry by reference. `target_size` controls nominal volume resolution;
`maximum_deviation` bounds geometric approximation. Start with a coarse mesh for model setup, then
perform a governed refinement study before using results quantitatively.

```fea
mesh bracket on geometry.bracket {
    tetrahedra {
        order = linear
        target_size = 1.5 m
        maximum_deviation = 0.05 m
    }
}
```

## 4. Define and assign material behavior

The steel declaration supplies elasticity and mass density. Separate assignments cover both solid
occurrences, not their faces. The fixed constraints and force target face regions. This distinction
lets the compiler prove complete constitutive coverage and correct boundary dimensionality; the
second support also prevents the disconnected right occurrence from retaining rigid-body modes.

```fea
material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}

model loaded_bracket {
    assign material.steel to geometry.bracket.region.left_body
    assign material.steel to geometry.bracket.region.right_body
    constraint left_support {
        structural.fixed { target = geometry.bracket.region.left_support }
    }
    constraint right_support {
        structural.fixed { target = geometry.bracket.region.right_support }
    }
    load service_traction {
        traction on geometry.bracket.region.left_load_face = [166.6666666667, 0, 0] Pa
    }
    load_case service { include load.service_traction }
}
```

The traction tuple is expressed in global XYZ coordinates. The selected end face has an area of
$6\,\mathrm{m^2}$, so `[166.6666666667, 0, 0] Pa` retains the original $1\,\mathrm{kN}$ resultant
while boundary quadrature distributes it consistently over the surface mesh. Targeting a face does
not implicitly rotate an arbitrary traction vector onto that face normal.

## 5. Define the study

The study binds geometry, mesh, and model, then names one structural static analysis and the
result fields to publish.

```fea
study structural_static {
    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.loaded_bracket
    analysis static { structural static {} }
    output displacement = analysis.static.field.node.displacement
    output von_mises = analysis.static.field.element.von_mises_stress
}
```

## 6. Check, plan, and run

```sh
runmat check docs/simulation/examples/structural-static.fea
runmat plan docs/simulation/examples/structural-static.fea
runmat run docs/simulation/examples/structural-static.fea
```

Read check diagnostics before planning. In the plan, verify the selected root, geometry, mesh,
model, and logical phases. After execution, inspect mesh quality,
property coverage, constraint rank, convergence, balance, and formulation evidence before reading
the displacement or stress field.

## 7. Plot an output

From RunMat code, select a typed result field and pass it to ordinary `plot`:

```matlab
run = fea.run(study);
results = fea.results(run);
stress = fea.field(results, fea.StructuralField.vonMises());
plot(stress, "Edges", true);
```

The `.m` object-construction path and `.fea` file path share canonical study and result contracts,
but their current public entrypoints differ. `runmat run file.fea` is the file workflow;
`fea.run(study)` accepts a typed `fea.Study` object, not a path.
