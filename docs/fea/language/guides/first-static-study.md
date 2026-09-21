# Build and run a static structural study

This guide uses the checked example at
[`docs/fea/examples/structural-static-experimental.fea`](../../examples/structural-static-experimental.fea).
Its geometry is deliberately small so the complete import, mesh, model, solve, and result path can
run in a release check.

## 1. Inspect the geometry

Before writing selectors, inspect the CAD file:

```sh
runmat run inspect_geometry.m
```

In RunMat code, `geometry.inspect(path)` reports units, topology roles, persistent ids, imported
names and tags, assemblies, and AP242 capabilities. Choose selectors from that inventory. Do not
guess a face ordinal from a viewport.

## 2. Declare geometry and regions

The example declares one solid material domain and two boundary-face regions. Each region states
its topology role and expected cardinality. Run `runmat fea check` after this step; a stale or
ambiguous selector should fail before meshing.

## 3. Declare the mesh

The mesh targets the same geometry by reference. `target_size` controls nominal volume resolution;
`maximum_deviation` bounds geometric approximation. Start with a coarse mesh for model setup, then
perform a governed refinement study before using results quantitatively.

## 4. Define and assign material behavior

The steel declaration supplies elasticity and mass density. The assignment targets the solid body,
not one of its faces. The fixed constraint and force target face regions. This distinction is what
allows the compiler to prove complete constitutive coverage and correct boundary dimensionality.

## 5. Define the study

The study binds geometry, mesh, and model, then names one structural static analysis. The example
sets `formulation = experimental` because the current runnable structural formulation has not been
promoted to production. Remove that line only when the status page and admission evidence identify
the selected formulation as production-qualified.

## 6. Check, plan, and run

```sh
runmat fea check docs/fea/examples/structural-static-experimental.fea
runmat fea plan docs/fea/examples/structural-static-experimental.fea
runmat fea run docs/fea/examples/structural-static-experimental.fea
```

Read check diagnostics before planning. In the plan, verify the selected root, geometry, mesh,
model, formulation requirement, and logical phases. After execution, inspect mesh quality,
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
but their current public entrypoints differ. `runmat fea run file.fea` is the file workflow;
`fea.run(study)` accepts a typed `fea.Study` object, not a path.
