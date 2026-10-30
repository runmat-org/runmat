---
description: Learn modal analysis in RunMat on a steel tuning fork. Model and mesh the part, extract its natural frequencies and mode shapes, read them the way a vibration engineer does, check the lowest frequency against an excitation band, and try design changes.
keywords: [modal analysis, natural frequency, mode shape, eigenfrequency, vibration, resonance, tuning fork, Tet10, frequency margin, excitation band]
doc_type: guide
family: [structural]
examples: [tuning-fork-modal]
---
A modal analysis finds the natural frequencies of a supported structure and the shape it moves in at each one. Use it to keep resonances away from the frequencies a machine, a road or a motor will drive, to see which parts of a design are flexible, and to choose the modes a forced-response study needs. A mode shape is a pattern of motion, not an amplitude: how far the structure actually moves depends on the forcing and the damping, which a [harmonic or transient response](../language/reference/procedures.md#structural-procedures) adds.

This guide takes one part from the question to a design decision.

## Keeping a tuning fork clear of a 0–250 Hz band

A steel tuning fork is clamped at the bottom of its stem and mounted on equipment that vibrates anywhere from 0 to 250 Hz.

- Is the fork's lowest natural frequency at least **20 % above the top of the excitation band**, that is, above **300 Hz**?
- Which motion does each of the first two modes involve?

<!-- fea:example:tuning-fork-modal:setup -->

## How it's modeled: free vibration of a cantilever

A mass on a spring vibrates at $f = \frac{1}{2\pi}\sqrt{k/m}$: stiffer means higher, heavier means lower. A uniform cantilever, such as one tine, has a first bending frequency of

$$
f_1 = \frac{1.875^2}{2\pi}\sqrt{\frac{EI}{\rho A L^4}},
$$

so the frequency rises with the tine's thickness and falls with the square of its length. These scalings are the design levers: shorten or thicken a tine to raise its frequency.

A real part is many springs and masses coupled together. Finite-element analysis builds the stiffness matrix $\mathbf K$ and the mass matrix $\mathbf M$ of the meshed part and finds the shapes $\boldsymbol\phi$ that vibrate at a single frequency,

$$
\mathbf K\boldsymbol\phi = \omega^2 \mathbf M \boldsymbol\phi, \qquad f = \frac{\omega}{2\pi}.
$$

Each solution is a mode: one natural frequency and its shape. This is the spring–mass equation with matrices in place of numbers. The [structural physics reference](../physics/structural.md) covers the eigen solver and the mass matrix.

## Model the fork

### Geometry

Import the fork from CAD and name the face that is clamped.

```fea source=tuning-fork-modal.fea
geometry fork {
    file = "./geometry/tuning-fork.step"
    units = mm
    region body : solid { select name "Tuning fork"; expect exactly_one }
    region fixed_end : face { select name "Stem end"; expect exactly_one }
}
```

<!-- fea:example:tuning-fork-modal:geometry -->

### Material

Frequencies come from the ratio of stiffness to mass, so the density matters as much as Young's modulus. Check both.

```fea source=tuning-fork-modal.fea
material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}
```

### Supports, and no loads

A modal analysis asks how the supported structure vibrates freely, so it has no loads. The supports, though, are part of the answer: the stiffer the support, the higher the frequencies. Here the stem's bottom face is fully fixed, an idealized rigid clamp. A real fixture is a little flexible and lowers the frequencies somewhat, so treat the computed values as an upper estimate for the mounted part. A part with no supports at all has six zero-frequency rigid-body modes before its first elastic one.

```fea source=tuning-fork-modal.fea
model supported_fork {
    assign material.steel to geometry.fork.region.body
    constraint fixed_end {
        structural.fixed { target = geometry.fork.region.fixed_end }
    }
}
```

## Mesh

Use quadratic tetrahedra (Tet10): mode shapes are bending shapes, and linear tetrahedra are too stiff in bending, which pushes frequencies up. The 30 mm target size is larger than the fork's features, so the 6 mm tines and the 0.1 mm surface deviation set the actual element size; the mesh has 11,144 nodes.

```fea source=tuning-fork-modal.fea
mesh fork on geometry.fork {
    tetrahedra {
        order = quadratic
        target_size = 30 mm
        maximum_deviation = 0.1 mm
    }
}
```

## Run

The analysis requests the number of modes and keeps the modal basis for later response steps.

```fea source=tuning-fork-modal.fea
study fork_modes {
    geometry = geometry.fork
    mesh = mesh.fork
    model = model.supported_fork
    analysis modes { structural modal { modes = 2 } retain basis }
    output shapes = analysis.modes.field.node.displacement
}
```

The complete project opens in RunMat Desktop with the STEP geometry, the study and the saved modes:

<!-- fea:example:tuning-fork-modal:edit -->

From the command line:

```sh
runmat check tuning-fork-modal.fea
runmat run tuning-fork-modal.fea
```

`run` meshes the fork, extracts the requested modes and publishes the frequencies and mode shapes. Read them in `.m` code with `fea.results(runId, "ModeIndices", [1 2])`; see [get started](../get-started.md#read-results-in-m-code).

## Results

### Mode 1

<!-- fea:example:tuning-fork-modal:mode-1 -->

The first natural frequency is **[364.85 Hz](receipt:tuning-fork-modal#mode-1-frequency)**. Color shows the relative amplitude of the mode shape, scaled so that its largest displacement is 1; the deformed shape is exaggerated for viewing. Rotate the view and note whether the tines move in the plane of the fork or across it, and whether they move together or against each other.

### Mode 2

<!-- fea:example:tuning-fork-modal:mode-2 -->

The second natural frequency is **[475.17 Hz](receipt:tuning-fork-modal#mode-2-frequency)**. Compare the tine tips and the bridge with mode 1.

Three habits help when reading modes:

- **Don't assume the order.** The lowest modes of a clamped fork are not necessarily the familiar audible mode, in which the tines move against each other in the plane of the fork. The stem's stiffness, the tine section and the bridge decide the order.
- **Ignore the amplitude and the sign.** A mode is normalized, not loaded; its colors show where the part moves most, not how far. The same mode can appear with every displacement reversed.
- **Look for the flexible part.** The region with the largest motion in the lowest mode is the place to stiffen if the frequency must rise.

### The decision

| Criterion | Limit | Result |
| --- | --- | --- |
| Lowest natural frequency | above 300 Hz (250 Hz + 20 %) | [364.85 Hz](receipt:tuning-fork-modal#mode-1-frequency) |

The first mode is 46 % above the top of the band, well clear. If the band reached 330 Hz, the margin would fall to 11 %, and a stiffer stem or a shorter tine would be the design response.

## Refining the mesh

A frequency is mesh-dependent like any result, and a coarse mesh is too stiff, so frequencies fall slightly as the mesh is refined. Before relying on a margin, rerun with a smaller `target_size` and confirm that the first frequency changes by much less than the margin. A 46 % margin tolerates a lot; an 11 % margin deserves a refinement run and a closer look at the support stiffness. The [mesh refinement guide](./mesh-refinement.md) walks through a refinement sequence.

## Try a change

- **Use aluminum** (E = 68.9 GPa, ρ = 2,700 kg/m³). The ratio E/ρ is almost the same as steel's, so the frequencies barely change: frequency scales with $\sqrt{E/\rho}$ when only the material changes.
- **Request more modes** (`modes = 10`) to see the next bending and twisting patterns and how far above the band they lie.
- **Shorten the tines in CAD** and re-export with the same face names. The tine bending frequencies rise roughly with the inverse square of the tine length.
- **Fix more of the stem.** Fixing extra faces raises the frequencies, just as a rigid clamp does compared with a real fixture.
- **Publish the frequencies as an output** to state the criterion in the study ([not yet in the current release](pending:output.mode-location)):

```fea fragment
output frequencies = analysis.modes.field.mode.modal_frequency
```

A retained modal basis feeds the modal-superposition procedures: harmonic response, transient response, response spectrum and random vibration (see [studies, analyses and state](../language/studies-analyses-and-state.md#order-steps-and-pass-state)).

## Next

- [Linear static analysis](./linear-static.md): deflection, stress and reactions under load.
- [Structural physics reference](../physics/structural.md): the eigenproblem, mass matrix and solver.
- [Modal validation](../validation/structural.md): how frequencies and mode bases are checked.
