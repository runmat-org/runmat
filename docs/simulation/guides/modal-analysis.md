---
description: A worked example of modal analysis. The natural frequencies of a clamped steel tuning fork are estimated from beam vibration theory and computed by the finite-element method in RunMat, from the eigenproblem Kφ = ω²Mφ to mode shapes and a check against an excitation band.
keywords: [modal analysis, natural frequency, mode shape, eigenvalue problem, free vibration, cantilever beam vibration, consistent mass matrix, block Lanczos, tuning fork, Tet10, excitation band, frequency margin]
doc_type: guide
family: [structural]
examples: [tuning-fork-modal]
---
A modal analysis finds the natural frequencies of a supported structure and the shape it vibrates in at each one. It is the first step in keeping resonances away from the frequencies a machine, a road or a motor will drive, and it supplies the modes that forced-response studies build on. This page works one example in full: the vibration theory of a cantilever, the finite-element eigenproblem that generalizes it, and the solution in RunMat.

## Example 1: Natural frequencies of a clamped tuning fork

A steel tuning fork is clamped at the bottom of its stem and mounted on equipment that vibrates anywhere from 0 to 250 Hz. To avoid resonance, its lowest natural frequency must lie at least 20 % above the top of that band.

**Given**

<!-- fea:example:tuning-fork-modal:setup -->

| Quantity | Value |
| --- | --- |
| Overall height | 100 mm; the fork is a 6 mm thick plate throughout |
| Stem | 8 mm wide, 25 mm long |
| Bridge | 36 mm wide, 10 mm tall |
| Tines | two, each 8 mm wide and 65 mm long above the bridge, 20 mm apart |
| Material | steel, E = 200 GPa, ν = 0.3, ρ = 7,850 kg/m³ |
| Support | the bottom face of the stem is fixed in all directions |
| Excitation band | 0 to 250 Hz |
| Required lowest frequency | 250 Hz × 1.2 = 300 Hz |

**Find**

1. The first two natural frequencies and the motion of each mode.
2. Whether the lowest natural frequency clears 300 Hz.

## Free vibration of a cantilever

### Single degree of freedom

A mass $m$ on a spring of stiffness $k$, displaced and released, oscillates at its natural frequency

$$
m\ddot x + kx = 0 \quad\Rightarrow\quad \omega = \sqrt{k/m}, \qquad f = \frac{\omega}{2\pi}.
$$

Stiffer means higher, heavier means lower. A continuous structure behaves like many springs and masses coupled together, and it has one natural frequency for each independent pattern of motion.

### Bending vibration of a beam

For a slender Euler–Bernoulli beam with bending stiffness $EI$ and mass per length $\rho A$, the transverse deflection $v(x,t)$ satisfies

$$
EI\,\frac{\partial^4 v}{\partial x^4} + \rho A\,\frac{\partial^2 v}{\partial t^2} = 0 .
$$

A free vibration has the form $v = \phi(x)\sin\omega t$, which separates into the eigenvalue problem $EI\,\phi'''' = \omega^2 \rho A\,\phi$. A cantilever has $\phi = \phi' = 0$ at the clamp and no moment or shear at the free end ($\phi'' = \phi''' = 0$). Those four conditions admit a nonzero $\phi$ only when

$$
\cos\beta L \cosh\beta L = -1, \qquad \beta^4 = \frac{\omega^2 \rho A}{EI},
$$

whose first root is $\beta_1 L = 1.875$. The first natural frequency is therefore

$$
f_1 = \frac{1.875^2}{2\pi}\sqrt{\frac{EI}{\rho A L^4}} .
$$

For a rectangular section bending across its thickness $h$, $I/A = h^2/12$, so $f_1$ grows in proportion to $h/L^2$ and to $\sqrt{E/\rho}$. These are the design levers: a shorter or thicker member is stiffer and vibrates faster.

Treating one tine as a cantilever clamped at the bridge, with L = 65 mm, gives about 1,160 Hz for bending across the 6 mm thickness and 1,540 Hz for bending across the 8 mm width. A tine alone would clear the band easily. The fork is more than its tines, though: the tines sit on a bridge and a slender stem, and the beam formula cannot say how those parts combine.

### Finite-element formulation

**Weak form.** For the elastic solid with displacement $\mathbf u$, density $\rho$ and no applied load, the principle of virtual work with inertia reads

$$
\int_\Omega \rho\, \mathbf w \cdot \ddot{\mathbf u}\, d\Omega + \int_\Omega \boldsymbol\varepsilon(\mathbf w) : \mathsf C : \boldsymbol\varepsilon(\mathbf u)\, d\Omega = 0
$$

for every virtual displacement $\mathbf w$ that vanishes on the clamped face.

**Discretization.** The fork is meshed with ten-node tetrahedra (Tet10), and the displacement in each element is interpolated from its nodes with quadratic shape functions, $\mathbf u^h = \sum_a N_a \mathbf u_a$, as in the [linear static example](./linear-static.md#finite-element-formulation). The two integrals produce the element stiffness and mass matrices:

$$
\mathbf K_e = \int_{\Omega_e} \mathbf B_e^{\mathsf T}\mathbf D\,\mathbf B_e\, d\Omega, \qquad
\mathbf M_e = \int_{\Omega_e} \rho\, \mathbf N^{\mathsf T}\mathbf N\, d\Omega .
$$

$\mathbf M_e$ is the consistent mass matrix: it uses the same shape functions as the stiffness, so the kinetic energy is computed with the same displacement field.

**Eigenproblem.** Assembling the elements, removing the clamped degrees of freedom as in a static analysis, and substituting $\mathbf u = \boldsymbol\phi \sin\omega t$ gives the generalized eigenvalue problem

$$
\mathbf K\,\boldsymbol\phi_i = \omega_i^2\, \mathbf M\,\boldsymbol\phi_i, \qquad f_i = \frac{\omega_i}{2\pi} .
$$

This is the spring–mass equation with matrices in place of numbers. Both matrices are symmetric, and $\mathbf M$ is positive definite, so the $\omega_i^2$ are real and the mode shapes are orthogonal with respect to the mass, $\boldsymbol\phi_i^{\mathsf T}\mathbf M\boldsymbol\phi_j = 0$ for $i \ne j$. A part with no supports has six zero frequencies, its rigid-body motions, before the first elastic mode.

**Solution method.** A model has as many modes as unknowns, but design needs only the lowest few. RunMat finds them with a shift-invert block Lanczos method: it factors $\mathbf K - \sigma\mathbf M$ once for a shift $\sigma$ and iterates on a small block of vectors, which converges first to the frequencies nearest the shift. The [structural physics reference](../physics/structural.md#how-do-i-run-a-modal-analysis-and-check-natural-frequencies) lists what each computed mode records.

## Solution

Each block below is an excerpt of `tuning-fork-modal.fea`.

### Geometry

The fork is imported from a STEP file, and the clamped face is selected by its CAD name.

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

Frequencies depend on the ratio of stiffness to mass, so the density enters $\mathbf M$ and matters as much as Young's modulus.

```fea source=tuning-fork-modal.fea
material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}
```

### Support, and no load

Free vibration has no applied load, so the model has only the support. The support is part of the answer: it removes degrees of freedom from $\mathbf K$ and $\mathbf M$, and a stiffer support gives higher frequencies. `structural.fixed` clamps the stem's bottom face rigidly.

```fea source=tuning-fork-modal.fea
model supported_fork {
    assign material.steel to geometry.fork.region.body
    constraint fixed_end {
        structural.fixed { target = geometry.fork.region.fixed_end }
    }
}
```

### Mesh

Mode shapes are bending shapes, so the mesh uses quadratic tetrahedra; linear tetrahedra are too stiff in bending and would push the frequencies up. The 30 mm target size is larger than the fork's features, so the 6 mm thickness and the 0.1 mm surface deviation set the actual element size. The mesh has 11,144 nodes.

```fea source=tuning-fork-modal.fea
mesh fork on geometry.fork {
    tetrahedra {
        order = quadratic
        target_size = 30 mm
        maximum_deviation = 0.1 mm
    }
}
```

### Study

The analysis requests the two lowest modes and retains the modal basis, so later response steps can reuse it. The output publishes the mode shapes.

```fea source=tuning-fork-modal.fea
study fork_modes {
    geometry = geometry.fork
    mesh = mesh.fork
    model = model.supported_fork
    analysis modes { structural modal { modes = 2 } retain basis }
    output shapes = analysis.modes.field.node.displacement
}
```

### Run

The complete project opens in RunMat Desktop with the STEP geometry, the study and the saved modes:

<!-- fea:example:tuning-fork-modal:edit -->

From the command line:

```sh
runmat check tuning-fork-modal.fea
runmat run tuning-fork-modal.fea
```

`run` meshes the fork, assembles $\mathbf K$ and $\mathbf M$, extracts the requested modes and publishes the frequencies and shapes. In `.m` code, `fea.results(runId, "ModeIndices", [1 2])` reads them; see [get started](../get-started.md#read-results-in-m-code).

### Mode 1

<!-- fea:example:tuning-fork-modal:mode-1 -->

The first natural frequency is **[364.85 Hz](receipt:tuning-fork-modal#mode-1-frequency)**. Color shows the relative amplitude of the mode shape, scaled so that its largest displacement is 1; the deformed shape is exaggerated. A mode shape has no physical amplitude, and its sign is arbitrary: the same mode can appear with every displacement reversed. How far the fork actually moves depends on the forcing and the damping, which a [harmonic or transient response](../language/reference/procedures.md#structural-procedures) adds.

### Mode 2

<!-- fea:example:tuning-fork-modal:mode-2 -->

The second natural frequency is **[475.17 Hz](receipt:tuning-fork-modal#mode-2-frequency)**. Comparing the stem, the bridge and the tine tips with mode 1 shows which direction each mode moves in and which member flexes most.

### Answers

1. The first two natural frequencies are [364.85 Hz](receipt:tuning-fork-modal#mode-1-frequency) and [475.17 Hz](receipt:tuning-fork-modal#mode-2-frequency). Both lie far below the 1.2–1.5 kHz estimates for a tine alone, so the tines are not the most flexible part: the slender 8 mm × 6 mm stem bends in these modes too. The stem's second moment of area is 256 mm⁴ for bending across its width and 144 mm⁴ across its thickness, a ratio whose square root, 1.33, is close to the ratio of the two frequencies, 1.30. That points to the whole fork swaying on its stem, across the plate's thickness in mode 1 and in its own plane in mode 2; the shape views show the direction of each.
2. The lowest frequency, 364.85 Hz, is above the required 300 Hz, and 46 % above the top of the band. The fork meets the criterion.

## Discussion

**Design meaning.** The margin is comfortable. If the band reached 330 Hz, the margin would fall to 11 %, and the design response would be to stiffen what moves in mode 1: a wider or shorter stem raises both frequencies far more than any change to the tines, because the stem is the flexible member.

**Model limits.** The fixed face is a perfectly rigid clamp. A real fixture is somewhat flexible and lowers every frequency, so the computed values are an upper estimate for the mounted fork. When the margin is small, model the fixture or measure its stiffness. The analysis is also linear and undamped: it gives frequencies and shapes, not amplitudes.

**Refinement.** A displacement-based finite-element model is slightly too stiff, so its frequencies approach the exact values from above and fall a little as the mesh is refined. Before relying on a margin, rerun with a smaller `target_size` and confirm that the first frequency changes by much less than the margin. A 46 % margin tolerates a lot; an 11 % margin deserves a refinement run. The [mesh refinement guide](./mesh-refinement.md) works through a refinement sequence.

## Try a change

**Problem 1.1.** Change the material to aluminum (E = 68.9 GPa, ρ = 2,700 kg/m³). Predict the effect before running.

*Answer.* When only the material changes, every frequency scales with $\sqrt{E/\rho}$. For aluminum that ratio is within 1 % of steel's, so the frequencies barely move: aluminum is lighter and less stiff in almost the same proportion.

**Problem 1.2.** Request ten modes (`modes = 10`) and look for the mode in which the tines move against each other in the plane of the fork, the motion that sounds a tuning fork's note. Compare its frequency with the 1,540 Hz tine estimate, and explain the difference from how the bridge holds each tine. If it is not among the first ten, request more.

**Problem 1.3.** Shorten the stem in the CAD and re-export it with the same face names. Which of the first two frequencies rises more, and why?

**Problem 1.4.** Publish the frequencies as an output, so the criterion can be stated as an assessment in the study ([not yet in the current release](pending:output.mode-location)):

```fea fragment
output frequencies = analysis.modes.field.mode.modal_frequency
```

A retained modal basis also feeds the modal-superposition procedures: harmonic response, transient response, response spectrum and random vibration (see [studies, analyses and state](../language/studies-analyses-and-state.md#order-steps-and-pass-state)).

## Next

- [Linear static analysis](./linear-static.md): deflection, stress and reactions under load.
- [Structural physics reference](../physics/structural.md): every structural procedure and its equations.
- [Structural validation](../validation/structural.md): how frequencies and mode bases are checked.
