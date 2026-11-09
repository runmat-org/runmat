# Trace impedance

A trace-impedance study computes the transmission-line parameters of a printed circuit board
trace from its cross-section: the characteristic impedance $Z_0$, the effective permittivity
$\varepsilon_\text{eff}$, the propagation delay per length, and the per-length capacitance and
inductance matrices. For a differential pair it also gives the differential and common-mode
impedances, the odd- and even-mode impedances and the near- and far-end coupling
coefficients. The cross-section comes from the board stackup: layer thicknesses and Dk, copper
thickness, trace width and etch angle, solder mask and reference planes.

The study is an `electromagnetic line_parameters` analysis
[not yet in the current release](pending:electromagnetic.line_parameters) on a `cross_section`
geometry [not yet in the current release](pending:electromagnetic.cross_section_geometry).

| Output | Symbol | Unit | Status |
| --- | --- | --- | --- |
| `impedance` | $Z_0$ per signal conductor | Ω | [not yet in the current release](pending:electromagnetic.line_parameters.impedance) |
| `effective_permittivity` | $\varepsilon_\text{eff}$ per signal conductor | 1 | [not yet in the current release](pending:electromagnetic.line_parameters.effective_permittivity) |
| `delay` | $t_d$ per length | s/m | [not yet in the current release](pending:electromagnetic.line_parameters.delay) |
| `capacitance`, `vacuum_capacitance` | $\mathbf C$, $\mathbf C_0$ | F/m | [not yet in the current release](pending:electromagnetic.line_parameters.capacitance) |
| `inductance` | $\mathbf L$ | H/m | [not yet in the current release](pending:electromagnetic.line_parameters.inductance) |
| `differential_impedance`, `common_mode_impedance`, `odd_mode_impedance`, `even_mode_impedance` | $Z_\text{diff}$, $Z_\text{comm}$, $Z_\text{odd}$, $Z_\text{even}$ | Ω | [not yet in the current release](pending:electromagnetic.line_parameters.differential_pair) |
| `near_end_coupling`, `far_end_coupling` | $K_b$, $K_f$ | 1 | [not yet in the current release](pending:electromagnetic.line_parameters.coupling) |

This version is lossless. Conductor loss from skin depth and dielectric loss from the loss
tangent (resistance and conductance per length, and attenuation) are a separate output
[not yet in the current release](pending:electromagnetic.line_parameters.loss).

## A 50 Ω microstrip and a 100 Ω pair

The study below sizes a single-ended microstrip and an edge-coupled differential pair on the
top layer of a four-layer board. The top copper is 35 µm (1 oz) under 20 µm of solder mask,
on 0.2 mm of prepreg (Dk 4.1) over the ground plane on `In1.Cu`. The etch leaves 60° sidewalls.

```fea
geometry single_ended {
    cross_section {
        layer "F.Mask" : solder_mask { thickness = 20 um; permittivity = 3.8 }
        layer "F.Cu" : signal { thickness = 35 um }
        layer "dielectric 1" : dielectric { thickness = 0.2 mm; permittivity = 4.1 }
        layer "In1.Cu" : plane { thickness = 35 um }
        trace line : signal { layer = "F.Cu"; width = 0.36 mm; etch_angle = 60 deg }
    }
}

geometry pair {
    cross_section {
        layer "F.Mask" : solder_mask { thickness = 20 um; permittivity = 3.8 }
        layer "F.Cu" : signal { thickness = 35 um }
        layer "dielectric 1" : dielectric { thickness = 0.2 mm; permittivity = 4.1 }
        layer "In1.Cu" : plane { thickness = 35 um }
        trace p : signal { layer = "F.Cu"; center = -0.2 mm; width = 0.23 mm; etch_angle = 60 deg }
        trace n : signal { layer = "F.Cu"; center = 0.2 mm; width = 0.23 mm; etch_angle = 60 deg }
    }
}

mesh single_ended on geometry.single_ended { triangles {} }
mesh pair on geometry.pair { triangles {} }

study microstrip {
    geometry = geometry.single_ended
    mesh = mesh.single_ended
    analysis line { electromagnetic line_parameters {} }
    output z0 = analysis.line.impedance.line
    output eps_eff = analysis.line.effective_permittivity.line
    output delay = analysis.line.delay.line
}

study differential_pair {
    geometry = geometry.pair
    mesh = mesh.pair
    analysis lines { electromagnetic line_parameters {} }
    output z_diff = analysis.lines.differential_impedance
    output z_common = analysis.lines.common_mode_impedance
    output next = analysis.lines.near_end_coupling
    output fext = analysis.lines.far_end_coupling
    output c = analysis.lines.capacitance
    output l = analysis.lines.inductance
}
```

`runmat run trace-impedance.fea --study microstrip` reports $Z_0$ close to 50 Ω and
`--study differential_pair` reports $Z_\text{diff}$ close to 100 Ω. To hit a target, change
the width (and, for the pair, the gap) and run again: each solve takes about a second.

## Describing the cross-section

A `cross_section` block lists the stackup from the top of the board down, then the traces.

| Layer role | Fields | Meaning |
| --- | --- | --- |
| `dielectric` | `thickness`, `permittivity` | core or prepreg with its Dk at the frequency of interest |
| `solder_mask` | `thickness`, `permittivity` | conformal mask over the outer copper next to it: `thickness` deep on the laminate between traces and over each trace's top and sidewalls |
| `signal` | `thickness`, optional `fill_permittivity` | the routed copper layer; between traces it is filled with mask or air on an outer layer and with the dielectric directly above on an inner layer, unless `fill_permittivity` says otherwise |
| `plane` | `thickness` | a full reference plane at 0 V |

| Trace field | Meaning |
| --- | --- |
| role `signal` or `ground` | a line whose parameters are reported, or coplanar ground copper held at 0 V |
| `layer` | the copper layer it is etched on; every trace of one section is on the same layer |
| `center` | lateral position of the trace centre (default 0) |
| `width` | width of the base face, the face bonded to the laminate |
| `etch_angle` | sidewall angle from the base face (default 90°); the far face is narrower by $2t/\tan\theta$ |
| `base = top` or `bottom` | which face is the base; by default the laminate side (bottom on the top layer, top on the bottom layer, bottom on an inner layer) |

The domain lies between the reference planes nearest the signal layer. Layers beyond those
planes do not affect the result, so a section may list the stackup only down to (or up to)
the plane. An outer layer without a plane on its open side is surrounded by air: the section
extends 60 dielectric heights to the sides and above, with insulating walls, which changes $Z_0$
by less than 0.05 %. `enclosure { half_width = …; air = … }` sets other distances. A line
needs a return conductor: a reference plane or `ground` traces.

### Taking the stackup from the board

`stackup = "./board.circuit.layout"` reads the layer list (names, thicknesses, Dk and copper
weights) from the board's layout file, so the section and the board cannot disagree. The
traces then name the layout's copper layers, for example `layer = "F.Cu"`. A layout with
several stackups (rigid-flex) names one with `stackup_name = rigid`.

### Broadside-coupled lines

Conductors on two copper layers of one section, such as a broadside-coupled pair
[not yet in the current release](pending:electromagnetic.line_parameters.broadside), solve as
one multiconductor line.

## How the parameters are computed

The line is quasi-TEM: its cross-section is small compared with the wavelength, so the
transverse field at each instant satisfies the 2D electrostatic equation on the section,

$$
\nabla\cdot\left(\varepsilon_0\varepsilon_r \nabla\varphi\right) = 0,
$$

with $\varphi$ prescribed on the conductors and zero normal flux on the enclosure walls.

**Capacitance.** Holding signal conductor $j$ at 1 V and every other conductor at 0 V gives
the potential $\varphi_j$. The Maxwell capacitance matrix per length is the energy form

$$
C_{ij} = \int_\Omega \varepsilon_0\varepsilon_r\, \nabla\varphi_i\cdot\nabla\varphi_j \, dA ,
$$

symmetric, with positive diagonal and negative off-diagonal terms. A second solve with every
$\varepsilon_r = 1$ gives $\mathbf C_0$.

**Inductance.** Copper and laminate are non-magnetic, so the inductance matrix follows from
the vacuum capacitance: $\mathbf L = \mu_0\varepsilon_0\,\mathbf C_0^{-1}$.

**Single line.**
$Z_0 = \sqrt{L/C}$, $\varepsilon_\text{eff} = C/C_0$ and $t_d = \sqrt{LC} = \sqrt{\varepsilon_\text{eff}}/c$ per metre.

**Several lines.** The characteristic impedance matrix of the lossless multiconductor line is
$\mathbf Z_c = \mathbf C^{-1/2}\left(\mathbf C^{1/2}\mathbf L\,\mathbf C^{1/2}\right)^{1/2}\mathbf C^{-1/2}$.
`impedance` is its diagonal, the impedance of each line with the others held at 0 V. For a
pair driven differentially, $\mathbf v = [1, -1]$, and in common, $\mathbf v = [1, 1]$:

$$
Z_\text{diff} = \mathbf v_\text{d}^T \mathbf Z_c \mathbf v_\text{d},\qquad
Z_\text{comm} = \tfrac14\,\mathbf v_\text{c}^T \mathbf Z_c \mathbf v_\text{c},\qquad
Z_\text{odd} = Z_\text{diff}/2,\qquad Z_\text{even} = 2Z_\text{comm},
$$

and the modal effective permittivity is $\mathbf v^T\mathbf C\mathbf v / \mathbf v^T\mathbf C_0\mathbf v$.
On a microstrip the odd mode travels faster than the even mode, because more of its field is in air.

**Coupling.** With the mutual terms $C_m = -C_{12}$ and $L_m = L_{12}$ against
$C_s = \sqrt{C_{11}C_{22}}$ and $L_s = \sqrt{L_{11}L_{22}}$:

$$
K_b = \tfrac14\left(\frac{L_m}{L_s} + \frac{C_m}{C_s}\right),\qquad
K_f = \tfrac12\left(\frac{C_m}{C_s} - \frac{L_m}{L_s}\right).
$$

A step of $\Delta V$ with rise time $t_r$ on one line of a coupled length $\ell$ gives a
near-end voltage $K_b\,\Delta V$ on the quiet line (once $2\ell/v > t_r$) and a far-end
pulse of about $K_f\,(\ell/v)\,\Delta V/t_r$. In a homogeneous dielectric (stripline)
$K_f = 0$; on microstrip $K_f$ is negative because inductive coupling dominates.

**Discretization.** The mesher builds a graded triangle mesh whose lines follow every layer
interface, the mask surfaces and the trace edges, including sloped sidewalls. Elements are
smallest at trace edges and corners, where the field is singular, and grow geometrically
away from them. On linear triangles the element matrix is

$$
\mathbf K_e = \frac{\varepsilon_0\varepsilon_r}{4A}\left(\mathbf b\mathbf b^T + \mathbf c\mathbf c^T\right),
$$

with $A$ the area and $b_i$, $c_i$ the corner coordinate differences. Conductor nodes are
prescribed; one sparse Cholesky factor of the free block serves every signal conductor as a
right-hand side, on the admitted CPU threads.

| Mesh field | Default | Meaning |
| --- | --- | --- |
| `edge_size` | smallest feature / 1000 | element size at trace edges and corners |
| `growth` | 1.1 | size ratio between neighbouring elements |
| `max_size` | twice the dielectric height | largest element, reached in the far field |

The smallest feature is the least of the copper thickness, trace widths, gaps, dielectric
and mask thicknesses. A section that would not fit the run's memory budget is rejected with
its size and the budget.

## Accuracy

Against Hammerstad and Jensen's microstrip forms the default mesh gives $Z_0$ and
$\varepsilon_\text{eff}$ within 0.5 %. Against Cohn's exact stripline and coupled-stripline
solutions it gives each impedance within 0.3 %. Refining the edge size converges $Z_0$ to better
than 0.1 %. The [electromagnetic validation page](../validation/electromagnetic.md) lists
these checks with their problems, tolerances and results, including agreement with the
IPC-2141A forms inside their stated ranges.

The parameters are quasi-static. They hold while the cross-section is small compared with
the wavelength in the dielectric, which covers board traces well into the gigahertz range;
dispersion of $\varepsilon_\text{eff}$ at higher frequencies and loss are not included.
