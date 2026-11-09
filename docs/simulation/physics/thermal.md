# Thermal

Thermal FEA predicts temperature, heat flow, and stored thermal energy under prescribed temperature,
heat flux, convection, and volumetric heat generation. RunMat separates thermal material ownership
from boundary conditions: conductivity, density, and specific heat belong to solid/cell material
regions; temperatures and fluxes target compatible regions on the same admitted geometry.

The typed `.fea` contracts and tetrahedral thermal finite-element assembly are implemented.
Compiled-study execution supports a thermal prefix feeding a same-mesh thermomechanical
preload and supported buckling, prestressed modal, direct harmonic or modal-reduced harmonic
consumer. This is not blanket admission of standalone thermal or arbitrary coupled graphs.
Authored models now preserve explicit thermal reference temperatures, and
`thermomechanical preload` binds a named retained thermal state to that compiled path.
The authored thermal-to-modal fixture passes actual-CAD execution after pre-execution driver/worker
replacement and cold import from declared objects. Other outer-root consumer/order combinations
and mid-phase restart remain unqualified; the equations below describe implemented assembly,
not a validation claim. See the
[thermal-prestress example](../examples/thermal-prestress.fea) and
[state-transfer contract](./thermomechanical#how-do-i-transfer-a-solved-temperature-field-into-structural-analysis).

## Retained thermal field histories

The thermal pipeline exposes temperature (K), temperature gradient (K/m), heat flux
(W/m²), and `thermal_energy_storage_rate_density` (W/m³) to the shared history evaluator.
The last quantity is a scalar element field: the volume average of
`rho * c_p * dT/dt`, stored in `thermal.energy_storage_rate` snapshots. It is distinct from
`heat_rate`, whose dimension is watts. A volume integral of storage-rate density has units of
power; an average retains W/m³. Exact requested history times are inserted into the solve schedule.

The insulated uniform-heating reference uses `Q = rho * c_p`, giving a 1 K/s temperature rise
and storage-rate density equal to Q for accepted time steps, independently of element volume.
Both Tet4 and Tet10 history outputs are checked against that analytic value. This does not qualify
the unfinished public thermal executor.

### Native coordination boundary

The native library has a single-step thermal coordinator for steady or
backward-Euler Tet4/Tet10 studies. It schedules the solve, shared observation partitions and
reductions, retained-history checkpoints/finalization, accuracy evaluation, every declared
temperature state, and final publication. Local and remote task-pool entrypoints use the same
coordinator. This is task placement, not a distributed thermal linear solve.

Observations consume the published thermal fields; they do not rerun the solve. A final-time
field retains its physical time coordinate. A complete history and a final field are separate
publication obligations: publishing one frame cannot satisfy a history declaration. Accuracy
validation binds the observation catalog to the original thermal workload and mesh.

Typed thermal-prestress completion retains the thermal prefix in local runtime publication and
remote CLI conversion. Portable thermal roots authenticate their complete declared object closure;
tasks transfer their declared input inventories rather than all historical catalog objects.
The authored modal fixture covers actual-CAD cold replay after pre-execution replacement.
The full outer-root consumer/order matrix, genuine durable whole-chain qualification,
bounded driver imports, checkpoint-prefix lineage for evicted samples and restart remain open gates.
Resource reservations are conservative admission bounds, not measured memory-use evidence.
Convergence, energy balance, thermoelastic transfer and formulation-specific release evidence
remain open.

### Authored observations and selective reads

Thermal-prestress runtime publication projects retained non-history observations and finalized
series reductions into the ordinary run. Histories retain their publication catalog and typed
object roots instead of being flattened into `run.observation_histories`. Discover them with
`storage::load_observation_publication(run_id)`; select a retained coordinate with
`analysis_observation_history_sample_by_run_id_op(run_id, observation_id, sample, context)`.
These are Rust runtime APIs, not new `.fea` syntax or plotting operations.

`ManifestAndChunks` reads the history metadata and selected sample partition; it does not read
unrelated history samples or canonical thermal fields. `WholeArtifact` necessarily reads the
whole history object. Both return the same typed sample, including physical coordinate, units,
source identity and reduction attribution. Missing, altered or unretained samples fail rather
than being interpolated or recomputed. Cold runtime reads use persisted objects, not the worker
session. Initial source authentication and persistence still inspect the complete declared closure;
selective query behavior does not establish bounded admission memory.

## What heat equation does RunMat assemble?

For temperature $T(\mathbf x,t)$, density $\rho$, specific heat $c_p$, conductivity $k$, and
volumetric source $Q$,

$$
\rho c_p\frac{\partial T}{\partial t}-\nabla\!\cdot(k\nabla T)=Q
\quad\text{in }\Omega.
$$

RunMat supports prescribed temperature on $\Gamma_T$, applied inward heat flux $\bar q$ on
$\Gamma_q$, and convection on $\Gamma_h$:

$$
T=\bar T,\qquad k\nabla T\cdot\mathbf n=\bar q,\qquad
-k\nabla T\cdot\mathbf n=h(T-T_\infty).
$$

Here $\mathbf n$ is the outward unit normal. Positive prescribed flux adds heat to the
body; the recovered heat-flux vector is $-k\nabla T$, whose outward normal component is
therefore $-\bar q$ on $\Gamma_q$. Convection uses the outward heat-loss convention.

Tetrahedral shape functions produce conductivity and consistent-capacity operators

$$
\mathsf K_T=\int_\Omega \mathsf B^{\mathsf T}k\mathsf B\,d\Omega
+\int_{\Gamma_h}\mathsf N^{\mathsf T}h\mathsf N\,d\Gamma,
$$

$$
\mathsf C_T=\int_\Omega \mathsf N^{\mathsf T}\rho c_p\mathsf N\,d\Omega,
\qquad
\mathbf q=\int_\Omega\mathsf N^{\mathsf T}Q\,d\Omega+
\int_{\Gamma_q}\mathsf N^{\mathsf T}\bar q\,d\Gamma+
\int_{\Gamma_h}\mathsf N^{\mathsf T}hT_\infty\,d\Gamma.
$$

Steady analysis solves $\mathsf K_T\mathbf T=\mathbf q$. The implemented transient assembly uses
backward Euler for one step,

$$
\left(\mathsf K_T+\frac{1}{\Delta t}\mathsf C_T\right)\mathbf T_{n+1}
=\mathbf q_{n+1}+\frac{1}{\Delta t}\mathsf C_T\mathbf T_n.
$$

Prescribed values are eliminated consistently. A steady connected component needs a temperature
anchor or positive Robin/convection weight; otherwise RunMat reports an underconstrained thermal
nullspace instead of inventing a reference temperature.

## Which thermal procedures are implemented?

| Procedure | Spatial/temporal realization | Required material/state |
| --- | --- | --- |
| `thermal steady` | Tet4/Tet10 Galerkin conduction plus a sparse linear solve | Conductivity and a well-posed reference/Robin boundary |
| `thermal transient` | The same spatial operator with consistent capacity and backward Euler | Conductivity, density, specific heat, time samples, and initial temperature |

Recovery computes solver-mesh temperature gradients
and heat-flow/energy-balance evidence; canonical volume and boundary projection are required by the
registered mesh-backed branches.

## How do I run a steady-state heat-transfer study?

```fea
model heated_part {
    assign material.aluminum to geometry.part.region.body

    constraint sink_temperature {
        thermal.temperature {
            target = geometry.part.region.heat_sink
            value = 293.15 K
        }
    }
    load applied_flux {
        thermal.heat_flux {
            target = geometry.part.region.hot_face
            value = 25000 W/m^2
        }
    }
}

study steady_temperature {
    geometry = geometry.part
    mesh = mesh.part
    model = model.heated_part
    analysis temperature { thermal steady {} }
    output temperature = analysis.temperature.field.node.temperature
}
```

The solid material must provide thermal conductivity. Density and specific heat are also required
when the capacity operator participates. `thermal.temperature` is an essential constraint;
`thermal.heat_flux` and `thermal.convection` are natural boundaries.

## How do I model convection to ambient air?

```fea
load exterior_convection {
    thermal.convection {
        target = geometry.part.region.exterior_faces
        ambient_temperature = 298.15 K
        coefficient = 35 W/m^2/K
    }
}
```

The coefficient $h$ contributes both a positive boundary matrix and the $hT_\infty$ source. Check
the actual exposed area and whether one coefficient is defensible across the entire selected region.
Natural convection, forced convection, and contact conductance are different models; do not reuse a
film coefficient as a contact resistance.

## How do I apply internal heat generation?

```fea
load die_loss {
    heat_generation on geometry.module.region.die = 8000000 W/m^3
}
```

Heat generation targets a volume and carries $\mathrm{W/m^3}$. Heat flux targets a surface and
carries $\mathrm{W/m^2}$. RunMat projects each source with an integration rule and records scalar
conservation evidence; an unmapped source or projection outside tolerance fails.

## How do I run transient thermal analysis with a power cycle?

```fea
signal power_cycle : heat_flux over time {
    samples = [
        { at = 0 s, value = 0 W/m^2 },
        { at = 2 s, value = 25000 W/m^2 },
        { at = 12 s, value = 25000 W/m^2 },
        { at = 14 s, value = 0 W/m^2 },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation cycling_flux {
    heat_flux {
        target = geometry.part.region.hot_face
        signal = signal.power_cycle
    }
}

analysis warmup {
    thermal transient {
        start_time = 0 s
        end_time = 20 s
        time_step = 0.1 s
    }
}
```

The declared time samples are part of the study. Choose $\Delta t$ small enough to resolve load
breakpoints and the shortest relevant diffusion time. Supply an initial temperature when zero kelvin
is not the intended state; initial state is never inferred from an ambient-temperature load.

## Why is my thermal model singular or physically wrong?

Common causes are:

- no prescribed temperature or positive convection term on a connected steady component;
- missing conductivity, density, or specific heat on a participating material region;
- a face selected where a solid heat source is required, or the reverse;
- overlapping temperature fields that prescribe conflicting nodal values;
- omitted contact/interface resistance between separate bodies;
- confusing positive inward prescribed flux with the outward component of recovered heat flux;
- time steps or mesh spacing too coarse for the thermal gradient.

RunMat separately reports missing material coverage, conflicting temperatures, unmapped sources,
complex ordinates unsupported by the thermal path, and failed projection conservation.

## What thermal outputs and validation checks should I use?

Request temperature fields and derived heat-flow quantities at locations supported by the procedure.
Check total applied power against boundary outflow and stored-energy change:

$$
P_{in}-P_{out}=\frac{d}{dt}\int_\Omega \rho c_p T\,d\Omega.
$$

Also perform mesh convergence at interfaces and hot spots, time-step convergence for transient work,
material-property sensitivity over the actual temperature range, and boundary-condition sensitivity.
Radiation is not among the currently documented thermal boundary kinds; if radiation matters, its
absence must be treated as a model limitation rather than hidden in an effective convection value.

See [thermal constraints](../constraints/thermal), [thermal loads](../loads/thermal),
[thermomechanical analysis](./thermomechanical), [electrothermal analysis](./electrothermal),
[outputs](../language/outputs-fields-and-observations), and [current status](../validation/status.md).
