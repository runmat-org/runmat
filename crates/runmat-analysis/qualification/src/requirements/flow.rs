//! Non-turbulent flow end state (REG-FLUID-001 to 022): solvers, boundaries, heat, materials, physics, outputs, verification and FSI.
use crate::requirement::*;
const FLOW: &[Envelope] = cells!(
    [
        Formulation::IncompressibleLaminarSteady,
        Formulation::IncompressibleLaminarTransient
    ],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const FLOWPAR: &[Envelope] = cells!(
    [
        Formulation::IncompressibleLaminarSteady,
        Formulation::IncompressibleLaminarTransient
    ],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy,
        ExecutionEnvelope::ParallelCpu
    ]
);
const STEADY: &[Envelope] = cells!(
    [Formulation::IncompressibleLaminarSteady],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const TRANSIENT: &[Envelope] = cells!(
    [Formulation::IncompressibleLaminarTransient],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const HEAT: &[Envelope] = cells!(
    [Formulation::ConjugateHeatTransfer],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const LOWMACH: &[Envelope] = cells!(
    [Formulation::LowMachFlow],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const NEWT: &[Envelope] = cells!(
    [Formulation::GeneralizedNewtonianFlow],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const VISCO: &[Envelope] = cells!(
    [Formulation::ViscoelasticFlow],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const POROUS: &[Envelope] = cells!(
    [Formulation::PorousMediaFlow],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const SCALAR: &[Envelope] = cells!(
    [Formulation::ScalarTransport],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const FREE: &[Envelope] = cells!(
    [Formulation::FreeSurfaceFlow],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const ROT: &[Envelope] = cells!(
    [Formulation::RotatingFrameFlow],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const LUBE: &[Envelope] = cells!(
    [Formulation::ThinFilmLubrication],
    [MeshOrder::CanonicalMixedTopology],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const EK: &[Envelope] = cells!(
    [Formulation::Electrokinetics],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const PART: &[Envelope] = cells!(
    [Formulation::ParticleTracing],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const COMP: &[Envelope] = cells!(
    [
        Formulation::CompressibleLaminarFlow,
        Formulation::PipeNetwork
    ],
    [MeshOrder::CanonicalMixedTopology],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const AXI: &[Envelope] = cells!(
    [
        Formulation::IncompressibleLaminarSteady,
        Formulation::IncompressibleLaminarTransient
    ],
    [MeshOrder::CanonicalMixedTopology],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const FSI: &[Envelope] = cells!(
    [Formulation::FluidStructureInteraction],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const OUT: &[Envelope] = cells!(
    [
        Formulation::IncompressibleLaminarSteady,
        Formulation::IncompressibleLaminarTransient,
        Formulation::ConjugateHeatTransfer
    ],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const SUITE: &[Envelope] = cells!(
    [
        Formulation::IncompressibleLaminarSteady,
        Formulation::IncompressibleLaminarTransient,
        Formulation::ConjugateHeatTransfer,
        Formulation::FluidStructureInteraction
    ],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [ExecutionEnvelope::PublicStudy]
);
const FEA_FLUID_SOLVERS_001: Requirement = Requirement {
    id: "FEA-FLUID-SOLVERS-001",
    title: "Robust, scalable and accurate laminar flow solvers",
    stages: &[Stage::Fluid],
    acceptance: "Non-turbulent flow solves robustly at every Reynolds number the laminar scope covers: a coupled Newton solver with pseudo-transient and Reynolds continuation, scalable block-preconditioned linear solvers, second-order time integration, and second-order accurate discretization on CAD tetrahedra, from `runmat run`.",
    owners: &["crates/runmat-analysis/fea/src/physics/cfd/canonical.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "coupled_newton_continuation",
            "A fully coupled pressure-based Newton solver with pseudo-transient continuation, Reynolds and rheology continuation and line search converges the Ghia lid-driven cavity at Re 1000 from rest in at most 15 Newton steps, with psi_min 0.1189366 at (0.5308, 0.5652) against the Botella-Peyret reference; stagnation is reported in the run evidence.",
            EvidenceKind::IndependentReference,
            FLOW,
        ),
        obligation(
            "residual_history",
            "Per-iteration residual history and nonlinear and linear iteration counts publish as results for every flow run.",
            EvidenceKind::PublicationIntegrity,
            FLOW,
        ),
        obligation(
            "block_linear_solvers",
            "Flow linear systems solve with FGMRES using block-triangular preconditioning (PCD or LSC Schur approximations, augmented Lagrangian at high Reynolds number) and AMG on velocity and pressure blocks, through the shared sparse layer including its GPU path; linear iterations stay within 1.5 times from 100 thousand to 10 million cells on the 3D cavity.",
            EvidenceKind::PerformanceMeasurement,
            FLOWPAR,
        ),
        obligation(
            "second_order_time",
            "BDF2 with adaptive steps is the default transient scheme and a rotational pressure-correction projection scheme serves long transients: Schafer-Turek 2D-2 (Re 100) gives c_D,max 3.22 to 3.24, c_L,max 0.99 to 1.01 and Strouhal number 0.295 to 0.305; 2D-3 gives c_D,max 2.950921 and c_L,max 0.47795 (references checked against the paper before use).",
            EvidenceKind::TemporalConvergence,
            TRANSIENT,
        ),
        obligation(
            "accurate_discretization",
            "Second-order coupled finite volume with non-orthogonal correction and polyhedral dual, and P2-P1 or VMS finite elements on Tet10, reach design order on Kovasznay flow at Re 40 and reproduce Schafer-Turek 2D-1 (c_D 5.579535, c_L 0.0106189, pressure drop 0.1175201) and 3D-1Z (c_D 6.185333, c_L 0.009401, pressure drop 0.171007).",
            EvidenceKind::SpatialConvergence,
            STEADY,
        ),
    ],
};
const FEA_FLUID_BOUNDARY_002: Requirement = Requirement {
    id: "FEA-FLUID-BOUNDARY-002",
    title: "Complete fluid boundary and excitation set",
    stages: &[Stage::Fluid],
    acceptance: "Every flow boundary condition and excitation the language declares runs: velocity profiles (function, table, fully developed), inlet_velocity and wall_motion excitations, pressure inlets, open outlets with backflow stabilization, Navier slip and symmetry, periodic with imposed pressure drop or flow rate, rotating walls, fan curves, screens and porous jumps, and inlet temperature and wall heat conditions. Backflow at outlets is stabilized, not rejected.",
    owners: &["crates/runmat-analysis/fea/src/physics/cfd/canonical.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "profiles_and_excitations",
            "Velocity profiles (function, table, fully developed, periodic-spline sampled pulsatile waveforms from a CSV) and the `inlet_velocity` and `wall_motion` excitations run; square duct reproduces fRe 56.91 and circular Hagen-Poiseuille fRe 64.",
            EvidenceKind::IndependentReference,
            FLOW,
        ),
        obligation(
            "pressure_and_open_outlets",
            "Pressure inlets and directional do-nothing open outlets with backflow stabilization run with flow reversing at the outlet; a closed pressure reference removes the null space without changing the solution.",
            EvidenceKind::SemanticRealization,
            FLOW,
        ),
        obligation(
            "slip_symmetry_periodic_rotating",
            "Navier slip, symmetry, periodic boundaries with imposed pressure drop or flow rate, and rotating walls run: periodic channel flow rate is exact, Taylor-Couette torque matches u_theta = A r + B / r.",
            EvidenceKind::IndependentReference,
            FLOW,
        ),
        obligation(
            "fans_screens_porous_jumps",
            "Fan curves, screens and porous jumps apply a pressure jump consistent with their authored law and conserve mass.",
            EvidenceKind::PhysicalBalance,
            FLOW,
        ),
        obligation(
            "thermal_boundaries",
            "Inlet temperature, wall temperature, wall heat flux and convective wall conditions apply to the fluid energy equation and conserve energy.",
            EvidenceKind::PhysicalBalance,
            HEAT,
        ),
    ],
};
const FEA_FLUID_HEAT_001: Requirement = Requirement {
    id: "FEA-FLUID-HEAT-001",
    title: "Heat in the fluid: forced and natural convection, CHT and low-Mach flow",
    stages: &[Stage::Fluid, Stage::FluidStructureInteraction],
    acceptance: "The energy equation runs in the fluid with forced convection, two-way Boussinesq natural convection, viscous heating and temperature-dependent properties, two-way conjugate heat transfer with thermal solids over perfect and resistive interfaces, radiation coupling, and low-Mach variable-density flow.",
    owners: &["crates/runmat-analysis/fea/src/physics/cht/interface.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "forced_convection",
            "Fluid energy transport reproduces the pipe Nusselt numbers 3.66 (constant wall temperature) and 4.364 (constant heat flux), and the Graetz thermal entry length.",
            EvidenceKind::IndependentReference,
            HEAT,
        ),
        obligation(
            "natural_convection",
            "Two-way Boussinesq natural convection reproduces the de Vahl Davis differentially heated cavity Nusselt numbers 1.118, 2.243, 4.519 and 8.800 at Ra 1e3 to 1e6 within 1 percent, the Tric cube values 1.0700, 2.0542, 4.3371 and 8.6407, and Rayleigh-Benard onset at Ra_c 1707.76; `conjugate_heat_transfer steady { buoyancy = boussinesq; gravity = ... }` runs the sealed enclosure study.",
            EvidenceKind::IndependentReference,
            HEAT,
        ),
        obligation(
            "viscous_heating_and_properties",
            "Viscous heating and temperature-dependent density, viscosity, conductivity and specific heat run and close the energy balance to the declared tolerance.",
            EvidenceKind::PhysicalBalance,
            HEAT,
        ),
        obligation(
            "two_way_cht",
            "Two-way conjugate heat transfer with thermal solids converges with conservative interface flux continuity (residual published) and reproduces series conduction and plug-flow references; the liquid cold plate study reports pressure drop and peak plate temperature at 0.03 kg/s and 1.2 kW.",
            EvidenceKind::PhysicalBalance,
            HEAT,
        ),
        obligation(
            "radiation_coupling",
            "Surface radiation couples to the fluid and solid energy balances with emissivity and ambient or enclosure temperature and closes the balance.",
            EvidenceKind::PhysicalBalance,
            HEAT,
        ),
        obligation(
            "low_mach_variable_density",
            "Low-Mach flow with density from temperature and pressure reproduces the heated vertical channel variable-density closed form.",
            EvidenceKind::IndependentReference,
            LOWMACH,
        ),
    ],
};
const FEA_FLUID_MATERIALS_001: Requirement = Requirement {
    id: "FEA-FLUID-MATERIALS-001",
    title: "Rheology, porous media and species transport",
    stages: &[Stage::Fluid],
    acceptance: "Fluid material behavior runs: generalized-Newtonian, viscoelastic, porous and passive-scalar models with their typed input forms and the shear-rate, apparent-viscosity and concentration fields.",
    owners: &["crates/runmat-analysis/fea/src/physics/cfd/canonical.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "generalized_newtonian",
            "Power law, Carreau and Carreau-Yasuda, Cross, Casson, regularized Bingham and Herschel-Bulkley with Arrhenius and WLF temperature dependence run, publish shear-rate and apparent-viscosity fields, reproduce the power-law channel profile and the Buckingham-Reiner pipe flow rate.",
            EvidenceKind::IndependentReference,
            NEWT,
        ),
        obligation(
            "viscoelastic",
            "Oldroyd-B, Giesekus and PTT with log-conformation run, and the Oldroyd-B cylinder-in-channel drag against Weissenberg number matches Alves et al.",
            EvidenceKind::IndependentReference,
            VISCO,
        ),
        obligation(
            "porous_zones",
            "Darcy, Brinkman and Forchheimer porous zones with anisotropic permeability and local thermal equilibrium run: Darcy u = (K/mu) G, the Brinkman channel profile and the Ergun packed-bed pressure drop; the filter study reports its housing-plus-cartridge pressure drop.",
            EvidenceKind::IndependentReference,
            POROUS,
        ),
        obligation(
            "species_and_scalars",
            "Passive scalar advection-diffusion, mass-action and Arrhenius reactions, residence time and a mixing index run: Graetz-type mass transfer and laminar tubular reactor conversion match their closed forms; the static mixer reports outlet concentration coefficient of variation and pressure drop.",
            EvidenceKind::IndependentReference,
            SCALAR,
        ),
    ],
};
const FEA_FLUID_PHYSICS_001: Requirement = Requirement {
    id: "FEA-FLUID-PHYSICS-001",
    title: "Free surface, rotating frames, lubrication, electrokinetics, particles, compressibility and reduced dimensions",
    stages: &[Stage::Fluid],
    acceptance: "The remaining non-turbulent flow physics run: free surface and two-phase flow, rotating frames and sliding mesh, thin-film lubrication, electrokinetics, particle tracing, compressible laminar flow with rarefied slip, one-dimensional pipe networks coupled to three-dimensional flow, and two-dimensional and axisymmetric studies.",
    owners: &["crates/runmat-analysis/fea/src/physics/cfd/canonical.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "free_surface_two_phase",
            "Volume-of-fluid or level-set free surface with surface tension, contact angle and Marangoni stress reproduces the static drop Laplace pressure 2 sigma / R with bounded spurious currents and Jurin capillary rise.",
            EvidenceKind::IndependentReference,
            FREE,
        ),
        obligation(
            "rotating_frames_and_sliding_mesh",
            "Rotating reference frames (MRF and SRF) and sliding mesh reproduce solid-body rotation in a rotating annulus and Taylor vortex onset at Re_c about 68.2 for radius ratio 0.5.",
            EvidenceKind::IndependentReference,
            ROT,
        ),
        obligation(
            "thin_film_lubrication",
            "The Reynolds equation with Elrod-JFO cavitation, journal and thrust bearings, squeeze film and coupling to structural deformation reproduces long and short journal bearing load and attitude angle (Sommerfeld); the journal bearing study reports load capacity at 3000 rpm and eccentricity ratio 0.6.",
            EvidenceKind::IndependentReference,
            LUBE,
        ),
        obligation(
            "electrokinetics",
            "Electroosmotic slip (Helmholtz-Smoluchowski), Poisson-Nernst-Planck and dielectrophoresis run; electroosmotic plug flow equals -epsilon zeta E / mu.",
            EvidenceKind::IndependentReference,
            EK,
        ),
        obligation(
            "particle_tracing",
            "Massless particles (residence time, mixing) and inertial particles with drag are traced one-way through a solved flow, and Stokes settling velocity is reproduced.",
            EvidenceKind::IndependentReference,
            PART,
        ),
        obligation(
            "compressible_and_networks",
            "Compressible laminar flow and rarefied slip reproduce microchannel gas flow with slip against the closed form, and one-dimensional pipe networks coupled to three-dimensional regions reproduce Darcy-Weisbach network pressure drop.",
            EvidenceKind::IndependentReference,
            COMP,
        ),
        obligation(
            "two_dimensional_and_axisymmetric",
            "Two-dimensional and axisymmetric studies run through the same study language and reproduce the three-dimensional pipe and sphere results.",
            EvidenceKind::IndependentReference,
            AXI,
        ),
    ],
};
const FEA_FLUID_OUTPUTS_001: Requirement = Requirement {
    id: "FEA-FLUID-OUTPUTS-001",
    title: "Flow outputs engineers ask for",
    stages: &[Stage::Fluid],
    acceptance: "Flow results publish the derived quantities designs are judged on, with correct dimensions, location and conservation: forces, moments and coefficients on named surfaces, wall shear stress and its time statistics, pressure drop and flow rate between named faces, heat-transfer quantities, vorticity and Q-criterion, and nodal velocity for streamlines.",
    owners: &["crates/runmat-analysis/fea/src/physics/cfd/canonical.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "forces_and_coefficients",
            "Forces, moments and drag and lift coefficients on named surfaces reproduce sphere Stokes drag 6 pi mu a U and C_D about 1.09 at Re 100, and cylinder C_D about 1.52 and wake length L/D about 2.3 at Re 40.",
            EvidenceKind::IndependentReference,
            OUT,
        ),
        obligation(
            "wall_shear_stress_fields",
            "A wall shear stress field, time-averaged wall shear stress and the oscillatory shear index publish for pulsatile flow; the carotid study reports minimum time-averaged wall shear stress in the sinus.",
            EvidenceKind::PhysicalBalance,
            OUT,
        ),
        obligation(
            "pressure_drop_and_flow_rate",
            "`pressure_drop` between named faces and flow rate through named faces publish and satisfy mass conservation to the declared tolerance.",
            EvidenceKind::PhysicalBalance,
            OUT,
        ),
        obligation(
            "heat_transfer_quantities",
            "Nusselt number, heat-transfer coefficient and bulk temperature publish for heated walls and agree with the energy balance.",
            EvidenceKind::PhysicalBalance,
            HEAT,
        ),
        obligation(
            "vorticity_and_streamlines",
            "Vorticity, Q-criterion and nodal velocity (for streamlines) publish with the declared shapes and locations.",
            EvidenceKind::PublicationIntegrity,
            OUT,
        ),
    ],
};
const FEA_FLUID_VERIFICATION_001: Requirement = Requirement {
    id: "FEA-FLUID-VERIFICATION-001",
    title: "Published flow verification suite",
    stages: &[Stage::Fluid, Stage::Release],
    acceptance: "Every flow witness is published as a suite page with mesh-convergence plots, wall time, memory and iteration counts, generated from runs, with the literature values checked against the source papers before use and misses published.",
    owners: &["crates/runmat-analysis/fea/src/physics/cfd/canonical.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "benchmark_suite_publication",
            "The suite at docs/simulation/validation/fluid contains the lid-driven cavity, Schafer-Turek 2D and 3D cases, Kovasznay, de Vahl Davis and Tric natural convection, duct and pipe references, Taylor-Couette, Turek-Hron FSI and the physics rows above, each with convergence, wall time, memory and iteration counts, regenerated from runs.",
            EvidenceKind::IndependentReference,
            SUITE,
        ),
    ],
};
const FEA_COUPLING_FSI_002: Requirement = Requirement {
    id: "FEA-COUPLING-FSI-002",
    title: "Fluid loads on structures and two-way FSI",
    stages: &[Stage::FluidStructureInteraction],
    acceptance: "Fluid pressure and wall shear drive structural studies one-way, two-way partitioned FSI with ALE mesh motion and large-deformation solids runs from the public command, and compliant-channel FSI matches lubrication theory.",
    owners: &["crates/runmat-analysis/fea/src/physics/fsi/interface.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "one_way_fluid_loads",
            "Pressure and wall shear stress from a solved flow map conservatively into a structural study as loads (total force and moment preserved), including wind-pressure-style mapped loads.",
            EvidenceKind::PhysicalBalance,
            FSI,
        ),
        obligation(
            "two_way_ale",
            "Two-way partitioned FSI with Aitken relaxation, Laplace ALE mesh motion with geometric conservation and Newmark structure reproduces the Turek-Hron FSI1, FSI2 and FSI3 tip displacements and forces within the published scatter.",
            EvidenceKind::IndependentReference,
            FSI,
        ),
        obligation(
            "compliant_channel",
            "A compliant channel reproduces lubrication-theory deflection within 5 percent and conserves interface work.",
            EvidenceKind::IndependentReference,
            FSI,
        ),
        obligation(
            "large_deformation_solids_in_fsi",
            "The structural side of FSI accepts large-deformation solids with the same coupling residuals and conservation.",
            EvidenceKind::PhysicalBalance,
            FSI,
        ),
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    FEA_FLUID_SOLVERS_001,
    FEA_FLUID_BOUNDARY_002,
    FEA_FLUID_HEAT_001,
    FEA_FLUID_MATERIALS_001,
    FEA_FLUID_PHYSICS_001,
    FEA_FLUID_OUTPUTS_001,
    FEA_FLUID_VERIFICATION_001,
    FEA_COUPLING_FSI_002,
];
