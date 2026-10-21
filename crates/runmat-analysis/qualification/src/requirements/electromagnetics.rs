//! Standing physical acceptance requirements. Witness links describe assertion scope,
//! not successful execution or release qualification.
use crate::requirement::*;
const ELECTROSTATIC: &[Envelope] = &[
    envelope(
        Formulation::Electrostatic,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
];
const ELECTROSTATIC_TET4: &[Envelope] = &[envelope(
    Formulation::Electrostatic,
    MeshOrder::Tet4,
    ExecutionEnvelope::SerialCpuNumericalKernel,
)];
const STATIC: &[Envelope] = &[
    envelope(
        Formulation::Electrostatic,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
];
const CONDUCTION_TET4: &[Envelope] = &[envelope(
    Formulation::ElectricalConduction,
    MeshOrder::Tet4,
    ExecutionEnvelope::SerialCpuNumericalKernel,
)];
const MAGNETIC: &[Envelope] = &[
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
];
const MAGNETIC_TET4: &[Envelope] = &[envelope(
    Formulation::Magnetostatic,
    MeshOrder::Tet4,
    ExecutionEnvelope::SerialCpuNumericalKernel,
)];
const MAXWELL: &[Envelope] = &[envelope(
    Formulation::HarmonicMaxwell,
    MeshOrder::EdgeElements,
    ExecutionEnvelope::SerialCpuNumericalKernel,
)];
const MAGNETIC_AND_MAXWELL: &[Envelope] = &[
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::EdgeElements,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
];
const ALL_KERNEL: &[Envelope] = &[
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::EdgeElements,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
];
const STATIC_PUBLIC: &[Envelope] = &[
    envelope(
        Formulation::Electrostatic,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
];
const MAGNETIC_PUBLIC: &[Envelope] = &[
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
];
const ALL_EXECUTION: &[Envelope] = &[
    STATIC_PUBLIC[0],
    STATIC_PUBLIC[1],
    STATIC_PUBLIC[2],
    STATIC_PUBLIC[3],
    MAGNETIC_PUBLIC[0],
    MAGNETIC_PUBLIC[1],
    envelope(
        Formulation::Electrostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::SerialCpuNativeProcess,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::EncryptedRemote,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PartitionedCpu,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::SerialCpuNativeProcess,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::EncryptedRemote,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PartitionedCpu,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::SerialCpuNativeProcess,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::EncryptedRemote,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PartitionedCpu,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::SerialCpuNativeProcess,
    ),
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::EncryptedRemote,
    ),
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PartitionedCpu,
    ),
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
    envelope(
        Formulation::Electrostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::ElectricalConduction,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::Magnetostatic,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::HarmonicMaxwell,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
];
const FEA_EM_SCALAR_001: Requirement = Requirement {
    id: "FEA-EM-SCALAR-001",
    title: "Distinct electrostatic and conducting scalar-potential operators",
    stages: &[Stage::Electromagnetics],
    acceptance: concat!(
        "Electrostatics and steady electrical conduction are separately typed ",
        "physical formulations with their own constitutive response, source units ",
        "and acceptance; neither may be silently substituted for the other.",
    ),
    owners: &["crates/runmat-analysis/fea/src/physics/electrical/mod.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "operator",
            concat!(
                "Assemble dielectric electrostatic and conductivity steady-current potential ",
                "operators by physical material region on canonical topology. Resolve exact ",
                "terminals, sources and boundary regions; reject absent or incompatible ",
                "material and domain intent.",
            ),
            EvidenceKind::IndependentReference,
            STATIC,
        ),
        obligation(
            "recovery",
            concat!(
                "Recover electric field, displacement/charge, current density, loss, stored ",
                "energy and terminal quantities appropriate to the chosen equation, with ",
                "physical units, topology and original solved-potential lineage.",
            ),
            EvidenceKind::PhysicalBalance,
            STATIC,
        ),
    ],
    witnesses: &[
        Witness {
            id: "FEA-WITNESS-EM-SLAB",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "physics::electrical::tests::homogeneous_capacitor_and_conductor_match_indepe",
                    "ndent_slab_field_terminals_and_energy",
                ),
                source: "crates/runmat-analysis/fea/src/physics/electrical/tests.rs",
            },
            claims: &[
                CoverageClaim {
                    obligation: "operator",
                    envelopes: STATIC,
                    boundary: concat!(
"Tet4/Tet10 homogeneous slab capacitor and conductor potentials, fields, ",
"terminal flux, energy and Joule power; not arbitrary geometry or nonlinear ",
"material.",
),
                },
                CoverageClaim {
                    obligation: "recovery",
                    envelopes: STATIC,
                    boundary: concat!(
                        "Same slab constitutive vectors and global terminal/energy identities; no ",
                        "general charge-density reconstruction claim.",
                    ),
                },
            ],
        },
        Witness {
            id: "FEA-WITNESS-EM-LAYERS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "physics::electrical::tests::layered_materials_match_series_capacitance_and_r",
                    "esistance_with_interface_flux_continuity",
                ),
                source: "crates/runmat-analysis/fea/src/physics/electrical/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "operator",
                envelopes: STATIC,
                boundary: concat!(
                    "Both orders, aligned two-material slab series capacitance/resistance and ",
                    "interface flux; no curved interface claim.",
                ),
            }],
        },
    ],
};
const FEA_EM_INPUT_001: Requirement = Requirement {
    id: "FEA-EM-INPUT-001",
    title: "Typed electrical and magnetic sources and rank-correct constraints",
    stages: &[Stage::Electromagnetics],
    acceptance: concat!(
        "Use shared typed excitation and constraint authorities for authored ",
        "voltage, charge, current, current-density, complex phasor, incident and ",
        "supported port inputs. Constraint plans remain rank-correct and ",
        "attributable to authored objects.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/electrical/boundary.rs",
        "crates/runmat-analysis/fea/src/physics/magnetic/plan.rs",
    ],
    dependencies: &[],
    obligations: &[
        obligation(
            "scalar_constraints",
            concat!(
                "Support prescribed potential, equipotential terminals, floating conductors, ",
                "integral charge/current conditions and gauge/reference handling; diagnose ",
                "rank deficiency, conflicts, unsupported combinations and incorrect source ",
                "units before solving.",
            ),
            EvidenceKind::SemanticRealization,
            STATIC,
        ),
        obligation(
            "magnetic_constraints",
            concat!(
                "Declare gauge, symmetry, periodic and supported complex phase-shifted ",
                "relations for magnetic and Maxwell operators, with source compatibility and ",
                "rank evidence.",
            ),
            EvidenceKind::SemanticRealization,
            MAGNETIC_AND_MAXWELL,
        ),
        obligation(
            "excitation",
            concat!(
                "Shared signals and realized artifacts preserve ",
                "voltage/charge/current/current-density and phasor magnitude, phase, exact ",
                "region realization, parameter histories where admissible and source/power ",
                "balances; no family-private history or callback authority.",
            ),
            EvidenceKind::ExecutionEquivalence,
            ALL_KERNEL,
        ),
    ],
    witnesses: &[
        Witness {
            id: "FEA-WITNESS-EM-SCALAR-REJECT",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "physics::electrical::tests::missing_anchors_conflicts_unsupported_intent_and",
                    "_invalid_material_fail_closed",
                ),
                source: "crates/runmat-analysis/fea/src/physics/electrical/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "scalar_constraints",
                envelopes: CONDUCTION_TET4,
                boundary: concat!(
                    "Tet4 rejects missing anchor, conflicting potentials, floating ",
                    "conductor, bad material, target and wrong harmonic intent; rejection is not ",
                    "implementation of floating conductors.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-EM-MAGNETIC-AUTHORED",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "physics::magnetic::tests::authored_material_current_ground_and_gauge_compile",
                    "_into_physical_solve",
                ),
                source: "crates/runmat-analysis/fea/src/physics/magnetic/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "magnetic_constraints",
                envelopes: MAGNETIC,
                boundary: concat!(
                    "Affine Tet4/Tet10 volume-current, full grounded-boundary and gauge plan ",
                    "compared with same kernel; rejects missing gauge, surface-current ",
                    "substitution and wrong domain. No independent physical-reference or ",
                    "periodic claim.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-EM-CHARGE-SUPERPOSITION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "physics::electrical::tests::charge_density::exact_volume_region_and_signed_s",
                    "uperposition_preserve_charge_and_fields",
                ),
                source: "crates/runmat-analysis/fea/src/physics/electrical/tests/charge_density.rs",
            },
            claims: &[CoverageClaim {
                obligation: "excitation",
                envelopes: ELECTROSTATIC,
                boundary: concat!(
                    "Both orders, constant real volume-charge exact target and signed ",
                    "superposition preserve potential, net/absolute charge and two projection ",
                    "records; no generated or frequency artifact equivalence.",
                ),
            }],
        },
    ],
};
const FEA_EM_MAGNETOSTATIC_001: Requirement = Requirement {
    id: "FEA-EM-MAGNETOSTATIC-001",
    title: "Documented magnetostatic formulation and physical recovery",
    stages: &[Stage::Electromagnetics],
    acceptance: concat!(
        "Declare vector or scalar potential, gauge, source representation and ",
        "validated material envelope. Resolve coils, currents, magnetic regions and ",
        "boundaries through canonical topology. Recover field, flux density, energy ",
        "and force only within independently supported formulation scope.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/magnetic/assembly.rs",
        "crates/runmat-analysis/fea/src/physics/magnetic/response.rs",
    ],
    dependencies: &[],
    obligations: &[
        obligation(
            "formulation",
            concat!(
                "Integrated magnetic operator and gauge enforce the documented physical ",
                "equations and source compatibility, with exact edge/region identities and ",
                "validated material restrictions.",
            ),
            EvidenceKind::IndependentReference,
            MAGNETIC,
        ),
        obligation(
            "fields",
            concat!(
                "Recover magnetic field, flux density and energy from the solved state, ",
                "report physical residual, gauge residual, source work and divergence/flux ",
                "closure; magnetic force needs separate formulation and accuracy evidence.",
            ),
            EvidenceKind::PhysicalBalance,
            MAGNETIC,
        ),
    ],
    witnesses: &[
        Witness {
            id: "FEA-WITNESS-EM-MAGNETIC-SERIES",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "physics::magnetic::tests::square_uniform_current_has_independent_potential_f",
                    "ield_and_energy_convergence",
                ),
                source: "crates/runmat-analysis/fea/src/physics/magnetic/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "formulation",
                envelopes: MAGNETIC_TET4,
                boundary: concat!(
                    "Affine Tet4 extruded unit square, uniform axial current and grounded ",
                    "boundary; independent Fourier-series potential, curl field and energy with ",
                    "tail bounds and three meshes. No Tet10 convergence or force evidence.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-EM-MAGNETIC-SCALING",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "physics::magnetic::tests::sign_permeability_and_affine_geometry_order_preser",
                    "ve_physical_scaling",
                ),
                source: "crates/runmat-analysis/fea/src/physics/magnetic/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "fields",
                envelopes: MAGNETIC,
                boundary: concat!(
                    "Affine Tet4 and Tet10 geometry with lowest-order edge basis: ",
                    "sign/permeability scaling against a Tet4 solve; Tet10 geometry is not ",
                    "higher-order magnetic approximation.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-EM-MAGNETIC-FLUX",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "physics::magnetic::tests::recovered_curl_has_shared_face_normal_flux_continu",
                    "ity_and_global_flux_closure",
                ),
                source: "crates/runmat-analysis/fea/src/physics/magnetic/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "fields",
                envelopes: MAGNETIC_TET4,
                boundary: concat!(
                    "Affine Tet4 shared-face normal curl continuity and global boundary flux ",
                    "closure; no general magnetic force recovery.",
                ),
            }],
        },
    ],
};
const FEA_EM_MAXWELL_001: Requirement = Requirement {
    id: "FEA-EM-MAXWELL-001",
    title: "Integrated harmonic edge-element Maxwell analysis",
    stages: &[Stage::Electromagnetics],
    acceptance: concat!(
        "Harmonic Maxwell is a distinct complex frequency-domain formulation using ",
        "canonical tetrahedral edge topology and integrated Nedelec functions rather ",
        "than approximate vector proxies.",
    ),
    owners: &["crates/runmat-analysis/fea/src/physics/magnetic/assembly.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "basis",
            concat!(
                "Complete oriented tetrahedral edge topology, integrated Nedelec basis and ",
                "material-interface continuity; assemble physical curl-curl, mass/loss, ",
                "source, boundary and gauge terms per material region.",
            ),
            EvidenceKind::IndependentReference,
            MAXWELL,
        ),
        obligation(
            "frequency",
            concat!(
                "Support complex coefficients and deterministic frequency-sweep composition, ",
                "complex source phasors, incident and supported port excitation with ",
                "reusable pattern/value identities and documented frequency conventions.",
            ),
            EvidenceKind::ExecutionEquivalence,
            MAXWELL,
        ),
        obligation(
            "reconstruction",
            concat!(
                "Use documented projection/reconstruction for vector recovery; retain ",
                "original divergence, energy, power, boundary and linear-solver balance ",
                "diagnostics.",
            ),
            EvidenceKind::PhysicalBalance,
            MAXWELL,
        ),
    ],
    witnesses: &[],
};
const FEA_EM_QUALIFICATION_001: Requirement = Requirement {
    id: "FEA-EM-QUALIFICATION-001",
    title: "Independent per-formulation evidence and governed electromagnetic execution",
    stages: &[Stage::Electromagnetics],
    acceptance: concat!(
        "Promote each electromagnetic formulation separately only after independent ",
        "physical, constraint, convergence, recovery, source and execution evidence ",
        "passes. Capability metadata and successful linear solves alone do not ",
        "establish release qualification.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/execution/electrical_static.rs",
        "crates/runmat-analysis/fea/src/execution/magnetic_static.rs",
    ],
    dependencies: &[],
    obligations: &[
        obligation(
            "references",
            concat!(
                "Pass independent capacitors, conductive bars, magnetostatic references, ",
                "cavity and waveguide modes, skin effect, material interfaces and ",
                "manufactured fields applicable to each formulation, with mesh and frequency ",
                "convergence and gauge/divergence, field, energy and power evidence.",
            ),
            EvidenceKind::SpatialConvergence,
            ALL_KERNEL,
        ),
        obligation(
            "execution",
            concat!(
                "Parallel independent frequencies and sources and justified mature operator ",
                "partitions preserve equivalent original fields and balances. Whole-task ",
                "resource admission, cancellation, retries, durable/remote lineage and ",
                "publication remain governed across every claimed placement.",
            ),
            EvidenceKind::ExecutionEquivalence,
            ALL_EXECUTION,
        ),
        obligation(
            "accuracy",
            concat!(
                "Publish correct formulation and release documentation with original ",
                "controls and evidence. Missing, failed or indeterminate mandatory accuracy ",
                "prevents production publication; cold claims require authenticated original ",
                "source lineage and never reconstruct lost physics by decoding summaries.",
            ),
            EvidenceKind::PublicationIntegrity,
            ALL_EXECUTION,
        ),
    ],
    witnesses: &[
        Witness {
            id: "FEA-WITNESS-EM-QUADRATIC",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "physics::electrical::tests::harmonic_quadratic_potential_has_tet10_exact_fie",
                    "ld_and_tet4_convergence",
                ),
                source: "crates/runmat-analysis/fea/src/physics/electrical/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "references",
                envelopes: STATIC,
                boundary: concat!(
                    "Static harmonic polynomial potential x²-y² for both scalar equations: Tet10 ",
                    "exact field/energy and Tet4 refinement across three meshes; harmonic ",
                    "polynomial is not harmonic Maxwell.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-EM-SCALAR-COLD",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "execution::electrical_static::tests::equation_order_matrix_retains_exact_fie",
                    "lds_and_cold_source_bound_identity",
                ),
                source: "crates/runmat-analysis/fea/src/execution/electrical_static/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "accuracy",
                envelopes: STATIC_PUBLIC,
                boundary: concat!(
                    "In-process both scalar equations and orders retain original slab fields and ",
                    "canonical source-bound cold identity; changed controls/equation/mesh ",
                    "reject. No producer death, remote or durable execution.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-EM-MAGNETIC-COLD",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "execution::magnetic_static::tests::magnetic_portable_source_bound_cold_resul",
                    "t_preserves_both_orders_and_signs",
                ),
                source: "crates/runmat-analysis/fea/src/execution/magnetic_static/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "accuracy",
                envelopes: MAGNETIC_PUBLIC,
                boundary: concat!(
                    "In-process affine both-order magnetic source-bound cold bytes and ",
                    "sign/zero-current inventory/work; no native process, restart, partition or ",
                    "measured RSS evidence.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-EM-CHARGED-REFINEMENT",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "physics::electrical::tests::charge_density::tet4_charged_slab_has_independen",
                    "t_spatial_field_and_energy_convergence",
                ),
                source: "crates/runmat-analysis/fea/src/physics/electrical/tests/charge_density.rs",
            },
            claims: &[CoverageClaim {
                obligation: "references",
                envelopes: ELECTROSTATIC_TET4,
                boundary: concat!(
                    "Uniform-charge Tet4 slab on three meshes: independent quadratic potential, ",
                    "linear field, charge conservation and energy convergence; no Tet10 or ",
                    "nonlinear dielectric claim.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-EM-ELECTRICAL-ACCURACY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "execution::accuracy_source::electrical_tests::electrical_receipts_supply_onl",
                    "y_retained_algebraic_equilibrium_evidence",
                ),
                source:
                    "crates/runmat-analysis/fea/src/execution/accuracy_source/electrical_tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "accuracy",
                envelopes: STATIC_PUBLIC,
                boundary: concat!(
                    "Both equations/orders retained receipts provide only algebraic ",
                    "residual/closure evidence; independent spatial/convergence obligations ",
                    "remain indeterminate. Not field-accuracy qualification.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-EM-MAGNETIC-ACCURACY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "execution::accuracy_source::magnetic_tests::magnetic_receipts_supply_only_or",
                    "iginal_algebraic_equilibrium_evidence",
                ),
                source:
                    "crates/runmat-analysis/fea/src/execution/accuracy_source/magnetic_tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "accuracy",
                envelopes: MAGNETIC_PUBLIC,
                boundary: concat!(
                    "Both affine geometry orders/signs receipts preserve original algebraic ",
                    "residual/gauge/energy claims and block production when independent ",
                    "convergence evidence is missing; no process or distributed proof.",
                ),
            }],
        },
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    FEA_EM_SCALAR_001,
    FEA_EM_INPUT_001,
    FEA_EM_MAGNETOSTATIC_001,
    FEA_EM_MAXWELL_001,
    FEA_EM_QUALIFICATION_001,
];
