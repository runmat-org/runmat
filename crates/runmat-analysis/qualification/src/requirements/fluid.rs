//! Standing incompressible Newtonian, laminar, single-phase fluid requirements.
//! Coupled heat/interface physics is a separate qualification responsibility.
//! Witness links declare assertion scope; they do not declare passing qualification.
use crate::requirement::*;
const STEADY_FORMULATION: Formulation = Formulation::IncompressibleLaminarSteady;
const TRANSIENT_FORMULATION: Formulation = Formulation::IncompressibleLaminarTransient;
const STEADY_KERNEL: &[Envelope] = &[envelope(
    STEADY_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::SerialCpuNumericalKernel,
)];
const TRANSIENT_KERNEL: &[Envelope] = &[envelope(
    TRANSIENT_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::SerialCpuNumericalKernel,
)];
const STEADY_NATIVE: &[Envelope] = &[envelope(
    STEADY_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::SerialCpuNativeProcess,
)];
const TRANSIENT_NATIVE: &[Envelope] = &[envelope(
    TRANSIENT_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::SerialCpuNativeProcess,
)];
const STEADY_PUBLIC: &[Envelope] = &[envelope(
    STEADY_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::SerialCpuPublicStudy,
)];
const TRANSIENT_PUBLIC: &[Envelope] = &[envelope(
    TRANSIENT_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::SerialCpuPublicStudy,
)];
const STEADY_DURABLE: &[Envelope] = &[envelope(
    STEADY_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::Durable,
)];
const TRANSIENT_DURABLE: &[Envelope] = &[envelope(
    TRANSIENT_FORMULATION,
    MeshOrder::CanonicalTetrahedra,
    ExecutionEnvelope::Durable,
)];
const CORE: &[Envelope] = &[
    STEADY_KERNEL[0],
    TRANSIENT_KERNEL[0],
    STEADY_NATIVE[0],
    TRANSIENT_NATIVE[0],
    STEADY_PUBLIC[0],
    TRANSIENT_PUBLIC[0],
    STEADY_DURABLE[0],
    TRANSIENT_DURABLE[0],
];
const TRANSIENT: &[Envelope] = &[
    TRANSIENT_KERNEL[0],
    TRANSIENT_NATIVE[0],
    TRANSIENT_PUBLIC[0],
    TRANSIENT_DURABLE[0],
];
const PUBLIC: &[Envelope] = &[
    STEADY_PUBLIC[0],
    TRANSIENT_PUBLIC[0],
    STEADY_NATIVE[0],
    TRANSIENT_NATIVE[0],
    STEADY_DURABLE[0],
    TRANSIENT_DURABLE[0],
];
const ALL: &[Envelope] = &[
    STEADY_KERNEL[0],
    TRANSIENT_KERNEL[0],
    STEADY_NATIVE[0],
    TRANSIENT_NATIVE[0],
    STEADY_PUBLIC[0],
    TRANSIENT_PUBLIC[0],
    STEADY_DURABLE[0],
    TRANSIENT_DURABLE[0],
    envelope(
        STEADY_FORMULATION,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::EncryptedRemote,
    ),
    envelope(
        TRANSIENT_FORMULATION,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::EncryptedRemote,
    ),
    envelope(
        STEADY_FORMULATION,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::PartitionedCpu,
    ),
    envelope(
        TRANSIENT_FORMULATION,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::PartitionedCpu,
    ),
    envelope(
        STEADY_FORMULATION,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
    envelope(
        TRANSIENT_FORMULATION,
        MeshOrder::CanonicalTetrahedra,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
const TOPOLOGY_001: Requirement = Requirement {
    id: "FEA-FLUID-TOPOLOGY-001",
    title: "Canonical conservative finite-volume projection",
    stages: &[
        Stage::Fluid
    ],
    acceptance: concat!(
"Flow uses complete canonical tetrahedral control-volume geometry and exact ",
"cell/face authority. Meshes outside validated geometry and quality limits ",
"are rejected; no channel reduction substitutes for the original domain.",
),
    owners: &[
        "crates/runmat-analysis/fea/src/topology/finite_volume.rs"
    ],
    dependencies: &[
    ],
    obligations: &[
        obligation(
            "inventory",
            concat!(
"Retain canonical cell identities, complete reciprocal cell-face incidence, ",
"owner/neighbour, oriented area vectors, cell volumes/centroids, centroid ",
"distances, interpolation weights, non-orthogonality/skewness metrics and ",
"exact original boundary provenance.",
),
            EvidenceKind::PhysicalBalance,
            CORE
        ),
        obligation(
            "geometry",
            concat!(
"Projection is deterministic under reordered source access and placement; ",
"oriented surfaces satisfy the divergence theorem and exact source/region ",
"ownership. Cache identity binds the original mesh and projection policy.",
),
            EvidenceKind::ExecutionEquivalence,
            CORE
        ),
        obligation(
            "quality",
            concat!(
"Admit supported tetrahedral order/geometry and governed ",
"non-orthogonality/skewness limits before work. Reject inversion, ",
"degeneracy, unsupported curvature, nonmanifold/missing/ambiguous boundaries ",
"or mutated source geometry; never silently reduce the model.",
),
            EvidenceKind::RejectionInvariant,
            CORE
        ),
        obligation(
            "budget",
            concat!(
"Count construction inventory and checked scratch/traversal work before ",
"allocation; exact ceilings, overflow and cancellation govern topology work.",
),
            EvidenceKind::ResourceAdmission,
            CORE
        ),
    ],
    witnesses: &[
        Witness {
            id: "FEA-WITNESS-FLUID-FV-GEOMETRY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"topology::finite_volume::tests::exact_two_cell_geometry_is_conservative_and_",
"retains_original_boundary_mapping",
),
                source: "crates/runmat-analysis/fea/src/topology/finite_volume/tests.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "inventory",
                    envelopes: STEADY_KERNEL,
                    boundary: concat!(
"Two affine Tet4 cells; oriented surface closure, volume and original face ",
"mapping. No skew/curved mesh envelope proof.",
)
                }
            ],
        },
        Witness {
            id: "FEA-WITNESS-FLUID-FV-ORDER",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"topology::finite_volume::tests::stable_cell_face_physics_is_invariant_to_reo",
"rdered_source_access_indices",
),
                source: "crates/runmat-analysis/fea/src/topology/finite_volume/tests.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "geometry",
                    envelopes: STEADY_KERNEL,
                    boundary: "Two-cell source access reordering, not execution placement equivalence."
                }
            ],
        },
        Witness {
            id: "FEA-WITNESS-FLUID-FV-REJECT",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"topology::finite_volume::tests::rejects_nonmanifold_missing_boundary_inversi",
"on_curvature_and_mutated_record",
),
                source: "crates/runmat-analysis/fea/src/topology/finite_volume/tests.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "quality",
                    envelopes: STEADY_KERNEL,
                    boundary: "Selected malformed source cases only, not complete quality-envelope convergence."
                }
            ],
        },
        Witness {
            id: "FEA-WITNESS-FLUID-FV-BUDGET",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: "topology::finite_volume::tests::exact_admission_bounds_succeed_and_each_one_below_rejects",
                source: "crates/runmat-analysis/fea/src/topology/finite_volume/tests.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "budget",
                    envelopes: STEADY_KERNEL,
                    boundary: concat!(
"Exact finite-volume construction counts and one-below limits; requested byte",
"s are not RSS.",
)
                }
            ],
        },
    ],
};
const EQUATIONS_001: Requirement = Requirement {
    id: "FEA-FLUID-EQUATIONS-001",
    title: "Original incompressible laminar mass and momentum equations",
    stages: &[
        Stage::Fluid
    ],
    acceptance: concat!(
"The supported initial flow envelope is incompressible, Newtonian, laminar ",
"and single phase. Steady and transient operators conserve original mass and ",
"momentum, use documented interpolation/stabilization and governed ",
"skew/non-orthogonal corrections, and solve pressure-velocity coupling with ",
"admissible shared nonsymmetric algebra.",
),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/momentum.rs",
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/pressure.rs",
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/iterate.rs"
    ],
    dependencies: &[
        concat!(
"Shared canonical topology, exact material assignment and general sparse ",
"solver/resource authorities.",
)
    ],
    obligations: &[
        obligation(
            "flux",
            concat!(
"Evaluate one original oriented owner-face mass flux and conservative ",
"advective, viscous and pressure momentum actions with equal/opposite ",
"internal-face contributions. Document interpolation, stabilization and ",
"non-orthogonal/skewness corrections within validated mesh limits.",
),
            EvidenceKind::PhysicalBalance,
            CORE
        ),
        obligation(
            "coupling",
            concat!(
"Pressure-velocity coupling solves original incompressibility and momentum ",
"equations. Component gauges/nullspaces are explicit, invalid or ",
"incompatible references reject, and physical residual checks cannot be ",
"replaced by solver convergence. Incompressibility is part of the mixed ",
"fluid formulation, never a generic user-authored equality constraint.",
),
            EvidenceKind::PhysicalBalance,
            CORE
        ),
        obligation(
            "pressure_state",
            concat!(
"Keep original reference-relative pressure as governing state through ",
"iteration, recovery and cold execution. Absolute pressure is a projection; ",
"converting it back must not erase fluctuations. Bind pressure references ",
"and retain owner-face flux rather than reconstructing missing original ",
"state.",
),
            EvidenceKind::OriginalStateComparison,
            CORE
        ),
    ],
    witnesses: &[
        Witness {
            id: "FEA-WITNESS-FLUID-FLUX-AFFINE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"physics::cfd::canonical::tests::affine_skew_face_velocity_and_nonzero_convec",
"tion_are_consistent",
),
                source: "crates/runmat-analysis/fea/src/physics/cfd/canonical/tests.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "flux",
                    envelopes: STEADY_KERNEL,
                    boundary: concat!(
"Independent affine skew-face flux and integrated nonzero advection; no ",
"general nonlinear field-convergence credit.",
)
                }
            ],
        },
        Witness {
            id: "FEA-WITNESS-FLUID-GAUGE-REJECT",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"physics::cfd::canonical::tests::closed_reference::closed_gauge_admission_rej",
"ects_missing_duplicate_outlet_and_incompatible_flux",
),
                source: "crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/closed_reference.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "coupling",
                    envelopes: STEADY_KERNEL,
                    boundary: "Closed gauge/reference and net flux rejection; not full pressure-velocity envelope proof."
                }
            ],
        },
        Witness {
            id: "FEA-WITNESS-FLUID-PRESSURE-ORIGINAL",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"physics::cfd::canonical::tests::pressure_reference::original_fluctuation_sur",
"vives_absolute_rounding_fresh_solve_and_cold_recovery",
),
                source: "crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/pressure_reference.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "pressure_state",
                    envelopes: STEADY_KERNEL,
                    boundary: concat!(
"Fourier duct scaling and two gauges retain bitwise original ",
"velocity/fluctuation/flux with and without acceleration; cold JSON replay ",
"only.",
)
                }
            ],
        },
        Witness {
            id: "FEA-WITNESS-FLUID-PRESSURE-NATIVE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native",
                target: CargoTarget::Library,
                name: concat!(
"analysis_kernel::fluid_steady::tests::pressure_state::native_pressure_projec",
"tion_preserves_original_governing_state",
),
                source: concat!(
"crates/runmat-execution-runner-native/src/analysis_kernel/fluid_steady/tests",
"/pressure_state.rs",
)
            },
            claims: &[
                CoverageClaim {
                    obligation: "pressure_state",
                    envelopes: STEADY_NATIVE,
                    boundary: concat!(
"Native result/receipt preserves original governing fields under gauge ",
"shift; not distributed algebra or producer-death transient evidence.",
)
                }
            ],
        },
    ],
};
const INPUTS_001: Requirement = Requirement {
    id: "FEA-FLUID-INPUTS-001",
    title: "Exact typed fluid boundaries, sources and initial conditions",
    stages: &[Stage::Fluid],
    acceptance: concat!(
        "Fluid cell domains, exact boundary regions, material laws, constraints, ",
        "sources and initial fields are typed original authority. Shared ",
        "spatial/time signal realization preserves declared flow and momentum ",
        "resultants and rejects wrong-dimensional, absent, ambiguous, conflicting or ",
        "incompatible bindings.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/boundary.rs",
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/initial.rs",
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/forcing.rs",
    ],
    dependencies: &[concat!(
        "Shared typed materials, cell-domain realization, constraints, excitations ",
        "and initial-condition authority.",
    )],
    obligations: &[
        obligation(
            "boundaries",
            concat!(
                "Implement component velocity, tangential moving-wall, symmetry, periodic ",
                "and pressure-reference constraints plus inlet velocity, outlet pressure and ",
                "mass-flow boundaries on exact regions. Reject unsupported normal wall ",
                "motion unless a qualified ALE formulation governs it.",
            ),
            EvidenceKind::SemanticRealization,
            CORE,
        ),
        obligation(
            "signals",
            concat!(
                "Use shared typed signals for inlet velocity, pressure, mass flow, wall ",
                "motion and body acceleration. Realize space/time dependence conservatively, ",
                "preserving declared mass/momentum resultants over whole boundaries, cells ",
                "and partitions with explicit sign/frame conventions and event sidedness.",
            ),
            EvidenceKind::SemanticRealization,
            CORE,
        ),
        obligation(
            "initial",
            concat!(
                "Resolve model-owned reusable initial velocity and pressure/reference fields ",
                "on exact cell scope from one typed finite uniform value, exact ",
                "non-frequency signal sample with coordinate/discontinuity side, or ",
                "immutable complete field artifact. Check content, topology, quantity, shape ",
                "and explicit vector frame. Study binding selects the consuming step; ",
                "compatible upstream state takes precedence unless a typed replacement ",
                "policy applies, and conflicts reject. Validate face flux and ",
                "constraint/gauge consistency before transient execution; no fabricated ",
                "terminal profile or inferred extremal boundary authority.",
            ),
            EvidenceKind::SemanticRealization,
            TRANSIENT,
        ),
        obligation(
            "rejection",
            concat!(
                "Reject missing/multiply matched/wrong-dimension material, cell, source or ",
                "boundary authority, quantity conflicts, stale input and exceeded input ",
                "realization budgets before numerical execution.",
            ),
            EvidenceKind::RejectionInvariant,
            CORE,
        ),
    ],
    witnesses: &[
        Witness {
            id: "FEA-WITNESS-FLUID-MASS-INTEGRAL",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "physics::cfd::canonical::tests::mass_flow::signed_mass_sources_preserve_exac",
                    "t_faces_partitions_and_cold_original_fields",
                ),
                source: "crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/mass_flow.rs",
            },
            claims: &[CoverageClaim {
                obligation: "signals",
                envelopes: STEADY_KERNEL,
                boundary: concat!(
                    "Signed rates across rotated exact faces; three modulo-index pieces ",
                    "demonstrate additive realization only, not distributed cell assembly.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-FLUID-BODY-TIME",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "physics::cfd::canonical::tests::time_signals::body_force_time_sampling_prese",
                    "rves_independent_volume_resultant",
                ),
                source:
                    "crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/time_signals.rs",
            },
            claims: &[CoverageClaim {
                obligation: "signals",
                envelopes: TRANSIENT_KERNEL,
                boundary:
                    "Independent volume-integrated time sampling; no partitioned algebra credit.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-FLUID-INPUT-REJECT",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "physics::cfd::canonical::tests::mass_flow::source_authority_conflicts_quanti",
                    "ty_and_budget_reject_before_solve",
                ),
                source: "crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/mass_flow.rs",
            },
            claims: &[CoverageClaim {
                obligation: "rejection",
                envelopes: STEADY_KERNEL,
                boundary:
                    "Selected duplicate/wrong-quantity/source authority and budget rejection only.",
            }],
        },
    ],
};
const TRANSIENT_001: Requirement = Requirement {
    id: "FEA-FLUID-TRANSIENT-001",
    title: "Physical transient integration and original cold chronology",
    stages: &[
        Stage::Fluid
    ],
    acceptance: concat!(
"Transient flow integrates original fluid inertia and pressure-velocity ",
"evolution over actual intervals. Immutable original cell velocity, pressure ",
"reference/fluctuation and owner-face flux govern cold continuation; ",
"schedules, events, work limits and outputs retain their original chronology.",
),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/transient.rs",
        "crates/runmat-analysis/fea/src/execution/fluid/time_block.rs"
    ],
    dependencies: &[
        concat!(
"Mandatory independent transient accuracy, admitted task resources and ",
"compatible complete state publication.",
)
    ],
    obligations: &[
        obligation(
            "inertia",
            concat!(
"Integrate actual momentum time derivatives rather than return a terminal ",
"steady profile; initial state and step interval bind the original inertial ",
"operator.",
),
            EvidenceKind::IndependentReference,
            TRANSIENT
        ),
        obligation(
            "convergence",
            concat!(
"Show temporal convergence independently of spatial error on a nontrivial ",
"transient reference, including a transient vortex; report field error and ",
"conservative balance.",
),
            EvidenceKind::TemporalConvergence,
            TRANSIENT
        ),
        obligation(
            "history",
            concat!(
"Preserve complete original interval state, scheduled observations, ",
"one-sided discontinuities, predecessor order and cumulative admitted work. ",
"Reject reordered/substituted state, invalid intervals, budgets and ",
"cancellation without accepting a suffix.",
),
            EvidenceKind::ColdReplay,
            TRANSIENT
        ),
        obligation(
            "producer_exit",
            concat!(
"After actual native producer termination a fresh consumer resumes original ",
"transient fields and outputs without producer memory; split and ",
"uninterrupted histories agree within the governed accuracy envelope.",
),
            EvidenceKind::ProcessLifecycle,
            TRANSIENT_NATIVE
        ),
    ],
    witnesses: &[
        Witness {
            id: "FEA-WITNESS-FLUID-INERTIA",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"physics::cfd::canonical::tests::transient::original_mass_term_matches_indepe",
"ndent_cell_integral_on_every_axis",
),
                source: "crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/transient.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "inertia",
                    envelopes: TRANSIENT_KERNEL,
                    boundary: concat!(
"Independent rho*volume*delta-U/dt cell action on every axis; not temporal fi",
"eld convergence.",
)
                }
            ],
        },
        Witness {
            id: "FEA-WITNESS-FLUID-TIME-CONVERGENCE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"physics::cfd::canonical::tests::transient::exponential_translation_pressure_",
"shows_first_order_time_convergence",
),
                source: "crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/transient.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "convergence",
                    envelopes: TRANSIENT_KERNEL,
                    boundary: concat!(
"Uniform exponential translation pressure on three time steps; no vortex or ",
"independent spatial sweep.",
)
                }
            ],
        },
        Witness {
            id: "FEA-WITNESS-FLUID-COLD-BLOCK",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"physics::cfd::canonical::tests::time_block::authored_time_block_cold_recover",
"y_and_split_match_original_acceleration",
),
                source: "crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/time_block.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "history",
                    envelopes: TRANSIENT_KERNEL,
                    boundary: concat!(
"Original acceleration history and split replay; ",
"chronology/tamper/budget/cancel checks, no native process credit.",
)
                }
            ],
        },
        Witness {
            id: "FEA-WITNESS-FLUID-TIME-PROCESS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native",
                target: CargoTarget::Library,
                name: concat!(
"analysis_kernel::fluid_transient::tests::terminated_native_producers_preserv",
"e_original_history_and_admission_rejects_substitution",
),
                source: "crates/runmat-execution-runner-native/src/analysis_kernel/fluid_transient/tests.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "producer_exit",
                    envelopes: TRANSIENT_NATIVE,
                    boundary: concat!(
"Child actually exits before new consumers resume; uniform translation and ",
"substitution/budget rejection only.",
)
                }
            ],
        },
    ],
};
const RECOVERY_001: Requirement = Requirement {
    id: "FEA-FLUID-RECOVERY-001",
    title: "Original fluid fields and physical conservation recovery",
    stages: &[
        Stage::Fluid
    ],
    acceptance: concat!(
"Recover original cell and face pressure, velocity, flux, stress and wall ",
"quantities consistently, retaining exact boundary ownership and signed ",
"mass/momentum balance. Acceptance includes original equation residual, ",
"continuity, constitutive face-flux closure and global/per-cell/per-boundary ",
"physical balance.",
),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/recovery.rs",
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/boundary_balances.rs"
    ],
    dependencies: &[
    ],
    obligations: &[
        obligation(
            "fields",
            concat!(
"Retain original cell fields and owner-face flux; recover face traces, ",
"stresses and wall quantities with declared location, units, frames and sign ",
"conventions. Cold consumers validate original source and reject changed ",
"fields/tractions rather than silently re-solving them.",
),
            EvidenceKind::OriginalStateComparison,
            CORE
        ),
        obligation(
            "balance",
            concat!(
"Report global, per-cell and per-boundary signed mass and momentum ",
"components from original forces/fluxes with physical normalization. ",
"Algebraic residual convergence alone cannot pass physical acceptance.",
),
            EvidenceKind::PhysicalBalance,
            CORE
        ),
    ],
    witnesses: &[
        Witness {
            id: "FEA-WITNESS-FLUID-BOUNDARY-BALANCE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"physics::cfd::canonical::tests::boundary_balances::grouped_original_translat",
"ion_reports_signed_mass_and_pressure_forces",
),
                source: "crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/boundary_balances.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "balance",
                    envelopes: STEADY_KERNEL,
                    boundary: concat!(
"Independent constant translation exact group signs/pressure forces; no ",
"general wall-shear convergence.",
)
                }
            ],
        },
        Witness {
            id: "FEA-WITNESS-FLUID-WALL-SHEAR",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"physics::cfd::canonical::tests::boundary_balances::original_couette_boundary",
"_shear_and_cold_attribution_are_preserved",
),
                source: "crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/boundary_balances.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "fields",
                    envelopes: STEADY_KERNEL,
                    boundary: concat!(
"Couette boundary shear/source attribution preserved under cold recovery; one",
" fixture only.",
)
                }
            ],
        },
        Witness {
            id: "FEA-WITNESS-FLUID-FIELD-REJECT",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"execution::fluid::tests::original_velocity_pressure_flux_traction_and_resour",
"ce_substitution_reject",
),
                source: "crates/runmat-analysis/fea/src/execution/fluid/tests.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "fields",
                    envelopes: STEADY_KERNEL,
                    boundary: "Portable original field/traction/resource substitution rejection; no process termination."
                }
            ],
        },
    ],
};
const ACCURACY_001: Requirement = Requirement {
    id: "FEA-FLUID-ACCURACY-001",
    title: "Independent flow benchmark and convergence acceptance",
    stages: &[Stage::Fluid],
    acceptance: concat!(
        "Production flow requires independent Poiseuille, Couette, lid-driven ",
        "cavity, backward-facing-step references within the declared Reynolds range, ",
        "a transient vortex, spatial/time convergence and conservation. Every ",
        "mandatory accuracy criterion must be satisfied by applicable current ",
        "evidence before publication; a missing independent discretization bound ",
        "remains open.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/cfd/canonical/qualification.rs",
        "crates/runmat-analysis/fea/src/results/accuracy/evaluation.rs",
    ],
    dependencies: &[concat!(
        "References are independent of the operator under test; original meshes, ",
        "tolerances and caps remain mandatory.",
    )],
    obligations: &[
        obligation(
            "steady_references",
            concat!(
                "Validate Poiseuille and Couette plus lid-driven cavity and backward-facing ",
                "step across the documented Reynolds range, with velocity/pressure/flow/wall ",
                "quantities and conservative closure.",
            ),
            EvidenceKind::IndependentReference,
            STEADY_KERNEL,
        ),
        obligation(
            "spatial",
            concat!(
                "Demonstrate spatial convergence and absolute error for non-affine flow, ",
                "pressure, integrated flow and wall stress within the documented ",
                "quality/Reynolds envelope; affine exactness alone does not establish ",
                "convergence.",
            ),
            EvidenceKind::SpatialConvergence,
            STEADY_KERNEL,
        ),
        obligation(
            "acceptance",
            concat!(
                "Shared accuracy assessment consumes original result, receipts and ",
                "applicable independent spatial/time evidence. Missing, failed or ",
                "indeterminate mandatory criteria block completed semantic publication; ",
                "imported provenance cannot create numerical proof.",
            ),
            EvidenceKind::AccuracyAcceptance,
            PUBLIC,
        ),
    ],
    witnesses: &[
        Witness {
            id: "FEA-WITNESS-FLUID-COUETTE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "physics::cfd::canonical::tests::affine_couette_uses_original_geometry_and_ba",
                    "lanced_physical_fields",
                ),
                source: "crates/runmat-analysis/fea/src/physics/cfd/canonical/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "steady_references",
                envelopes: STEADY_KERNEL,
                boundary: concat!(
                    "Independent affine Couette velocity, wall shear and original balance; not ",
                    "cavity/backward-facing step.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-FLUID-DUCT-CONVERGENCE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "physics::cfd::canonical::tests::rectangular_duct_compares_fourier_reference_",
                    "with_controlled_tail",
                ),
                source: "crates/runmat-analysis/fea/src/physics/cfd/canonical/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "spatial",
                envelopes: STEADY_KERNEL,
                boundary: concat!(
                    "Controlled-tail non-affine Fourier duct: velocity, flow and wall shear ",
                    "refinement; not pressure/cavity/backward-facing-step envelope.",
                ),
            }],
        },
    ],
};
const EXECUTION_001: Requirement = Requirement {
    id: "FEA-FLUID-EXECUTION-001",
    title: "Governed fluid execution and semantic publication",
    stages: &[Stage::Fluid],
    acceptance: concat!(
        "Fluid execution retains original equations and fields across supported ",
        "placement, admitted resources and immutable publication. Serial correctness ",
        "precedes partitioned cell assembly and distributed linear algebra. Every ",
        "remote, distributed or accelerator claim requires direct applicable ",
        "execution and equivalence evidence; host fallback or fanout is not that ",
        "evidence.",
    ),
    owners: &[
        "crates/runmat-analysis/fea/src/execution/fluid/mod.rs",
        "crates/runmat-analysis/execution/src/fluid_study.rs",
        "crates/runmat-execution-runner-native/src/analysis_kernel/fluid_steady.rs",
    ],
    dependencies: &[
        concat!(
            "Shared immutable artifacts, task admission, state/observation/accuracy ",
            "authorities and attested release/documentation.",
        ),
        concat!(
            "Conjugate heat transfer requires separately qualified thermal and fluid ",
            "operators and conservative exact-interface closure; fluid evidence alone ",
            "cannot qualify the coupled system.",
        ),
    ],
    obligations: &[
        obligation(
            "resources",
            concat!(
                "Admit original projection, source/initial realization, pressure/momentum ",
                "operators, solver work, retained histories/outputs and transport before ",
                "execution; cancellation and exact ceilings reject safely. Measured ",
                "whole-process/backend usage must support claimed resource envelopes; ",
                "requested bytes are not RSS.",
            ),
            EvidenceKind::ResourceAdmission,
            ALL,
        ),
        obligation(
            "placement",
            concat!(
                "Implement partitioned cell assembly after verified serial equations, then ",
                "distributed linear algebra. Preserve original boundary/source flow and ",
                "momentum resultants and numerical fields across ",
                "serial/native/durable/encrypted-remote/partitioned/claimed-accelerator ",
                "execution with supported precision and authenticated lifecycle.",
            ),
            EvidenceKind::ExecutionEquivalence,
            ALL,
        ),
        obligation(
            "publication",
            concat!(
                "Publish original complete state, observations, receipts and mandatory ",
                "accuracy evidence atomically under exact source/phase lineage. Failed or ",
                "partial imports cannot install completed authority. Include ",
                "supported-envelope documentation and reproducible release evidence; no ",
                "implicit channel fallback.",
            ),
            EvidenceKind::PublicationIntegrity,
            PUBLIC,
        ),
    ],
    witnesses: &[
        Witness {
            id: "FEA-WITNESS-FLUID-RESOURCE-CEILING",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "execution::fluid::tests::authored_resource_ceiling_and_original_wire_profile",
                    "_are_enforced_before_expansion",
                ),
                source: "crates/runmat-analysis/fea/src/execution/fluid/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "resources",
                envelopes: STEADY_KERNEL,
                boundary: concat!(
                    "Original payload and authored requested ceilings before expansion; no ",
                    "measured RSS/GPU/distributed bound.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-FLUID-COLD-PUBLICATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native",
                target: CargoTarget::Library,
                name: concat!(
                    "analysis_kernel::fluid_steady::tests::chain::publication::original_native_fl",
                    "uid_observation_complete_state_and_accuracy_cold_chain",
                ),
                source: concat!(
                    "crates/runmat-execution-runner-native/src/analysis_kernel/fluid_steady/tests",
                    "/chain/publication.rs",
                ),
            },
            claims: &[CoverageClaim {
                obligation: "publication",
                envelopes: STEADY_DURABLE,
                boundary: concat!(
                    "Native durable complete-state/observation chain rejects missing mandatory ",
                    "spatial evidence; not a qualified convergence certificate.",
                ),
            }],
        },
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    TOPOLOGY_001,
    EQUATIONS_001,
    INPUTS_001,
    TRANSIENT_001,
    RECOVERY_001,
    ACCURACY_001,
    EXECUTION_001,
];
