//! Standing requirements for complete accepted nonlinear state and cold continuation.
//! State preservation does not close constitutive accuracy, adaptation or publication.
use crate::requirement::*;
const J2: &[Envelope] = &[
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuPreparedController,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuPreparedController,
    ),
];
const CONTACT: &[Envelope] = &[
    envelope(
        Formulation::SmallStrainJ2PenaltyCoulombContact,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuPreparedController,
    ),
    envelope(
        Formulation::SmallStrainJ2PenaltyCoulombContact,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuPreparedController,
    ),
];
const J2_NATIVE: &[Envelope] = &[
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNativeProcess,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNativeProcess,
    ),
];
const PREPARED: &[Envelope] = &[J2[0], J2[1], CONTACT[0], CONTACT[1]];
const NATIVE: &[Envelope] = &[
    J2_NATIVE[0],
    J2_NATIVE[1],
    envelope(
        Formulation::SmallStrainJ2PenaltyCoulombContact,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNativeProcess,
    ),
    envelope(
        Formulation::SmallStrainJ2PenaltyCoulombContact,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNativeProcess,
    ),
];
pub(crate) const ACCEPTED_HISTORY: Requirement = Requirement {
    id: "FEA-NONLINEAR-STATE-001",
    title: "Complete original accepted nonlinear history through cold continuation",
    stages: &[
        Stage::NonlinearContact
    ],
    acceptance: concat!(
"Preserve complete original material/contact accepted history in immutable ",
"canonical checkpoint artifacts. Restore cold initial outputs, validate ",
"receiving-state physical compatibility, retain controller chronology and ",
"survive native producer termination within each declared envelope. Element ",
"averages and digest-only checks cannot replace original point-state ",
"authority. Checkpoints and incomplete fields must not be presented as ",
"completed converged base states.",
),
    owners: &[
        "crates/runmat-analysis/fea/src/solve/nonlinear/solid.rs",
        "crates/runmat-analysis/fea/src/solve/nonlinear/driver/continuation.rs",
        "crates/runmat-execution-runner-native/src/analysis_kernel/nonlinear_block.rs",
    ],
    dependencies: &[
        concat!(
"Independent J2/contact constitutive, equilibrium and mesh/load/penalty ",
"accuracy must satisfy their mandatory acceptance criteria.",
),
        concat!(
"Whole task resource admission, supported public contact controls and ",
"mandatory accuracy-gated semantic publication.",
),
        concat!(
"Supported feature/initial transitions and separately qualified adaptation ",
"transfer; no transfer coverage inferred here.",
),
        concat!(
"Typed state-transition manifests retain producer-step, topology/partition, ",
"field/history, reference, constraint, precision and completeness lineage; ",
"dependent steps are admitted only after immutable compatible state ",
"publication. Activation, retry, restart and branching retain distinct ",
"meanings and identities.",
),
    ],
    obligations: &[
        obligation(
            "points",
            concat!(
"Persist every original ordered integration-point constitutive state and ",
"complete contact active/friction history in immutable canonical artifacts, ",
"with accepted displacement, load parameter and cumulative physical energy ",
"state. Equilibrium and recovery consume that same accepted point history; ",
"element averages cannot replace restart authority.",
),
            EvidenceKind::OriginalStateComparison,
            PREPARED
        ),
        obligation(
            "rollback",
            concat!(
"Rejected Newton/line-search/cutback trials and cancellation must not mutate ",
"accepted material/contact history or publish an accepted suffix.",
),
            EvidenceKind::RejectionInvariant,
            PREPARED
        ),
        obligation(
            "initial_outputs",
            concat!(
"A cold block restores initial plastic/contact outputs from its original ",
"accepted point history rather than zeroing or reinitializing it.",
),
            EvidenceKind::ColdReplay,
            PREPARED
        ),
        obligation(
            "cold_validation",
            concat!(
"Reject corrupted, reordered, missing, incompatible or rehashed substituted ",
"scope/history before accepting a cold continuation. Validate the receiving ",
"formulation, topology, integration inventory, properties, constraints, ",
"reference conventions, point ordering, accepted load parameter and physical ",
"state; digests alone are insufficient.",
),
            EvidenceKind::RejectionInvariant,
            PREPARED
        ),
        obligation(
            "chronology",
            concat!(
"Retain original scheduled target and cutbacks, cumulative failed/replay ",
"work and caps, accepted/output ordinals and predecessor order across cold ",
"blocks.",
),
            EvidenceKind::ControllerChronology,
            PREPARED
        ),
        obligation(
            "producer_exit",
            concat!(
"A new native consumer after producer termination resumes original complete ",
"history and outputs for homogeneous and nonuniform supported bodies, ",
"without the producer memory.",
),
            EvidenceKind::ProcessLifecycle,
            NATIVE
        ),
    ],
    witnesses: &[
        Witness {
            id: "FEA-WITNESS-J2-BLOCKS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"solve::nonlinear::driver::tests::blocks::cold_blocks_preserve_original_point",
"_history_schedule_and_physical_outputs",
),
                source: "crates/runmat-analysis/fea/src/solve/nonlinear/driver/tests/blocks.rs",
            },
            claims: &[
                CoverageClaim {
                    obligation: "points",
                    envelopes: J2,
                    boundary: "Homogeneous core cold blocks; original histories and outputs, not continuum accuracy."
                },
                CoverageClaim {
                    obligation: "initial_outputs",
                    envelopes: J2,
                    boundary: "Homogeneous core cold blocks; original histories and outputs, not continuum accuracy."
                },
                CoverageClaim {
                    obligation: "chronology",
                    envelopes: J2,
                    boundary: "Homogeneous core cold blocks; original histories and outputs, not continuum accuracy."
                },
            ],
        },
        Witness {
            id: "FEA-WITNESS-J2-TAMPER",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"solve::nonlinear::driver::tests::blocks::altered_cold_controller_and_point_s",
"tate_reject_without_mutating_original",
),
                source: "crates/runmat-analysis/fea/src/solve/nonlinear/driver/tests/blocks.rs",
            },
            claims: &[
                CoverageClaim {
                    obligation: "cold_validation",
                    envelopes: &[
                        J2[
                            1
                        ]
                    ],
                    boundary: "Altered controller and original point state reject; no independent past-trajectory proof."
                },
            ],
        },
        Witness {
            id: "FEA-WITNESS-J2-YIELDED-TAMPER",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"solve::nonlinear::driver::tests::blocks::yielded_substitution::rehashed_cold",
"_yielded_history_requires_physical_validation",
),
                source: concat!(
"crates/runmat-analysis/fea/src/solve/nonlinear/driver/tests/blocks/yielded_s",
"ubstitution.rs",
),
            },
            claims: &[
                CoverageClaim {
                    obligation: "cold_validation",
                    envelopes: J2,
                    boundary: concat!(
"Rehashed yielded history substitution requires physical validation, beyond c",
"ontent hashes.",
)
                },
            ],
        },
        Witness {
            id: "FEA-WITNESS-J2-ROLLBACK",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: "solve::nonlinear::solid::tests::failed_newton_and_cutback_do_not_commit_caller_history",
                source: "crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests.rs",
            },
            claims: &[
                CoverageClaim {
                    obligation: "rollback",
                    envelopes: &[
                        J2[
                            0
                        ]
                    ],
                    boundary: "Caller history remains unchanged after failed Newton/cutback."
                },
            ],
        },
        Witness {
            id: "FEA-WITNESS-J2-CANCEL",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"solve::nonlinear::driver::tests::blocks::cancelled_continuation_does_not_mut",
"ate_or_publish_an_accepted_suffix",
),
                source: "crates/runmat-analysis/fea/src/solve/nonlinear/driver/tests/blocks.rs",
            },
            claims: &[
                CoverageClaim {
                    obligation: "rollback",
                    envelopes: &[
                        J2[
                            0
                        ]
                    ],
                    boundary: "Cancelled block preserves original prefix and publishes no accepted suffix."
                },
            ],
        },
        Witness {
            id: "FEA-WITNESS-J2-POINT-CANCEL",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"solve::nonlinear::solid::tests::cancelled_point_evaluation_reaches_newton_wi",
"thout_mutating_history",
),
                source: "crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests.rs",
            },
            claims: &[
                CoverageClaim {
                    obligation: "rollback",
                    envelopes: J2,
                    boundary: concat!(
"Both-order point-evaluation cancellation preserves caller history; not ",
"complete failed-Newton, line-search or controller suffix coverage.",
)
                },
            ],
        },
        Witness {
            id: "FEA-WITNESS-J2-CAP",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"solve::nonlinear::driver::tests::blocks::one_increment_blocks_count_replay_c",
"ost_without_resetting_global_limits",
),
                source: "crates/runmat-analysis/fea/src/solve/nonlinear/driver/tests/blocks.rs",
            },
            claims: &[
                CoverageClaim {
                    obligation: "chronology",
                    envelopes: &[
                        J2[
                            0
                        ]
                    ],
                    boundary: "One-increment cold replay costs count toward original global limits."
                },
            ],
        },
        Witness {
            id: "FEA-WITNESS-J2-CUTBACK",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"solve::nonlinear::driver::tests::blocks::cutback_boundary_retains_original_t",
"arget_and_failed_work_across_cold_blocks",
),
                source: "crates/runmat-analysis/fea/src/solve/nonlinear/driver/tests/blocks.rs",
            },
            claims: &[
                CoverageClaim {
                    obligation: "chronology",
                    envelopes: J2,
                    boundary: "Original cutback target and failed work survive cold blocks."
                },
            ],
        },
        Witness {
            id: "FEA-WITNESS-CONTACT-BLOCKS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"solve::nonlinear::solid::tests::contact::blocks::full_authored_contact_cold_",
"blocks_preserve_schedule_original_points_and_work",
),
                source: "crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests/contact/blocks.rs",
            },
            claims: &[
                CoverageClaim {
                    obligation: "points",
                    envelopes: CONTACT,
                    boundary: concat!(
"Prepared constrained affine-body fixture; trusted test loads. No ",
"public/native contact or generic contact convergence credit.",
)
                },
                CoverageClaim {
                    obligation: "initial_outputs",
                    envelopes: CONTACT,
                    boundary: concat!(
"Prepared constrained affine-body fixture; trusted test loads. No ",
"public/native contact or generic contact convergence credit.",
)
                },
                CoverageClaim {
                    obligation: "chronology",
                    envelopes: CONTACT,
                    boundary: concat!(
"Prepared constrained affine-body fixture; trusted test loads. No ",
"public/native contact or generic contact convergence credit.",
)
                },
            ],
        },
        Witness {
            id: "FEA-WITNESS-CONTACT-COLD",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"solve::nonlinear::solid::tests::contact::cold::combined_scope_history_and_ge",
"neral_admission_reject_substitution",
),
                source: "crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests/contact/cold.rs",
            },
            claims: &[
                CoverageClaim {
                    obligation: "cold_validation",
                    envelopes: &[
                        CONTACT[
                            1
                        ]
                    ],
                    boundary: "Combined scope/material/contact history and GENERAL admission substitution reject."
                },
            ],
        },
        Witness {
            id: "FEA-WITNESS-CONTACT-ENERGY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"solve::nonlinear::solid::tests::contact::energy::accepted_energy_rejects_mis",
"sing_negative_foreign_and_changed_point_work",
),
                source: "crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests/contact/energy.rs",
            },
            claims: &[
                CoverageClaim {
                    obligation: "cold_validation",
                    envelopes: CONTACT,
                    boundary: concat!(
"Accepted contact energy/current point work validation; not proof of past ",
"numerical-loss trajectory.",
)
                },
            ],
        },
        Witness {
            id: "FEA-WITNESS-CONTACT-ROLLBACK",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"solve::nonlinear::solid::tests::contact::rollback::rejected_coupled_newton_a",
"nd_cancellation_preserve_complete_original_histories",
),
                source: "crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests/contact/rollback.rs",
            },
            claims: &[
                CoverageClaim {
                    obligation: "rollback",
                    envelopes: CONTACT,
                    boundary: "Rejected/cancelled coupled trials preserve complete original history."
                },
            ],
        },
        Witness {
            id: "FEA-WITNESS-J2-PROCESS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native",
                target: CargoTarget::Library,
                name: concat!(
"analysis_kernel::nonlinear_block::tests::process::terminated_producers_resum",
"e_original_history",
),
                source: concat!(
"crates/runmat-execution-runner-native/src/analysis_kernel/nonlinear_block/te",
"sts/process.rs",
),
            },
            claims: &[
                CoverageClaim {
                    obligation: "producer_exit",
                    envelopes: J2_NATIVE,
                    boundary: concat!(
"Native producer actually exits before cold consumers resume; homogeneous unc",
"oupled J2 only.",
)
                },
            ],
        },
        Witness {
            id: "FEA-WITNESS-J2-LAYERED-PROCESS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native",
                target: CargoTarget::Library,
                name: concat!(
"analysis_kernel::nonlinear_block::tests::layered::terminated_native_blocks_p",
"reserve_nonuniform_material_history",
),
                source: concat!(
"crates/runmat-execution-runner-native/src/analysis_kernel/nonlinear_block/te",
"sts/layered.rs",
),
            },
            claims: &[
                CoverageClaim {
                    obligation: "producer_exit",
                    envelopes: J2_NATIVE,
                    boundary: concat!(
"Native exited-producer aligned nonuniform J2 reference; no uniform ",
"moving-front accuracy or contact credit.",
)
                },
            ],
        },
    ],
};
const NONLINEAR_DOMAIN: &[Envelope] = &[
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::SerialCpuNativeProcess,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::EncryptedRemote,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PartitionedCpu,
    ),
    envelope(
        Formulation::NonlinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
const J2_OPERATOR_DOMAIN: &[Envelope] = &[
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::SerialCpuNativeProcess,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::EncryptedRemote,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PartitionedCpu,
    ),
    envelope(
        Formulation::SmallStrainJ2,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
const CONTACT_OPERATOR_DOMAIN: &[Envelope] = &[
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::SerialCpuNativeProcess,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::EncryptedRemote,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PartitionedCpu,
    ),
    envelope(
        Formulation::PenaltyCoulombContact,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
const FEA_NONLINEAR_OPERATOR_001: Requirement = Requirement {
    id: "FEA-NONLINEAR-OPERATOR-001",
    title: "State-dependent nonlinear mechanics and supported plastic laws",
    stages: &[Stage::NonlinearContact],
    acceptance: concat!(
        "State-dependent nonlinear mechanics and supported plastic laws must satisfy ",
        "all applicable supported-envelope criteria. Independent physical, semantic, ",
        "accuracy, resource and publication evidence are separate; missing, failed ",
        "or indeterminate mandatory evidence prevents production acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/solve/nonlinear/solid.rs"],
    obligations: &[
        obligation(
            "kinematics_tangent_newton",
            concat!(
                "Define nonlinear kinematics supported initially and reject unsupported ",
                "combinations. Assemble internal force and consistent or documented ",
                "algorithmic tangent from current state. Implement Newton iteration with ",
                "real residual and increment criteria, line search, and load/time stepping.",
            ),
            EvidenceKind::IndependentReference,
            J2_OPERATOR_DOMAIN,
        ),
        obligation(
            "point_state_constitutive_laws",
            concat!(
                "Store integration-point constitutive state. Implement return mapping and ",
                "tangent for each explicitly supported plastic law. Feed plastic state into ",
                "equilibrium and recovery.",
            ),
            EvidenceKind::IndependentReference,
            J2_OPERATOR_DOMAIN,
        ),
        obligation(
            "canonical_state_physical_cutbacks",
            concat!(
                "Persist state in canonical artifacts suitable for restart and adaptation ",
                "transfer. Add cutback/retry policies driven by physical convergence rather ",
                "than synthetic scalar damping.",
            ),
            EvidenceKind::IndependentReference,
            J2_OPERATOR_DOMAIN,
        ),
        obligation(
            "references",
            concat!(
                "Evidence includes constitutive point tests, elastoplastic bars, cyclic ",
                "paths where supported, large-displacement beams, limit-point cases only ",
                "when the continuation method supports them, and mesh/load-step convergence.",
            ),
            EvidenceKind::IndependentReference,
            J2_OPERATOR_DOMAIN,
        ),
    ],
    witnesses: &[Witness {
        id: "FEA-WITNESS-J2-CONTINUUM-TET4",
        locator: WitnessLocator::CargoTest {
            package: "runmat-analysis-fea",
            target: CargoTarget::Library,
            name: concat!(
                "solve::nonlinear::solid::continuum_qualification::manufactured_fully_yielded",
                "_cube_refines_and_checks_load_step_sensitivity",
            ),
            source:
                "crates/runmat-analysis/fea/src/solve/nonlinear/solid/continuum_qualification.rs",
        },
        claims: &[CoverageClaim {
            obligation: "references",
            envelopes: &[J2_OPERATOR_DOMAIN[0]],
            boundary: concat!(
                "Original uniform Tet4 continuum gate asserts ",
                "displacement/stress/point-history/energy/reaction refinement and load ",
                "schedule sensitivity; linked executable is unassessed here and recorded ",
                "scientific failures remain failures.",
            ),
        }],
    }],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
const FEA_CONTACT_OPERATOR_001: Requirement = Requirement {
    id: "FEA-CONTACT-OPERATOR-001",
    title: "Authored surface contact, search, friction and coupled inequality equilibrium",
    stages: &[
        Stage::NonlinearContact
    ],
    acceptance: concat!(
"Authored surface contact, search, friction and coupled inequality ",
"equilibrium must satisfy all applicable supported-envelope criteria. ",
"Independent physical, semantic, accuracy, resource and publication evidence ",
"are separate; missing, failed or indeterminate mandatory evidence prevents ",
"production acceptance.",
),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/structural/contact/mod.rs"
    ],
    obligations: &[
        obligation(
            "surface_search_gap",
            concat!(
"Resolve authored primary and secondary surface identities through the ",
"shared topology projection. Build deterministic broad-phase and ",
"narrow-phase search. Compute oriented normals, closest projection, gap, ",
"quadrature, and active-set state.",
),
            EvidenceKind::IndependentReference,
            CONTACT_OPERATOR_DOMAIN
        ),
        obligation(
            "normal_friction_equilibrium",
            concat!(
"Implement a documented normal enforcement method and friction law. Assemble ",
"contact residual and tangent into nonlinear equilibrium. Update contact ",
"state as geometry changes.",
),
            EvidenceKind::IndependentReference,
            CONTACT_OPERATOR_DOMAIN
        ),
        obligation(
            "history_search_diagnostics",
            concat!(
"Preserve contact history across restart and supported adaptation. Diagnose ",
"initial penetration, unmatched surfaces, normal ambiguity, search ",
"exhaustion, and parameter sensitivity. The contact system owns ",
"deterministic surface pairing and search updates.",
),
            EvidenceKind::IndependentReference,
            CONTACT_OPERATOR_DOMAIN
        ),
        obligation(
            "gap_active_enforcement",
            concat!(
"The contact system owns gap and normal definitions, active-set or ",
"complementarity state, and normal enforcement.",
),
            EvidenceKind::IndependentReference,
            CONTACT_OPERATOR_DOMAIN
        ),
        obligation(
            "friction_tangent_history",
            concat!(
"The contact system owns tangential/friction state, residual and tangent ",
"contributions, and history/restart data.",
),
            EvidenceKind::IndependentReference,
            CONTACT_OPERATOR_DOMAIN
        ),
        obligation(
            "complementarity_penetration",
            "The contact system owns complementarity and penetration evidence.",
            EvidenceKind::IndependentReference,
            CONTACT_OPERATOR_DOMAIN
        ) ,
        obligation(
            "references",
            concat!(
"Evidence includes gap closure, patch contact, Hertz-type normal contact, ",
"sliding/friction, objectivity, action-reaction balance, and ",
"mesh/contact-parameter convergence.",
),
            EvidenceKind::IndependentReference,
            CONTACT_OPERATOR_DOMAIN
        ) ,
        obligation(
            "inequality",
            concat!(
"Authored contact interfaces compile to nonlinear inequalities, never the ",
"equality matrix. Diagnose tied/contacting surface overlap without erasing ",
"their mathematical distinction. Stable authored identities survive changing ",
"geometry while normals, pairings and active state update in the owning ",
"nonlinear formulation.",
),
            EvidenceKind::RejectionInvariant,
            CONTACT_OPERATOR_DOMAIN
        )
    ],
    witnesses: &[
        Witness {
            id: "FEA-WITNESS-CONTACT-OBJECTIVITY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"physics::structural::contact::penalty::tests::objectivity::material_attached",
"_frame_and_point_traction_are_objective_under_proper_rotation",
),
                source: "crates/runmat-analysis/fea/src/physics/structural/contact/penalty/tests/objectivity.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "references",
                    envelopes: &[
                        CONTACT_OPERATOR_DOMAIN[
                            0
                        ]
                    ],
                    boundary: concat!(
"Tri3 contact frame and penalty friction point-law proper-rotation ",
"objectivity; no global Hertz, sliding body, action-reaction or mesh/penalty ",
"convergence credit.",
)
                }
            ]
        }
    ],
    dependencies: &[
        concat!(
"Shared typed authoring, canonical topology/material/constraint/excitation ",
"authorities; governed resources and execution; independently qualified ",
"accuracy and semantic publication.",
)
    ]
};
const FEA_NONLINEAR_GOVERNANCE_001: Requirement = Requirement {
    id: "FEA-NONLINEAR-GOVERNANCE-001",
    title: "Converged nonlinear states, supported laws and independent qualification",
    stages: &[Stage::NonlinearContact],
    acceptance: concat!(
        "Converged nonlinear states, supported laws and independent qualification ",
        "must satisfy all applicable supported-envelope criteria. Independent ",
        "physical, semantic, accuracy, resource and publication evidence are ",
        "separate; missing, failed or indeterminate mandatory evidence prevents ",
        "production acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/pipeline/nonlinear.rs"],
    obligations: &[
        obligation(
            "operator_constitutive_contact",
            concat!(
                "Implement nonlinear state, residual, and tangent assembly. Implement ",
                "supported plastic integration and history. Implement contact search, ",
                "enforcement, friction, and tangent coupling.",
            ),
            EvidenceKind::AccuracyAcceptance,
            NONLINEAR_DOMAIN,
        ),
        obligation(
            "inequality_restart_converged_base",
            concat!(
                "Validate contact as a nonlinear inequality system alongside, but distinct ",
                "from, equality constraints. Implement restart and supported transfer of ",
                "nonlinear state. Publish only converged nonlinear base states that satisfy ",
                "the declared transition envelope; distinguish physical state from an ",
                "initial guess or incomplete checkpoint.",
            ),
            EvidenceKind::AccuracyAcceptance,
            NONLINEAR_DOMAIN,
        ),
        obligation(
            "downstream_checkpoints_release",
            concat!(
                "Qualify selected nonlinear base states for buckling or prestressed ",
                "downstream procedures only after their tangent, contact, material-history, ",
                "and transition evidence passes. Add nonlinear task checkpoints and bounded ",
                "retries. Add release evidence and documentation for each formulation ",
                "separately.",
            ),
            EvidenceKind::AccuracyAcceptance,
            NONLINEAR_DOMAIN,
        ),
        obligation(
            "executable_laws_independent_physics",
            concat!(
                "Keep state-dependent follower loads, controllers, and constitutive/user ",
                "laws outside ordinary excitation callbacks; reject them until the separate ",
                "executable-law ABI and tangent contract exist. Equilibrium is satisfied by ",
                "the implemented state-dependent operator. Constitutive and contact ",
                "benchmarks pass independently.",
            ),
            EvidenceKind::AccuracyAcceptance,
            NONLINEAR_DOMAIN,
        ),
        obligation(
            "sensitivity_conflict_diagnostics",
            concat!(
                "Contact complementarity, equality-constraint violation, and ",
                "overlap/conflict diagnostics pass independently. Load-step, mesh, and ",
                "contact-parameter sensitivity are governed.",
            ),
            EvidenceKind::AccuracyAcceptance,
            NONLINEAR_DOMAIN,
        ),
    ],
    witnesses: &[],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    ACCEPTED_HISTORY,
    FEA_NONLINEAR_OPERATOR_001,
    FEA_CONTACT_OPERATOR_001,
    FEA_NONLINEAR_GOVERNANCE_001,
];
