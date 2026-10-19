//! Standing acceptance for formulation-specific adaptive production workflows.
use crate::requirement::*;
const DOMAIN: &[Envelope] = &[
    envelope(
        Formulation::AdaptiveAnalysis,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::AdaptiveAnalysis,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuNumericalKernel,
    ),
    envelope(
        Formulation::AdaptiveAnalysis,
        MeshOrder::Tet4,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::AdaptiveAnalysis,
        MeshOrder::Tet10,
        ExecutionEnvelope::SerialCpuPublicStudy,
    ),
    envelope(
        Formulation::AdaptiveAnalysis,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::SerialCpuNativeProcess,
    ),
    envelope(
        Formulation::AdaptiveAnalysis,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::AdaptiveAnalysis,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::EncryptedRemote,
    ),
    envelope(
        Formulation::AdaptiveAnalysis,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::PartitionedCpu,
    ),
    envelope(
        Formulation::AdaptiveAnalysis,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
const FEA_ADAPTATION_LOOP_001: Requirement = Requirement {
    id: "FEA-ADAPTATION-LOOP-001",
    title: "Family-appropriate adaptive loops and separately qualified transfer",
    stages: &[Stage::Adaptation],
    acceptance: concat!(
        "Family-appropriate adaptive loops and separately qualified transfer must ",
        "satisfy all applicable supported-envelope criteria. Independent physical, ",
        "semantic, accuracy, resource and publication evidence are separate; ",
        "missing, failed or indeterminate mandatory evidence prevents production ",
        "acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/adaptation/iteration.rs"],
    obligations: &[
        obligation(
            "family_estimator_canonical_order",
            concat!(
                "Estimators must be formulation-appropriate and validated; a structural ",
                "stress estimator is not a universal estimator. Adaptation decisions use ",
                "stable identities and canonical evidence. Tet4/Tet10 order changes preserve ",
                "correct field and integration-point state.",
            ),
            EvidenceKind::SpatialConvergence,
            DOMAIN,
        ),
        obligation(
            "qualified_transfer_governed_loop",
            concat!(
                "Plastic, contact, coupled-interface, and temporal state transfer must be ",
                "separately validated before adaptation is enabled for those families. ",
                "Topology change alone never constitutes convergence. Iteration limits, ",
                "resource budgets, cancellation, retry, and publication fencing apply to the ",
                "full loop.",
            ),
            EvidenceKind::SpatialConvergence,
            DOMAIN,
        ),
        obligation(
            "loop",
            concat!(
                "Compose solve, family recovery, estimation, canonical marking, h/p ",
                "adaptation, new mesh/lineage validation, primary field/history transfer, ",
                "measured transfer error, resolve and estimator/target convergence into a ",
                "governed continue/converge/reject orchestrator.",
            ),
            EvidenceKind::ControllerChronology,
            DOMAIN,
        ),
    ],
    witnesses: &[],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
const FEA_ADAPTATION_GOVERNANCE_001: Requirement = Requirement {
    id: "FEA-ADAPTATION-GOVERNANCE-001",
    title: "Accuracy-driven adaptation, persistent constraints and durable lineage",
    stages: &[
        Stage::Adaptation
    ],
    acceptance: concat!(
"Accuracy-driven adaptation, persistent constraints and durable lineage must ",
"satisfy all applicable supported-envelope criteria. Independent physical, ",
"semantic, accuracy, resource and publication evidence are separate; ",
"missing, failed or indeterminate mandatory evidence prevents production ",
"acceptance.",
),
    owners: &[
        "crates/runmat-analysis/fea/src/adaptation/iteration.rs"
    ],
    obligations: &[
        obligation(
            "family_transfer_constraints",
            concat!(
"Compose adaptive loops for each formulation with a validated estimator. ",
"Validate field and state transfer. Rebuild constraint plans from persistent ",
"identities and validate constraint-plan lineage after every h- or p-change.",
),
            EvidenceKind::AccuracyAcceptance,
            DOMAIN
        ),
        obligation(
            "durable_accuracy_lineage",
            concat!(
"Integrate adaptation graphs with durable and remote execution. Publish ",
"iteration lineage and target-quantity convergence in results. Drive ",
"stopping and acceptance through the shared accuracy plan, preserving family ",
"estimator evidence rather than inventing a separate adaptive tolerance ",
"model.",
),
            EvidenceKind::AccuracyAcceptance,
            DOMAIN
        ),
        obligation(
            "release_convergence_transfer",
            concat!(
"Add release evidence and documentation. Adaptation reduces the governed ",
"estimator and converges the selected target quantity. Transfer error ",
"remains below family-specific limits.",
),
            EvidenceKind::AccuracyAcceptance,
            DOMAIN
        ),
        obligation(
            "constraint_state_lifecycle",
            concat!(
"Constraint targets, relations, reactions, and state remain valid across ",
"adaptation. Cancellation, retry, restart, and replay preserve valid ",
"iteration lineage.",
),
            EvidenceKind::AccuracyAcceptance,
            DOMAIN
        ) ,
        obligation(
            "constraint_lineage",
            concat!(
"Constraints bind to persistent geometry and topology identities rather than ",
"incidental node numbers. After h- or p-adaptation, the analysis projection ",
"rebuilds the canonical constraint plan, checks target coverage and pairing, ",
"transfers necessary state, and records old/new plan lineage.",
),
            EvidenceKind::SemanticRealization,
            DOMAIN
        )
    ],
    witnesses: &[
        Witness {
            id: "FEA-WITNESS-ADAPTATION-CONSTRAINTS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"adaptation::iteration::tests::constraint_lineage::persistent_clamp_rebuilds_",
"new_midside_constraints_without_numeric_node_aliases",
),
                source: "crates/runmat-analysis/fea/src/adaptation/iteration/tests/constraint_lineage.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "family_transfer_constraints",
                    envelopes: &[
                        DOMAIN[
                            0
                        ],
                        DOMAIN[
                            1
                        ]
                    ],
                    boundary: concat!(
"Real Tet4 to Tet10 persistent face clamp rebuilds9 to18 rows, retains ",
"entity identity and rejects foreign/missing lineage. No reaction/state ",
"transfer or adaptive solve convergence claim.",
)
                }
            ]
        },
        Witness {
            id: "FEA-WITNESS-ADAPTATION-ITERATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
"adaptation::iteration::tests::governed::iteration_binds_canonical_solver_mes",
"h_lineage_transfer_and_evidence",
),
                source: "crates/runmat-analysis/fea/src/adaptation/iteration/tests/governed.rs"
            },
            claims: &[
                CoverageClaim {
                    obligation: "durable_accuracy_lineage",
                    envelopes: &[
                        DOMAIN[
                            0
                        ]
                    ],
                    boundary: concat!(
"Canonical adaptation records use constructed converged solver facts and ",
"quantities, real mesh refinement/transfer fixtures; rejects ",
"missing/indeterminate evidence and capped refinement. No independently ",
"solved adaptive target convergence or remote lifecycle claim.",
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
pub(crate) const REQUIREMENTS: &[Requirement] =
    &[FEA_ADAPTATION_LOOP_001, FEA_ADAPTATION_GOVERNANCE_001];
