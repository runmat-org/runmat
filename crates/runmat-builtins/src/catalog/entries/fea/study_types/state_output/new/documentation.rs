use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`id` is non-empty text. `kind` is a `fea.StateKind` (structural primary, kinematics, stress/strain, geometric stiffness, material or contact history, thermal, fluid, electromagnetic, acoustic, modal basis, buckling modes, moving mesh, coupled interface, or integrator). `completeness` is complete, checkpoint, or partial; `persistence` is canonical, untilDependentsComplete, or checkpointOnly.",
        "Model validation requires every step to publish at least one state, ids to be unique within the step, the step's physics family to be able to publish the kind, a complete state not to be checkpoint-only, and a checkpoint state to be checkpoint-only.",
        "Only a complete, persisted state may cross a procedure boundary; branches additionally require canonical persistence.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Collect state outputs in a cell array and pass them as the `stateOutputs` argument (5th) of `fea.ProcedureStep.configure`. Downstream steps consume them through `fea.StateInput.upstream` and a `fea.StateTransition`.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `id` is empty or an argument is not the expected typed value.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "equilibrium",
        title: "Publish the converged static equilibrium",
        program: "state = fea.StateOutput.new(\"static_equilibrium\", fea.StateKind.structuralPrimary(), ...\n    fea.StateCompleteness.complete(), fea.StatePersistence.canonical());\n",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(state), 'fea.StateOutput'));\nassert(strcmp(state.state_id, 'static_equilibrium'));\nassert(strcmp(state.kind, 'structural_primary'));\nassert(strcmp(state.persistence, 'canonical'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StateOutput.new",
        summary: "Declare an immutable state that a procedure step publishes.",
        description: "`fea.StateOutput.new(id, kind, completeness, persistence)` builds a `fea.StateOutput` declaring a typed state, such as a converged equilibrium or a modal basis, that a step publishes for later steps.",
        keywords: &["fea", "sequence", "state", "output", "publish", "step"],
        related: &["fea.StateInput.upstream", "fea.StateTransition.new", "fea.ProcedureStep.configure"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Why must every step publish a state?", answer: "The state is how the sequence proves what a step produced; even a single static step publishes its equilibrium." }, BuiltinDocumentationFaq { question: "Can a checkpoint feed a downstream step?", answer: "No. Only complete persisted states cross procedure boundaries; checkpoints serve restarts." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/state_output/tests.rs::state_output_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed sequence round trip", location: "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::configured_sequence_round_trips_with_typed_study_contracts" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Compiled sequence execution", location: "crates/runmat-vm/tests/fea_study_contracts.rs::typed_study_sequence_compiles_and_executes_through_the_vm" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Incomplete upstream state", location: "crates/runmat-analysis/core/src/validate_sequence/tests.rs::rejects_incomplete_upstream_state" },
        ],
        notes: &[],
    });
