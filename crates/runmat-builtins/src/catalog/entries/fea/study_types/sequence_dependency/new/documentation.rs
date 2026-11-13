use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Both step ids are non-empty text. `reason` is `stateTransfer`, `authoredOrdering`, `sharedInterface`, or `adaptation`.",
        "Model validation rejects unknown step ids, a step depending on itself, duplicate edges, and cycles. Every `fea.StateTransition` needs a direct dependency with reason `stateTransfer` between its producer and consumer.",
        "Dependencies describe physical order only. The execution planner derives a topological order and parallel levels, and steps without a path between them may run concurrently.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass it, alone or in a cell array, as the `dependencies` argument (3rd) of `fea.AnalysisSequence.new`.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when a step id is empty or `reason` is not a `fea.SequenceDependencyReason`.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "preload-modes",
        title: "Run the modal step after the preload",
        program: "dependency = fea.SequenceDependency.new(\"preload\", \"modes\", fea.SequenceDependencyReason.stateTransfer());",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(dependency), 'fea.SequenceDependency'));\nassert(strcmp(dependency.predecessor, 'preload'));\nassert(strcmp(dependency.successor, 'modes'));\nassert(strcmp(dependency.reason, 'state_transfer'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.SequenceDependency.new",
        summary: "Declare a typed ordering edge between two procedure steps.",
        description: "`fea.SequenceDependency.new(predecessorStepId, successorStepId, reason)` builds a `fea.SequenceDependency` stating that one step must complete before another and why.",
        keywords: &["fea", "sequence", "dependency", "ordering", "step"],
        related: &["fea.StateTransition.new", "fea.AnalysisSequence.new"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Is listing order in the sequence enough?", answer: "No. Order comes only from dependencies; steps without one may run in parallel." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/sequence_dependency/tests.rs::sequence_dependency_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Dependency cycles", location: "crates/runmat-analysis/core/src/validate_sequence/tests.rs::rejects_dependency_cycles" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Parallel topology", location: "crates/runmat-analysis/core/src/validate_sequence/tests.rs::exposes_deterministic_parallel_sequence_topology" },
        ],
        notes: &[],
    });
