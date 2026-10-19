use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`id` and `producerStepId` are non-empty text. `stateIds` and `consumerStepIds` accept a text scalar, a string array, or a cell array of text.",
        "Model validation requires a unique branch id, a known producer, at least one state with no repeats, each state published by the producer as complete and canonical, and at least two distinct known consumers that are downstream of the producer and each consume at least one of the branch's states.",
        "A branch describes physical fan-out; the execution planner decides whether local or cluster resources run the consumers concurrently.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass it, alone or in a cell array, as the `branches` argument (5th) of `fea.AnalysisSequence.new`.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when an id is empty or a list contains a non-text element.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "modal-fanout",
        title: "Share one modal basis with two response steps",
        program: "branch = fea.SequenceBranch.new(\"modal_fanout\", \"modes\", \"modal_basis\", {\"harmonic\", \"random\"});",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(branch), 'fea.SequenceBranch'));\nassert(strcmp(branch.branch_id, 'modal_fanout'));\nassert(numel(branch.consumer_steps) == 2);\nassert(numel(branch.state_ids) == 1);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.SequenceBranch.new",
        summary: "Publish states once for several independent downstream steps.",
        description: "`fea.SequenceBranch.new(id, producerStepId, stateIds, consumerStepIds)` builds a `fea.SequenceBranch` declaring that states published by one step are consumed independently by several downstream steps.",
        keywords: &["fea", "sequence", "branch", "fan-out", "state", "parallel"],
        related: &["fea.StateOutput.new", "fea.StateTransition.new", "fea.AnalysisSequence.new"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Why at least two consumers?", answer: "A branch exists to share states across independent consumers; a single consumer is an ordinary transition." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/sequence_branch/tests.rs::sequence_branch_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Branch validation", location: "crates/runmat-analysis/core/src/validate_sequence/tests.rs::rejects_invalid_branches" },
        ],
        notes: &[],
    });
