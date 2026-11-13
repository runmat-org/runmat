use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "All ids are non-empty text. `kind` must equal the kind the producer publishes; `useAs` says how the consumer uses it (for example `prestress` or `initialGuess`); `conflict` says how it combines with any authored initial conditions: reject, replace the initial condition, or augment disjoint quantities. Two definitions of the same state are never merged silently.",
        "Model validation requires the producer to exist and be upstream, to publish `stateId` as a complete persisted state of the required kind, the consumer's physics family to consume that kind, and exactly one `fea.StateTransition` to govern the input.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Collect state inputs in a cell array and pass them as the `stateInputs` argument (4th) of `fea.ProcedureStep.configure`.",
        "Pair it with a `fea.SequenceDependency` (reason `stateTransfer`) and a `fea.StateTransition` in `fea.AnalysisSequence.new`. A continuation restart additionally needs an upstream input with `fea.StateUse.restart()`.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when an id is empty or an argument is not the expected typed value.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "prestress",
        title: "Use a preload step's stress as prestress",
        program: "input = fea.StateInput.upstream(\"prestress\", fea.StateKind.structuralStressStrain(), ...\n    \"preload\", \"converged_stress\", fea.StateUse.prestress(), fea.InitialStateConflictPolicy.reject());",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(input), 'fea.StateInput'));\nassert(strcmp(input.source.upstream.producer_step, 'preload'));\nassert(strcmp(input.source.upstream.state_id, 'converged_stress'));\nassert(strcmp(input.use_as, 'prestress'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StateInput.upstream",
        summary: "Consume a typed state published by an upstream procedure step.",
        description: "`fea.StateInput.upstream(id, kind, producerStepId, stateId, useAs, conflict)` builds a `fea.StateInput` that takes the state `stateId` published by step `producerStepId`.",
        keywords: &["fea", "sequence", "state", "input", "upstream", "prestress"],
        related: &["fea.StateOutput.new", "fea.StateTransition.new", "fea.SequenceDependency.new", "fea.ProcedureStep.configure"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Does declaring the input transfer the state?", answer: "No. The transfer is governed by a `fea.StateTransition`; validation rejects an upstream input with no governing transition." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/state_input/tests.rs::state_input_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Upstream state kind", location: "crates/runmat-analysis/core/src/validate_sequence/tests.rs::rejects_incompatible_upstream_state_kind" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Governed transitions", location: "crates/runmat-analysis/core/src/validate_sequence/tests.rs::rejects_upstream_input_without_governed_transition" },
        ],
        notes: &[],
    });
