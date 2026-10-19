use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Exactly seven arguments are required. The four ids are non-empty text. `method` is reuse, conservativeProjection, interpolation, linearize, deriveOperatorContribution, or initialGuessOnly; `topology` is requireIdentical, projectWithEvidence, or adaptedMeshLineage; `reference` is preserve, receivingStepOrigin, or explicitLinearizationPoint.",
        "Model validation requires unique ids, an existing upstream producer and a different consumer, a direct `stateTransfer` dependency between them, a complete persisted producer state, and exactly one matching upstream `fea.StateInput` on the consumer.",
        "Policy rules: `reuse` requires identical topology; adapted-mesh lineage requires projection or interpolation; `initialGuessOnly` is used exactly when the input's use is `initialGuess`; `linearize` and `deriveOperatorContribution` require `explicitLinearizationPoint`.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass it, alone or in a cell array, as the `transitions` argument (4th) of `fea.AnalysisSequence.new`.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when the argument count is not seven, an id is empty or not text, or a policy is not the expected typed value.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "prestress",
        title: "Linearize a preload state into a modal step",
        program: concat!(
            "transition = fea.StateTransition.new(\"prestress_transfer\", \"preload\", \"modes\", \"converged_stress\", ...\n",
            "    fea.StateTransitionMethod.linearize(), fea.TopologyTransitionPolicy.requireIdentical(), ...\n",
            "    fea.ReferenceStatePolicy.explicitLinearizationPoint());",
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(transition), 'fea.StateTransition'));\nassert(strcmp(transition.transition_id, 'prestress_transfer'));\nassert(strcmp(transition.method, 'linearize'));\nassert(strcmp(transition.reference, 'explicit_linearization_point'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StateTransition.new",
        summary: "Govern how one published state crosses from a producer step to a consumer step.",
        description: "`fea.StateTransition.new(id, producerStepId, consumerStepId, stateId, method, topology, reference)` builds a `fea.StateTransition` that transfers the state `stateId` from one step to another with an explicit method, topology rule, and reference-state rule.",
        keywords: &["fea", "sequence", "state", "transition", "transfer", "prestress"],
        related: &["fea.StateInput.upstream", "fea.StateOutput.new", "fea.SequenceDependency.new", "fea.AnalysisSequence.new"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Can a transition change the mesh?", answer: "Only with `projectWithEvidence` or `adaptedMeshLineage`, and then with projection or interpolation, never with plain reuse." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/state_transition/tests.rs::state_transition_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Governed transitions", location: "crates/runmat-analysis/core/src/validate_sequence/tests.rs::rejects_upstream_input_without_governed_transition" },
        ],
        notes: &[],
    });
