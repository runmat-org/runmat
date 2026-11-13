use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Takes no arguments and always returns the same immutable target.",
        "A whole-model target has no single measure, so `integral` and `resultant` reductions that need an area or volume are rejected for most quantities; reaction-force resultants are sums of nodal forces and remain valid.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the target as the third argument of `fea.Observation.field`, `fea.Observation.history`, `fea.Observation.probe`, or `fea.Observation.reduction`. The observation is attached to a step through `fea.ProcedureStep.configure`, and model validation resolves the target against the analysis model before any solve work starts.",
        "This is the usual target for full displacement or stress fields, global strain energy, and whole-model reaction totals.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:Internal` if the target cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "full-field",
        title: "Target the full model",
        program: "target = fea.ObservationTarget.wholeModel();",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(target), 'fea.ObservationTarget'));\nassert(strcmp(target.value, 'whole_model'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ObservationTarget.wholeModel",
        summary: "Target the entire analysis model for an observation.",
        description: "`fea.ObservationTarget.wholeModel()` builds a `fea.ObservationTarget` covering every entity of the analysis model, such as all nodes for a displacement field.",
        keywords: &["fea", "observation", "target", "whole", "model", "global"],
        related: &["fea.ObservationTarget.volume", "fea.ObservationCost.new", "fea.Observation.field", "fea.Observation.history", "fea.Observation.probe", "fea.Observation.reduction"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Is a whole-model field expensive?", answer: "It can be. Declare a realistic `fea.ObservationCost` and use `fea.PartialAccessPolicy.manifestAndChunks()` so consumers fetch only the chunks they need." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/targets.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/observation_target/tests.rs::observation_target_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed sequence round trip", location: "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::configured_sequence_round_trips_with_typed_study_contracts" },
        ],
        notes: &[],
    });
