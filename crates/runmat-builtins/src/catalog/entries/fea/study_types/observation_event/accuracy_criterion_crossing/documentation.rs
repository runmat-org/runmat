use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Takes no arguments. The event is owned by the accuracy plan.",
        "An accuracy event names its source accuracy report, so the sample is attributed to the criterion that produced it.",
        "Valid for any non-static independent domain.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the event to `fea.SamplingPolicy.event(event)`; the resulting policy is the sampling argument of an observation request.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:Internal` if the event cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "accuracycriterioncrossing",
        title: "Sample when an accuracy criterion is crossed",
        program: "event = fea.ObservationEvent.accuracyCriterionCrossing();\npolicy = fea.SamplingPolicy.event(event);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(event), 'fea.ObservationEvent'));\nassert(strcmp(event.value.accuracy, 'criterion_crossing'));\nassert(strcmp(class(policy), 'fea.SamplingPolicy'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ObservationEvent.accuracyCriterionCrossing",
        summary: "Sample when an accuracy criterion crosses its tolerance.",
        description: "`fea.ObservationEvent.accuracyCriterionCrossing()` returns the typed `fea.ObservationEvent` for sample when an accuracy criterion crosses its tolerance.",
        keywords: &["fea", "observation", "event", "accuracy", "criterion"],
        related: &["fea.SamplingPolicy.event", "fea.Observation.history"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Where are the criteria defined?", answer: "In the step's `fea.AccuracyPlan`, built with `fea.AccuracyPlan.new` and `fea.AccuracyCriterion.new`." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/sampling.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/observation_event/tests.rs::observation_event_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Event ownership", location: "crates/runmat-runtime/src/builtins/fea/observations/sampling.rs::tests::event_constructors_preserve_their_domain_owner" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Event/domain compatibility", location: "crates/runmat-analysis/core/src/validate_outputs/sampling.rs::tests::solver_transition_events_require_a_physical_progress_domain" },
        ],
        notes: &[],
    });
