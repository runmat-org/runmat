use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`event` must be a `fea.ObservationEvent` from one of its constructors. The event keeps its owner: solver events name a completed result or failed attempt, series events name their source history, and accuracy events name their source report.",
        "`firstYield` and `contactStatusChange` require a time or load-factor domain. `convergenceFailure`, `peakResponse`, and `accuracyCriterionCrossing` are valid for any non-static domain.",
        "If the event never occurs the history is valid with zero samples, so “observed no event” stays distinct from “not run”. Because the inventory can be empty, model validation rejects series reductions on event-sampled observations.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the policy as the `sampling` argument (11th) of `fea.Observation.field`, `history`, `probe`, or `reduction`, or as the third argument of `fea.Observation.diagnostic`. Model validation checks it against the independent domain of the step that owns the observation (`fea.IndependentDomain.*`).",
        "When observation evaluation is partitioned, each event occurrence is its own fan-out with its attribution in every task and reduction identity.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `event` is not a `fea.ObservationEvent`.",
        "`RunMat:fea:Internal` if the policy cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "first-yield",
        title: "Sample at first yield of a nonlinear step",
        program: "policy = fea.SamplingPolicy.event(fea.ObservationEvent.firstYield());",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(policy), 'fea.SamplingPolicy'));\nassert(strcmp(policy.value.event.solver, 'first_yield'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.SamplingPolicy.event",
        summary: "Retain a sample whenever a typed solver, series, or accuracy event occurs.",
        description: "`fea.SamplingPolicy.event(event)` builds a `fea.SamplingPolicy` that samples when the given `fea.ObservationEvent` occurs, such as first yield or a contact status change.",
        keywords: &["fea", "observation", "sampling", "event", "yield", "contact"],
        related: &["fea.ObservationEvent.firstYield", "fea.ObservationEvent.peakResponse", "fea.Observation.history"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "What if the event happens several times?", answer: "Each occurrence is a separate sample with its own attribution." }, BuiltinDocumentationFaq { question: "Can I take the maximum over event samples?", answer: "Not with a series reduction, because the inventory may be empty. Reduce a separate, nonempty history instead." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/sampling.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/sampling_policy/tests.rs::sampling_policy_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Event sampling", location: "crates/runmat-runtime/src/builtins/fea/observations/sampling.rs::tests::sampling_event_keeps_the_typed_event" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Event/domain compatibility", location: "crates/runmat-analysis/core/src/validate_outputs/sampling.rs::tests::solver_transition_events_require_a_physical_progress_domain" },
        ],
        notes: &[],
    });
