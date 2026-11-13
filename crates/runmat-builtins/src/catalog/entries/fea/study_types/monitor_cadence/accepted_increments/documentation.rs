use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`interval` must be a nonnegative integer scalar; integer classes and integer-valued doubles are accepted exactly. Model validation requires it to be nonzero.",
        "Only accepted increments count, so cut-backs do not trigger extra evaluations. The cadence is deterministic and independent of wall-clock speed.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass it as the third argument of `fea.Observation.monitor`.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `interval` is not a nonnegative integer scalar.",
        "`RunMat:fea:Internal` if the cadence cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "every-five",
        title: "Evaluate every five accepted increments",
        program: "cadence = fea.MonitorCadence.acceptedIncrements(5);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(cadence), 'fea.MonitorCadence'));\nassert(cadence.value.accepted_increments.interval == 5);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.MonitorCadence.acceptedIncrements",
        summary: "Evaluate a monitor every N accepted solver increments.",
        description: "`fea.MonitorCadence.acceptedIncrements(interval)` builds a `fea.MonitorCadence` that evaluates a monitor after every `interval` accepted increments.",
        keywords: &["fea", "monitor", "cadence", "increments", "solver"],
        related: &["fea.MonitorCadence.wallSeconds", "fea.Observation.monitor"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Is `acceptedIncrements(0)` allowed?", answer: "The constructor accepts it, but model validation rejects a zero interval." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/requests.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/monitor_cadence/tests.rs::monitor_cadence_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
