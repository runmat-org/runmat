use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`interval` must be a finite real scalar in seconds; integer classes are accepted. Model validation requires it to be positive.",
        "Wall-clock cadence depends on machine speed, so the evaluation points can differ between runs. Use `acceptedIncrements` when a stop decision must be reproducible.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass it as the third argument of `fea.Observation.monitor`, typically with `fea.MonitorStopCondition.none()` for progress reporting.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `interval` is not a finite real numeric scalar.",
        "`RunMat:fea:Internal` if the cadence cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "progress",
        title: "Report progress every 30 s",
        program: "cadence = fea.MonitorCadence.wallSeconds(30);\nprogress = fea.Observation.monitor(\"progress\", \"peak_displacement\", cadence, fea.MonitorStopCondition.none());",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(cadence), 'fea.MonitorCadence'));\nassert(cadence.value.wall_seconds.interval == 30);\nassert(strcmp(class(progress), 'fea.Observation'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.MonitorCadence.wallSeconds",
        summary: "Evaluate a monitor at a wall-clock interval in seconds.",
        description: "`fea.MonitorCadence.wallSeconds(interval)` builds a `fea.MonitorCadence` that evaluates a monitor about every `interval` seconds of elapsed run time.",
        keywords: &["fea", "monitor", "cadence", "wall", "clock", "seconds"],
        related: &["fea.MonitorCadence.acceptedIncrements", "fea.Observation.monitor"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Is a wall-clock stop reproducible?", answer: "Not exactly; evaluation points depend on run speed. Prefer `acceptedIncrements` for stop conditions." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/requests.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/monitor_cadence/tests.rs::monitor_cadence_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
