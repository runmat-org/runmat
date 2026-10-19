use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Takes no arguments. The condition is absent, so the object has no `value` property to read.",
        "The monitor still evaluates at its cadence and reports progress.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the condition as the fourth argument of `fea.Observation.monitor`.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:Internal` if the condition cannot be converted into a RunMat object.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "progress-only",
        title: "Progress-only monitor",
        program: "condition = fea.MonitorStopCondition.none();\nmonitor = fea.Observation.monitor(\"progress\", \"tip_deflection\", ...\n    fea.MonitorCadence.acceptedIncrements(10), condition);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(condition), 'fea.MonitorStopCondition'));\nassert(strcmp(class(monitor), 'fea.Observation'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.MonitorStopCondition.none",
        summary: "Use a monitor for progress reporting only, without stopping.",
        description: "`fea.MonitorStopCondition.none()` builds a `fea.MonitorStopCondition` that never stops the step.",
        keywords: &["fea", "monitor", "stop", "none", "progress"],
        related: &["fea.MonitorStopCondition.lessThan", "fea.Observation.monitor"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Why does a progress-only monitor need a condition argument?", answer: "Every monitor states its stop behavior explicitly; `none()` is that explicit choice." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/requests.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/monitor_stop_condition/tests.rs::monitor_stop_condition_constructors_follow_their_positional_contract" },
        ],
        notes: &["Known gap: the registered immutable-value schema requires a `value` field, so encoding a standalone `none` condition through the value codec is rejected. The condition nested inside a monitor is unaffected."],
    });
