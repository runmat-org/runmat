use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`quantity` is a `fea.ObservationQuantity`; `canonicalValue` is a finite real scalar in that quantity's canonical SI unit. Integer classes are accepted.",
        "Model validation requires the threshold to be finite and `quantity` to match the monitored source's quantity.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the condition as the fourth argument of `fea.Observation.monitor`.",
        "A typical use is ending a load ramp once peak stress passes the yield strength.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `quantity` is not a `fea.ObservationQuantity` or `canonicalValue` is not a finite real scalar.",
        "`RunMat:fea:Internal` if the condition cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "yield",
        title: "Stop when von Mises stress passes 250 MPa",
        program: "condition = fea.MonitorStopCondition.greaterThan(fea.ObservationQuantity.vonMisesStress(), 250e6);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(condition), 'fea.MonitorStopCondition'));\nassert(condition.value.greater_than.threshold.canonical_value == 250e6);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.MonitorStopCondition.greaterThan",
        summary: "Stop a step when a monitored scalar exceeds a threshold.",
        description: "`fea.MonitorStopCondition.greaterThan(quantity, canonicalValue)` builds a `fea.MonitorStopCondition` that stops the step when the monitored value is above `canonicalValue`.",
        keywords: &["fea", "monitor", "stop", "threshold", "greater", "limit"],
        related: &["fea.MonitorStopCondition.lessThan", "fea.Observation.monitor"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Can I give the threshold in MPa?", answer: "No. Thresholds are canonical SI values, so 250 MPa is `250e6`." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/requests.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/monitor_stop_condition/tests.rs::monitor_stop_condition_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
