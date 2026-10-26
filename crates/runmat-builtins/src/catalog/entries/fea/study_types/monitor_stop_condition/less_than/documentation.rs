use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`quantity` is a `fea.ObservationQuantity`; `canonicalValue` is a finite real scalar in that quantity's canonical SI unit (for example pascals for stress). Integer classes are accepted.",
        "Model validation requires the threshold to be finite and `quantity` to match the monitored source observation's quantity, so a stress threshold cannot be compared with a displacement.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the condition as the fourth argument of `fea.Observation.monitor`.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `quantity` is not a `fea.ObservationQuantity` or `canonicalValue` is not a finite real scalar.",
        "`RunMat:fea:Internal` if the condition cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "residual",
        title: "Stop when the residual drops below 1e-8",
        program: "condition = fea.MonitorStopCondition.lessThan(fea.ObservationQuantity.residual(), 1e-8);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(condition), 'fea.MonitorStopCondition'));\nassert(condition.value.less_than.threshold.canonical_value == 1e-8);\nassert(strcmp(condition.value.less_than.threshold.quantity, 'residual'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.MonitorStopCondition.lessThan",
        summary: "Stop a step when a monitored scalar falls below a threshold.",
        description: "`fea.MonitorStopCondition.lessThan(quantity, canonicalValue)` builds a `fea.MonitorStopCondition` that stops the step when the monitored value is below `canonicalValue`.",
        keywords: &["fea", "monitor", "stop", "threshold", "less", "residual"],
        related: &["fea.MonitorStopCondition.greaterThan", "fea.MonitorStopCondition.absoluteChangeBelow", "fea.Observation.monitor"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Which unit is the threshold in?", answer: "The canonical SI unit of the quantity, with no unit conversion applied." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/requests.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/monitor_stop_condition/tests.rs::monitor_stop_condition_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
