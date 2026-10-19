use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`quantity` is a `fea.ObservationQuantity`, `canonicalValue` a finite real scalar in its canonical SI unit, and `consecutive` a nonnegative integer scalar (integer classes accepted).",
        "Model validation requires the threshold to be finite and nonnegative, `consecutive` to be nonzero, and `quantity` to match the monitored source's quantity.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the condition as the fourth argument of `fea.Observation.monitor`.",
        "Use it to end a pseudo-transient or nonlinear solve once the response has settled.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when an argument has the wrong type, the threshold is nonfinite, or `consecutive` is not a nonnegative integer.",
        "`RunMat:fea:Internal` if the condition cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "settled",
        title: "Stop after three evaluations with change below 1e-8",
        program: "condition = fea.MonitorStopCondition.absoluteChangeBelow(fea.ObservationQuantity.residual(), 1e-8, 3);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(condition), 'fea.MonitorStopCondition'));\nassert(condition.value.absolute_change_below.consecutive == 3);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.MonitorStopCondition.absoluteChangeBelow",
        summary: "Stop a step once a monitored scalar stops changing.",
        description: "`fea.MonitorStopCondition.absoluteChangeBelow(quantity, canonicalValue, consecutive)` builds a `fea.MonitorStopCondition` that stops the step after the monitored value changes by less than `canonicalValue` for `consecutive` evaluations in a row.",
        keywords: &["fea", "monitor", "stop", "change", "stagnation", "steady"],
        related: &["fea.MonitorStopCondition.lessThan", "fea.MonitorCadence.acceptedIncrements", "fea.Observation.monitor"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Is the change relative?", answer: "No. It is an absolute change in the quantity's canonical unit between consecutive evaluations." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/requests.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/monitor_stop_condition/tests.rs::monitor_stop_condition_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
