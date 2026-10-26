use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`id` and `sourceObservationId` are non-empty text. `cadence` is a `fea.MonitorCadence`; `stop` is a `fea.MonitorStopCondition`, where `fea.MonitorStopCondition.none()` reports progress only.",
        "Model validation requires the source to name a field, history, probe, or reduction in the same step that produces scalar values, forbids a monitor observing itself, and requires the stop threshold's quantity to match the source quantity.",
        "A monitor can stop work; it cannot modify a load, material, constraint, or solver state.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Add the monitor and its source observation to the same step's observation list in `fea.ProcedureStep.configure`.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when an id is empty, or `cadence` or `stop` is not the expected typed value.",
        "`RunMat:fea:Internal` if the request cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "residual-stop",
        title: "Stop when the residual stagnates",
        program: "monitor = fea.Observation.monitor(\"stop_on_residual\", \"residual_norm\", ...\n    fea.MonitorCadence.acceptedIncrements(5), ...\n    fea.MonitorStopCondition.absoluteChangeBelow(fea.ObservationQuantity.residual(), 1e-8, 3));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(monitor), 'fea.Observation'));\nassert(strcmp(monitor.value.monitor.source_observation_id, 'residual_norm'));\nassert(monitor.value.monitor.cadence.accepted_increments.interval == 5);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Observation.monitor",
        summary: "Watch a scalar observation during a solve and optionally stop the step.",
        description: "`fea.Observation.monitor(id, sourceObservationId, cadence, stop)` builds a `fea.Observation` that evaluates a scalar observation at a bounded cadence and can stop work when a typed condition is met.",
        keywords: &["fea", "observation", "monitor", "stop", "progress", "solver"],
        related: &["fea.MonitorCadence.acceptedIncrements", "fea.MonitorStopCondition.lessThan", "fea.Observation.reduction", "fea.ProcedureStep.configure"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Can a monitor watch a vector field?", answer: "No. The source must produce scalar values, for example a reduction or a single component." }, BuiltinDocumentationFaq { question: "Does stopping discard results?", answer: "The step stops at the monitored point; the monitor never changes loads, materials, or solver state." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/requests.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/observation/tests.rs::observation_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
