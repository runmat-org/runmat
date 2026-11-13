use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`id` is exact text and must not be empty. It names a connector authored in the same `fea.Model`.",
        "The constructor only records the reference; model validation confirms the connector exists and that the quantity is available at `fea.FieldLocation.connector()`.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the target as the third argument of `fea.Observation.field`, `fea.Observation.history`, `fea.Observation.probe`, or `fea.Observation.reduction`. The observation is attached to a step through `fea.ProcedureStep.configure`, and model validation resolves the target against the analysis model before any solve work starts.",
        "Use connector targets for fastener forces and connector elongation histories.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when the id is empty or whitespace (`target id must not be empty`).",
        "`RunMat:fea:Internal` if the target cannot be converted into a RunMat object; this indicates a runtime defect, not an input problem.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "fastener",
        title: "Observe one fastener",
        program: "target = fea.ObservationTarget.connector(\"bolt_1\");",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(target), 'fea.ObservationTarget'));\nassert(strcmp(target.value.connector.connector_id, 'bolt_1'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ObservationTarget.connector",
        summary: "Target a named connector for an observation.",
        description: "`fea.ObservationTarget.connector(id)` builds a `fea.ObservationTarget` that refers to a model connector (for example a spring, bushing, or fastener) by its identifier.",
        keywords: &["fea", "observation", "target", "connector", "fastener", "spring"],
        related: &["fea.ObservationTarget.constraint", "fea.ObservationTarget.interface", "fea.Observation.field", "fea.Observation.history", "fea.Observation.probe", "fea.Observation.reduction"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Can I target several connectors in one observation?", answer: "No. Declare one observation per connector so each result keeps its own identity." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "FEA models", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/models") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/targets.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/observation_target/tests.rs::observation_target_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
