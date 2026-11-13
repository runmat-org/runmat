use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`id` is exact text and must not be empty. It names a port authored in the same `fea.Model`.",
        "The constructor only records the reference; model validation confirms the port exists and that the requested quantity is defined there.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the target as the third argument of `fea.Observation.field`, `fea.Observation.history`, `fea.Observation.probe`, or `fea.Observation.reduction`. The observation is attached to a step through `fea.ProcedureStep.configure`, and model validation resolves the target against the analysis model before any solve work starts.",
        "Ports are the natural target for terminal current, mass flow rate through an inlet, or potential at an electrode.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when the id is empty or whitespace (`target id must not be empty`).",
        "`RunMat:fea:Internal` if the target cannot be converted into a RunMat object; this indicates a runtime defect, not an input problem.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "inlet",
        title: "Observe flow through an inlet port",
        program: "target = fea.ObservationTarget.port(\"coolant_inlet\");",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(target), 'fea.ObservationTarget'));\nassert(strcmp(target.value.spatial.port.port_id, 'coolant_inlet'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ObservationTarget.port",
        summary: "Target a named model port for an observation.",
        description: "`fea.ObservationTarget.port(id)` builds a `fea.ObservationTarget` that refers to a model port (for example an electrical terminal or a fluid inlet) by its identifier.",
        keywords: &["fea", "observation", "target", "port", "terminal", "inlet"],
        related: &["fea.ObservationTarget.interface", "fea.Observation.field", "fea.Observation.history", "fea.Observation.probe", "fea.Observation.reduction"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Can a port target carry a spatial reduction?", answer: "Yes, when the quantity and reduction are compatible; model validation checks the combination." }],
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
