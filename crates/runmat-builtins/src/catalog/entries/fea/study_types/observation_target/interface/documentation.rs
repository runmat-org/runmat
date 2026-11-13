use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`id` is exact text and must not be empty. It names an interface authored in the same `fea.Model`; matching is by identifier, not by geometry.",
        "The constructor only records the reference. Model validation rejects the observation if no interface with that id exists or if the quantity is not produced at an interface location.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the target as the third argument of `fea.Observation.field`, `fea.Observation.history`, `fea.Observation.probe`, or `fea.Observation.reduction`. The observation is attached to a step through `fea.ProcedureStep.configure`, and model validation resolves the target against the analysis model before any solve work starts.",
        "Interface targets suit contact pressure, contact gap, and interface flux observations, usually with `fea.FieldLocation.interface()` as the source and output location.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when the id is empty or whitespace (`target id must not be empty`).",
        "`RunMat:fea:Internal` if the target cannot be converted into a RunMat object; this indicates a runtime defect, not an input problem.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "contact-pair",
        title: "Observe a bolted-joint contact interface",
        program: "target = fea.ObservationTarget.interface(\"flange_contact\");",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(target), 'fea.ObservationTarget'));\nassert(strcmp(target.value.spatial.interface.interface_id, 'flange_contact'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ObservationTarget.interface",
        summary: "Target a named model interface for an observation.",
        description: "`fea.ObservationTarget.interface(id)` builds a `fea.ObservationTarget` that refers to a model interface (for example a bonded or contact pair) by its identifier.",
        keywords: &["fea", "observation", "target", "interface", "contact", "bonded"],
        related: &["fea.ObservationTarget.port", "fea.ObservationTarget.connector", "fea.Observation.field", "fea.Observation.history", "fea.Observation.probe", "fea.Observation.reduction"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Is the id checked when I create the target?", answer: "Only for being non-empty. The model check confirms the interface exists." }],
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
