use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`id` is exact text and must not be empty. It names a constraint authored in the same `fea.Model`.",
        "The executed constraint plan is the sole authority for which degrees of freedom belong to the group. Constraint-target reaction totals remain fail-closed until evaluation supplies that exact plan, so RunMat never estimates membership from geometry.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the target as the third argument of `fea.Observation.field`, `fea.Observation.history`, `fea.Observation.probe`, or `fea.Observation.reduction`. The observation is attached to a step through `fea.ProcedureStep.configure`, and model validation resolves the target against the analysis model before any solve work starts.",
        "Constraint targets are used for reaction forces and moments and for constraint-violation diagnostics at `fea.FieldLocation.constraint()`.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when the id is empty or whitespace (`target id must not be empty`).",
        "`RunMat:fea:Internal` if the target cannot be converted into a RunMat object; this indicates a runtime defect, not an input problem.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "fixed-base",
        title: "Target the fixed base support",
        program: "target = fea.ObservationTarget.constraint(\"fixed_base\");",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(target), 'fea.ObservationTarget'));\nassert(strcmp(target.value.constraint.constraint_id, 'fixed_base'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ObservationTarget.constraint",
        summary: "Target a named constraint group for an observation.",
        description: "`fea.ObservationTarget.constraint(id)` builds a `fea.ObservationTarget` that refers to one authored constraint group, such as a fixed support, by its identifier.",
        keywords: &["fea", "observation", "target", "constraint", "support", "reaction"],
        related: &["fea.ObservationTarget.wholeModel", "fea.ObservationTarget.connector", "fea.Observation.field", "fea.Observation.history", "fea.Observation.probe", "fea.Observation.reduction"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Can I get the total reaction at a support today?", answer: "Whole-model and exact-node reaction targets are supported now. Constraint-group reaction totals are rejected until the executed constraint plan is available to evaluation." }],
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
