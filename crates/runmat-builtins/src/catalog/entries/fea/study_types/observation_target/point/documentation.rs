use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "The position must be a typed `fea.Position` from `fea.Position.meters([x y z])` or `fea.Position.millimeters([x y z])`; it is stored in meters. Plain numeric vectors are rejected so units cannot be guessed.",
        "A point is a physical location, not the nearest mesh node. RunMat locates the owning Tet4 or Tet10 element during evaluation and interpolates with that element's shape functions. A point outside the admitted solid fails resolution.",
        "Point observations require a nodal source location, `fea.FieldLocation.point()` as the output location, and `fea.RecoveryPolicy.shapeFunctionInterpolation()`; other combinations fail model validation.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the target as the third argument of `fea.Observation.field`, `fea.Observation.history`, `fea.Observation.probe`, or `fea.Observation.reduction`. The observation is attached to a step through `fea.ProcedureStep.configure`, and model validation resolves the target against the analysis model before any solve work starts.",
        "Point targets are the usual choice for `fea.Observation.probe` and for displacement or temperature histories at a sensor location.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `position` is not a `fea.Position` object.",
        "`RunMat:fea:Internal` if the target cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "tip-sensor",
        title: "Target a tip sensor 100 mm from the root",
        program: "tip = fea.Position.millimeters([100 0 5]);\ntarget = fea.ObservationTarget.point(tip);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(target), 'fea.ObservationTarget'));\nassert(isequal(target.value.point.position.meters, [0.1 0 0.005]));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ObservationTarget.point",
        summary: "Target one physical point for an interpolated observation.",
        description: "`fea.ObservationTarget.point(position)` builds a `fea.ObservationTarget` that evaluates a field at one physical position given as a typed `fea.Position`.",
        keywords: &["fea", "observation", "target", "point", "probe", "position"],
        related: &["fea.Position.meters", "fea.ObservationTarget.path", "fea.Observation.probe", "fea.Observation.history"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Why not pass `[0.1 0 0]` directly?", answer: "The target requires a typed `fea.Position` so the unit is explicit. Use `fea.Position.meters` or `fea.Position.millimeters`." }, BuiltinDocumentationFaq { question: "Does RunMat snap to the nearest node?", answer: "No. The value is interpolated inside the owning element with its shape functions." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/targets.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/observation_target/tests.rs::observation_target_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
