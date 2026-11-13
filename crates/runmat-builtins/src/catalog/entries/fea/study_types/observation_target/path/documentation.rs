use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`positions` must be a host numeric matrix with exactly three columns (x, y, z in meters) and at least two rows. Each row becomes one ordered `fea.Position`; the order is retained with the result.",
        "Integer-class matrices are accepted and converted once to binary64 meters. Every coordinate must be finite. Whether a position lies inside the model is checked during evaluation, where a point outside the admitted solid fails.",
        "Path observations require a nodal source location, `fea.FieldLocation.point()` output, and `fea.RecoveryPolicy.shapeFunctionInterpolation()`. RunMat interpolates each position in its owning Tet4 or Tet10 element and retains the owning element with the result.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the target as the third argument of `fea.Observation.field`, `fea.Observation.history`, `fea.Observation.probe`, or `fea.Observation.reduction`. The observation is attached to a step through `fea.ProcedureStep.configure`, and model validation resolves the target against the analysis model before any solve work starts.",
        "Use a path for through-thickness stress profiles, temperature along a centerline, or deflection along a beam.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `positions` is not a host numeric matrix, is not N-by-3 with N at least two, or contains NaN or Inf.",
        "`RunMat:fea:Internal` if the target cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "centerline",
        title: "Sample a 20 mm centerline every 10 mm",
        program: "positions = [0.00 0 0; 0.01 0 0; 0.02 0 0];\ntarget = fea.ObservationTarget.path(positions);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(target), 'fea.ObservationTarget'));\nassert(numel(target.value.path.positions) == 3);\nassert(isequal(target.value.path.positions{3}.meters, [0.02 0 0]));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ObservationTarget.path",
        summary: "Target an ordered polyline of physical positions for an observation.",
        description: "`fea.ObservationTarget.path(positions)` builds a `fea.ObservationTarget` that evaluates a field at each row of an N-by-3 matrix of positions in meters, in row order.",
        keywords: &["fea", "observation", "target", "path", "line", "profile"],
        related: &["fea.ObservationTarget.point", "fea.Observation.field", "fea.Observation.probe"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "What units are the coordinates in?", answer: "Meters. Scale millimeter coordinates by 1e-3 before calling `path`." }, BuiltinDocumentationFaq { question: "Is a single point accepted?", answer: "No. A path needs at least two rows; use `fea.ObservationTarget.point` for one location." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/targets.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/observation_target/tests.rs::observation_target_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Finite path positions", location: "crates/runmat-runtime/src/builtins/fea/observations/tests.rs::path_targets_reject_nonfinite_coordinates" },
        ],
        notes: &[],
    });
