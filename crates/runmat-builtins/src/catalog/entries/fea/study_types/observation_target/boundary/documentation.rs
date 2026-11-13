use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::example_fixtures::BOX_STUDY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "The region must be a `geometry.Region`, typically from `geometry.region(asset, selector)`. RunMat reads its persistent region id and records a face (boundary) target with one-or-more cardinality; the geometry asset itself is not copied into the target.",
        "Boundary targets measure area. A spatial `integral` or `resultant` reduction over a boundary target multiplies the source quantity by area, so a pressure integral becomes a force and a stress resultant is converted through the Cauchy traction.",
        "The constructor is pure and allocates only a small host object. It does not mesh the geometry or check that the region exists in a model; that happens when the model is validated.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the target as the third argument of `fea.Observation.field`, `fea.Observation.history`, `fea.Observation.probe`, or `fea.Observation.reduction`. The observation is attached to a step through `fea.ProcedureStep.configure`, and model validation resolves the target against the analysis model before any solve work starts.",
        "Typical uses are surface displacement fields, contact or traction resultants on a loaded face, and heat flux through a wall. Meshing rewrites boundary references to persistent source-topology ids together with the constraints and excitations that use the same region.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:geometry:region:InvalidRegion` when `region` is not a valid `geometry.Region` object.",
        "`RunMat:fea:Internal` if the target cannot be converted into a RunMat object; this indicates a runtime defect, not an input problem.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "loaded-face",
        title: "Observe the boundary of a loaded region",
        program: "part = geometry.load(\"box.brep\", \"Units\", \"m\");\ntipFace = geometry.region(part, \"face_000002\");\ntarget = fea.ObservationTarget.boundary(tipFace);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BOX_STUDY,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(target), 'fea.ObservationTarget'));\nassert(strcmp(target.value.spatial.region.region_id, 'face_000002'));\nassert(strcmp(target.value.spatial.region.entity_dimension, 'face'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ObservationTarget.boundary",
        summary: "Target the boundary faces of a geometry region for an observation.",
        description: "`fea.ObservationTarget.boundary(region)` builds a `fea.ObservationTarget` that evaluates an observation on the boundary faces of a typed `geometry.Region`.",
        keywords: &["fea", "observation", "target", "boundary", "surface", "region"],
        related: &["fea.ObservationTarget.volume", "geometry.region", "fea.Observation.field", "fea.Observation.history", "fea.Observation.probe", "fea.Observation.reduction"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Does the target keep a copy of the geometry?", answer: "No. It keeps the persistent region id. The model's geometry supplies the faces when the target is resolved." }, BuiltinDocumentationFaq { question: "How is this different from `fea.ObservationTarget.volume`?", answer: "A boundary target evaluates on the region's boundary faces and measures area; a volume target evaluates on its cells and measures volume." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
            BuiltinDocumentationLink { label: "Geometry", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/geometry") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/targets.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/observation_target/tests.rs::observation_target_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
