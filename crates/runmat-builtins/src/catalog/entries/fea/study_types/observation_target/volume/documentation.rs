use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::study_types::fixtures::STUDY_TYPES_CUBE;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "The region must be a `geometry.Region`. RunMat records its persistent region id as a volume (mesh-element) target with one-or-more cardinality.",
        "Volume targets measure volume. A spatial `integral` reduction integrates the source quantity over the region with the admitted element quadrature, and `average`, `rootMeanSquare`, and `euclideanNorm` use the same quadrature.",
        "The constructor is pure and host-only; region existence is checked during model validation, not here.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the target as the third argument of `fea.Observation.field`, `fea.Observation.history`, `fea.Observation.probe`, or `fea.Observation.reduction`. The observation is attached to a step through `fea.ProcedureStep.configure`, and model validation resolves the target against the analysis model before any solve work starts.",
        "Use a volume target for peak von Mises stress in a part, strain energy in a component, or the average temperature of a solid body.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:geometry:region:InvalidRegion` when `region` is not a valid `geometry.Region` object.",
        "`RunMat:fea:Internal` if the target cannot be converted into a RunMat object; this indicates a runtime defect, not an input problem.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "solid-body",
        title: "Observe a solid region",
        program: "geom = geometry.load(\"cube.stl\", \"Units\", \"meter\");\nbody = geometry.region(geom, \"name:Default Region\");\ntarget = fea.ObservationTarget.volume(body);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: STUDY_TYPES_CUBE,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(target), 'fea.ObservationTarget'));\nassert(strcmp(target.value.spatial.region.entity_dimension, 'volume'));\nassert(strcmp(target.value.spatial.region.expected_topology, 'mesh_element'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ObservationTarget.volume",
        summary: "Target the volume cells of a geometry region for an observation.",
        description: "`fea.ObservationTarget.volume(region)` builds a `fea.ObservationTarget` that evaluates an observation over the mesh cells of a typed `geometry.Region`.",
        keywords: &["fea", "observation", "target", "volume", "solid", "region"],
        related: &["fea.ObservationTarget.boundary", "geometry.region", "fea.Observation.field", "fea.Observation.history", "fea.Observation.probe", "fea.Observation.reduction"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Can I observe several bodies at once?", answer: "Select a region that contains all of them, or declare one observation per body. Each target names exactly one region." }],
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
