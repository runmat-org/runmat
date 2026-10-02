use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    constructor_inference::{infer_constructor, FeaConstructorKind},
    descriptors::FEA_AUTHOR_STUDY_DESCRIPTOR,
    example_fixtures::CLOSED_CUBE_STUDY,
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::AuthorStudy)
}
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "reject-incomplete-summary", title: "Reject incomplete mesh evidence before authoring",
    program: "geom = geometry.load('cube.stl');\nstudy = fea.authorStudy('candidate', geom, '{}', 'Profile', 'linear_static_structural');",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem, fixture: CLOSED_CUBE_STUDY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::ExpectedError { identifier: "RunMat:fea:InvalidInput" },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.authorStudy", summary: "Author a typed FEA study from mesh-authoring evidence.",
    description: "`fea.authorStudy(id, geometry, meshAuthoringSummary, Name, Value, ...)` validates a typed geometry asset and solve-ready mesh-authoring summary, builds a model, and persists authoring evidence.",
    keywords: &["fea", "author", "study", "mesh", "evidence"], related: &["fea.study", "fea.run", "fea.validate"],
    sections: &[BuiltinDocumentationSection { heading: "Inputs and effects", paragraphs: &[
        "The first three operands are a host text study id, `geometry.Asset`, and a host mesh-authoring summary object or JSON text. Complete Name, Value pairs follow. Profile is required and selects the solver family; an explicit RunKind must agree with it.",
        "The summary must follow `mesh-authoring-summary/v1`, be solve-ready, and identify an available material region and fully recovered boundary regions for the selected model. Optional region and force choices are validated against it.",
        "Options include Profile, RunKind/Kind, Backend, ModelId, MaterialRegion/MaterialRegionId, BoundaryConditionRegion aliases, DrivingConditionRegion aliases, StructuralForceN aliases, AnalysisMeshArtifactPath, AnalysisMeshEvidenceArtifactPath, and DiagramObservation.",
        "Authoring creates a typed `fea.Study` and writes evidence but does not solve it. Backend is stored for later execution; this builtin itself performs no numeric provider dispatch. Structural mesh counts stay exact integers; force and quality measurements are physical floating values.",
        "Already loaded STL geometry needs no OCCT; topology-bearing CAD can require its configured importer before this call. Resident or distributed geometry/summary handles are not implicitly gathered.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Does authoring solve the study?", answer: "No. It constructs and validates a study and records authoring evidence; pass it to fea.run to solve." }],
    links: &[BuiltinDocumentationLink { label: "Using FEA", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/using-fea") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/author_study.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Authoring summary and persisted mesh", location: "crates/runmat-runtime/src/builtins/fea/author_study/tests.rs" }],
    notes: &["A solve-ready mesh-authoring summary must describe the same authored mesh and regions supplied for the study; an arbitrary JSON object is not sufficient."],
});
pub const FEA_AUTHOR_STUDY_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.authorStudy",
        documentation: DOCUMENTATION,
        descriptor: &FEA_AUTHOR_STUDY_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        effects: OperationEffectProfile::EvidenceWrite,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_AUTHOR_STUDY_CATALOG_ENTRY];
