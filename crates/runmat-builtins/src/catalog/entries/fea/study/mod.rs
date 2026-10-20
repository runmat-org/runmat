use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    constructor_inference::{infer_constructor, FeaConstructorKind},
    descriptors::FEA_STUDY_DESCRIPTOR,
    example_fixtures::CLOSED_CUBE_STUDY,
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::Study)
}
const STUDY_PROGRAM: &str = concat!(
    "geom = geometry.load('cube.stl');\n",
    "steel = fea.material('steel', 'YoungsModulusPa', int64(200000000000), 'PoissonRatio', 0.3);\n",
    "target = fea.PropertyRegionTarget.allSolids(geom);\n",
    "assignment = fea.materialAssignment(target, steel);\n",
    "region = geometry.region(geom, 'region_default');\n",
    "fixed = fea.StructuralConstraint.fixed('fixed', region);\n",
    "step = fea.StructuralStep.static('static');\n",
    "model = fea.model('cube_model', geom, 'Materials', {steel}, 'MaterialAssignments', {assignment}, 'Constraints', {fixed}, 'Steps', {step});\n",
    "study = fea.study('closed_cube_static', geom, model);\n",
);
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "construct-study",
        title: "Create a study from a geometry asset",
        program: STUDY_PROGRAM,
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: CLOSED_CUBE_STUDY,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(class(study), 'fea.Study'));",
        },
    },
    BuiltinExample {
        id: "explicit-model-study",
        title: "Keep the explicit model and geometry identity",
        program: STUDY_PROGRAM,
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: CLOSED_CUBE_STUDY,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(class(study), 'fea.Study')); assert(strcmp(study.model.model_id, model.model_id)); assert(isequal(study.geometry.revision, geom.revision));",
        },
    },
];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.study", summary: "Create a typed FEA study.",
    description: "`fea.study(id, geometry, model, Name, Value, ...)` requires an explicit geometry.Asset and fea.Model.",
    keywords: &["fea", "study"], related: &["fea.model", "fea.sweep", "fea.run"],
    sections: &[BuiltinDocumentationSection { heading: "Typed boundary", paragraphs: &[
        "Mesh and Execution are the only options. Mesh is a jointly validated meshing output; Execution is a typed fea.ExecutionPlan validated against the model sequence. A one-path loader form is not supported.",
        "Host structural identities, revisions, shapes, and counts remain exact. Physics conversions occur only at their explicit owning boundaries; resident inputs are not implicitly gathered.",
    ] }], examples: EXAMPLES, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "FEA API", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/using-fea") }],
    implementation: &[BuiltinDocumentationLink { label: "Current implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed Runtime FEA contracts", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::tests::typed_constructors_build_full_study_and_sweep_objects" }],
    notes: &["Catalog examples require execution qualification; these declarations are not solver or numerical conformance receipts."],
});
pub const FEA_STUDY_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.study",
        documentation: DOCUMENTATION,
        descriptor: &FEA_STUDY_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        effects: OperationEffectProfile::ArtifactWrite,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_STUDY_CATALOG_ENTRY];
