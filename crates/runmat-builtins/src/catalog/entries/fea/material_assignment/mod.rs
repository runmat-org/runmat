use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    constructor_inference::{infer_constructor, FeaConstructorKind},
    descriptors::FEA_MATERIAL_ASSIGNMENT_DESCRIPTOR,
    example_fixtures::CLOSED_CUBE_STUDY,
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::MaterialAssignment)
}
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "assign-steel",
    title: "Assign steel to a named region",
    program: "geom = geometry.load('cube.stl');\nsteel = fea.material('steel', 'YoungsModulusPa', 200e9, 'PoissonRatio', 0.3);\ntarget = fea.PropertyRegionTarget.allSolids(geom);\nassignment = fea.materialAssignment(target, steel);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: CLOSED_CUBE_STUDY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(assignment), 'fea.MaterialAssignment'));",
    },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.materialAssignment", summary: "Assign a typed FEA material to a geometry-bound target.",
    description: "`fea.materialAssignment(target, material, Name, Value, ...)` requires a fea.PropertyRegionTarget and fea.Material, not text replacements.",
    keywords: &["fea", "material_assignment"], related: &["fea.model", "fea.study", "fea.results"],
    sections: &[BuiltinDocumentationSection { heading: "Typed boundary", paragraphs: &[
        "ExpectedMaterial/ExpectedMaterialId also requires a typed material. Confidence is structural text and defaults to verified. Model assembly checks the target geometry identity and revision.",
        "Host structural identities, revisions, shapes, and counts remain exact. Physics conversions occur only at their explicit owning boundaries; resident inputs are not implicitly gathered.",
    ] }], examples: EXAMPLES, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "FEA API", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/using-fea") }],
    implementation: &[BuiltinDocumentationLink { label: "Current implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed Runtime FEA contracts", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::tests::typed_constructors_build_full_study_and_sweep_objects" }],
    notes: &["Catalog examples require execution qualification; these declarations are not solver or numerical conformance receipts."],
});
pub const FEA_MATERIAL_ASSIGNMENT_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.materialAssignment",
        documentation: DOCUMENTATION,
        descriptor: &FEA_MATERIAL_ASSIGNMENT_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_MATERIAL_ASSIGNMENT_CATALOG_ENTRY];
