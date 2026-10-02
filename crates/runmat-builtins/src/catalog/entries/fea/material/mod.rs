use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    descriptors::FEA_MATERIAL_DESCRIPTOR,
    constructor_inference::{infer_constructor, FeaConstructorKind},
    integer::FEA_MATERIAL_INTEGER_CAPABILITIES,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::Material)
}
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "structural-steel", title: "Create a structural material", program: "steel = fea.material('steel', 'YoungsModulusPa', int64(200000000000), 'PoissonRatio', 0.30);",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(steel), 'fea.Material'));" },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.material", summary: "Create typed RunMat FEA material data.",
    description: "`fea.material(id, Name, Value, ...)` creates mechanical material data with optional thermal, acoustic, electrical, and plastic models.",
    keywords: &["fea", "material", "mechanical", "thermal", "electrical"], related: &["fea.model", "fea.materialAssignment"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "The material id is exact text and the optional name defaults to that id.",
        "Finite real numeric mechanical, thermal, acoustic, electrical, plastic, and electrical frequency-response fields accept all eight built-in integer classes. Exact integers cross one host-only binary64 model-storage boundary and sufficiently wide values can round.",
        "Required fields, nested model shape, finiteness, and unknown fields are validated before return; resident numeric fields are rejected.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Are material properties stored in their input integer class?", answer: "No. Integers remain exact through argument parsing, then cross one binary64 physical-property boundary." }],
    links: &[BuiltinDocumentationLink { label: "FEA models", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/models") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/material.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "FEA material constructors", location: "crates/runmat-runtime/src/builtins/fea/tests/construction.rs" }],
    notes: &["Distributed numeric option handles are rejected: whole-value materialization currently precedes material field validation and cannot preserve failure-before-effect ordering."],
});
pub const FEA_MATERIAL_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.material",
        documentation: DOCUMENTATION,
        descriptor: &FEA_MATERIAL_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &FEA_MATERIAL_INTEGER_CAPABILITIES,
        integer_audit: None,
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_MATERIAL_CATALOG_ENTRY];
