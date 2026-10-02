use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    descriptors::FEA_MATERIAL_ASSIGNMENT_DESCRIPTOR,
    constructor_inference::{infer_constructor, FeaConstructorKind},
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
    program: "assignment = fea.materialAssignment('name:Bracket', 'steel');",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(assignment), 'fea.MaterialAssignment'));",
    },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.materialAssignment", summary: "Assign a RunMat FEA material to a region.",
    description: "`fea.materialAssignment(region, material, Name, Value, ...)` creates an exact region-to-material assignment with optional expected-material and evidence-confidence metadata.",
    keywords: &["fea", "material", "assignment", "region"], related: &["fea.material", "fea.model"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Region, assigned material, and optional expected material are exact text identifiers. Confidence is an enumerated structural value and defaults to verified.",
        "No numeric field exists; integer and resident numeric inputs are rejected before assignment construction.",
        "Region existence and material consistency are checked when the assignment is assembled into or validated with a model.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Does construction prove the region exists?", answer: "No. Construction preserves the selector; model assembly and validation resolve it against geometry." }],
    links: &[BuiltinDocumentationLink { label: "FEA models", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/models") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/material.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "FEA constructor tests", location: "crates/runmat-runtime/src/builtins/fea/tests/construction.rs" }],
    notes: &["Region and material selectors are local structural text; execution-owned distributed handles are not valid selectors."],
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
