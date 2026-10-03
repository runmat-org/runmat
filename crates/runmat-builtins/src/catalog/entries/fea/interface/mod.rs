use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    constructor_inference::{infer_constructor, FeaConstructorKind},
    descriptors::FEA_INTERFACE_DESCRIPTOR,
    integer::FEA_INTERFACE_INTEGER_CAPABILITIES,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::Interface)
}
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "contact-interface", title: "Create a contact interface", program: "contact = fea.interface('joint', 'face:a', 'face:b', 'Kind', 'contact', 'FrictionCoefficient', uint8(0));",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(contact), 'fea.Interface'));" },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.interface", summary: "Create a typed RunMat FEA interface.",
    description: "`fea.interface(id, primaryRegion, secondaryRegion, Name, Value, ...)` creates contact, fluid-structure, or conjugate heat-transfer interface data.",
    keywords: &["fea", "interface", "contact", "fsi", "cht"], related: &["fea.model", "fea.domain"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Interface id and both region selectors remain exact text; Kind defaults to contact.",
        "Contact fields are penalty stiffness scale, maximum penetration ratio, and friction coefficient; FSI fields are normal stiffness, damping ratio, and relaxation factor; CHT fields are thermal conductance, contact resistance, and relaxation factor.",
        "All eight integer classes are accepted for finite real numeric fields and cross one host-only binary64 model-storage boundary after field and shape validation. Unknown, nonfinite, nonscalar, and resident numeric fields are rejected.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Does interface construction use the GPU?", answer: "No. It creates host model data; backend execution happens only in a later solve." }],
    links: &[BuiltinDocumentationLink { label: "FEA models", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/models") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/domain_interface.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Interface constructor", location: "crates/runmat-runtime/src/builtins/fea/tests/construction.rs" }],
    notes: &["Distributed numeric option handles are rejected before any materialization; interface kind and field schema must be validated first."],
});
pub const FEA_INTERFACE_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.interface",
        documentation: DOCUMENTATION,
        descriptor: &FEA_INTERFACE_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &FEA_INTERFACE_INTEGER_CAPABILITIES,
        integer_audit: None,
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_INTERFACE_CATALOG_ENTRY];
