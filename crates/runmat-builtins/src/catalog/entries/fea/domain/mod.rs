use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    descriptors::FEA_DOMAIN_DESCRIPTOR,
    inference::{infer_constructor, FeaConstructorKind},
    integer::FEA_DOMAIN_INTEGER_CAPABILITIES,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::Domain)
}
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "cfd-domain", title: "Create a CFD domain", program: "domain = fea.domain('cfd', 'ReferenceDensityKgPerM3', uint16(1), 'InletVelocityMPerS', int16(12));",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(domain), 'fea.Domain'));" },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.domain", summary: "Create a typed RunMat FEA physics domain.",
    description: "`fea.domain(kind, Name, Value, ...)` creates thermo-mechanical, electro-thermal, electromagnetic, or CFD domain data for a RunMat FEA model.",
    keywords: &["fea", "domain", "thermal", "electromagnetic", "cfd"], related: &["fea.model", "fea.interface"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The domain kind selects supported fields; unknown fields are rejected before construction. Resident numeric fields are rejected; no provider execution occurs.", "Finite real physical fields accept all eight built-in integer classes and cross one host-only binary64 model-storage conversion. A nested thermo-field-source revision is a structural nonnegative integer and stays exact."] },
        BuiltinDocumentationSection { heading: "Integer fields", paragraphs: &["Thermo-mechanical reference temperature, delta, regional deltas, and time coordinates/scales; electro-thermal reference temperature, voltage, conductivity scales, and time coordinates/scales; electromagnetic frequency and current; CFD density, viscosity, inlet velocity, turbulence intensity, and time coordinates/scales. Thermo field-source revision is exact structural identity."] },
    ], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Does `revision` become binary64?", answer: "No. Revision is structural model identity validated and stored as an exact nonnegative integer." }],
    links: &[BuiltinDocumentationLink { label: "Using FEA", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/using-fea") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/domain_interface.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Domain constructor", location: "crates/runmat-runtime/src/builtins/fea/tests/construction.rs" }],
    notes: &["Legacy requires_feature is obsolete; nested revision remains separate from physical integer conversion."],
});
pub const FEA_DOMAIN_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.domain",
        documentation: DOCUMENTATION,
        descriptor: &FEA_DOMAIN_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &FEA_DOMAIN_INTEGER_CAPABILITIES,
        integer_audit: None,
        filesystem_read: false,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_DOMAIN_CATALOG_ENTRY];
