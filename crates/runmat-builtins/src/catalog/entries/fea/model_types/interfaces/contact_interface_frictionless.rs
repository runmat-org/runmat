//! Canonical documentation for fea.ContactInterface.frictionless; original sidecar prose is preserved.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ContactInterface.frictionless",
    summary: "Join two geometry regions with frictionless contact.",
    description: "Creates immutable frictionless contact physics between two distinct regions from one admitted geometry revision.",
    keywords: &["fea", "interface", "contact", "frictionless"],
    related: &[],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Rejects regions from different geometry revisions.", "Retains geometry identity for portable execution.", "Contact enforcement parameters are solver policy, not interface inputs."] }],
    examples: &[BuiltinExample {
        id: "author-contact-interface-frictionless",
        title: "Create frictionless contact",
        program: "geom = geometry.load('interface-patches.obj');\nleft = geometry.region(geom, 'region_left');\nright = geometry.region(geom, 'region_right');\ncontact = fea.ContactInterface.frictionless(\"joint\", left, right);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: super::fixture::PATCHES,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(contact), 'fea.ContactInterface'));" },
    }],
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "FEA models", target: BuiltinDocumentationLinkTarget::Documentation("../../fea/models") }],
    implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/fea/interfaces.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/interfaces.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Geometry-bound interface constructor tests", location: "builtins::fea::interfaces::tests" }],
    notes: &[],
});
pub const CONTACT_FRICTIONLESS_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::entry::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fea.ContactInterface.frictionless",
    DOCUMENTATION,
    &contract::CONTACT_FRICTIONLESS_DESCRIPTOR,
    &[],
    Some(&contract::TYPED_INTERFACE_INTEGER_AUDIT),
);
