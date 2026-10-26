//! Canonical documentation for fea.ContactInterface.coulomb; original sidecar prose is preserved.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ContactInterface.coulomb",
    summary: "Join two geometry regions with Coulomb friction.",
    description: "Creates contact with a positive typed friction coefficient between two regions from one geometry revision.",
    keywords: &["fea", "interface", "contact", "friction"],
    related: &[],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Requires `fea.FrictionCoefficient`.", "Rejects zero; use `frictionless` when no friction is intended.", "Contact enforcement parameters remain solver policy."] }],
    examples: &[BuiltinExample {
        id: "author-contact-interface-coulomb",
        title: "Create frictional contact",
        program: "geom = geometry.load('interface-patches.obj');\nleft = geometry.region(geom, 'region_left');\nright = geometry.region(geom, 'region_right');\nmu = fea.FrictionCoefficient.ratio(0.2);\ncontact = fea.ContactInterface.coulomb(\"joint\", left, right, mu);",
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
pub const CONTACT_COULOMB_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::entry::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fea.ContactInterface.coulomb",
    DOCUMENTATION,
    &contract::CONTACT_COULOMB_DESCRIPTOR,
    &[],
    Some(&contract::TYPED_INTERFACE_INTEGER_AUDIT),
);
