//! Canonical documentation for fea.ElectromagneticDomain.atFrequency; original sidecar prose is preserved.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ElectromagneticDomain.atFrequency",
    summary: "Create an electromagnetic frequency-domain definition.",
    description: "`domain = fea.ElectromagneticDomain.atFrequency(frequency)` creates an electromagnetic domain at a positive typed frequency.",
    keywords: &["fea", "domain", "electromagnetic", "frequency"],
    related: &[],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["`frequency` must be an `fea.Frequency`; bare numeric values are rejected.", "Currents and current densities are authored as signals and excitations.", "A model accepts at most one electromagnetic domain."] }],
    examples: &[BuiltinExample {
        id: "author-electromagnetic-domain-at-frequency",
        title: "Create a 60 Hz domain",
        program: "domain = fea.ElectromagneticDomain.atFrequency(fea.Frequency.hertz(60));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(domain), 'fea.ElectromagneticDomain'));" },
    }],
    faqs: &[],
    links: &[crate::fea_contract::doc_links::GET_STARTED],
    implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/fea/domains.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/domains.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed domain constructor tests", location: "builtins::fea::tests::fea_domain_constructors_preserve_typed_physics_boundaries" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Analysis integration tests", location: "tests/analysis" }],
    notes: &[],
});
pub const ELECTROMAGNETIC_DOMAIN_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::entry::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fea.ElectromagneticDomain.atFrequency",
    DOCUMENTATION,
    &contract::ELECTROMAGNETIC_DOMAIN_DESCRIPTOR,
    &[],
    Some(&contract::TYPED_DOMAIN_INTEGER_AUDIT),
);
