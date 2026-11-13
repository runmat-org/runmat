//! Canonical documentation for fea.ElectroThermalDomain.atReferenceTemperature; original sidecar prose is preserved.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ElectroThermalDomain.atReferenceTemperature",
    summary: "Create an electro-thermal domain with a typed constitutive reference temperature.",
    description: "`domain = fea.ElectroThermalDomain.atReferenceTemperature(temperature)` creates the constitutive domain for electro-thermal procedures.",
    keywords: &["fea", "domain", "electro-thermal", "temperature"],
    related: &[],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["`temperature` must be an `fea.Temperature`; bare numeric values are rejected.", "Electrical and thermal excitations are authored independently from the domain.", "A model accepts at most one electro-thermal domain."] }],
    examples: &[BuiltinExample {
        id: "author-electro-thermal-domain-at-reference-temperature",
        title: "Create the domain",
        program: "domain = fea.ElectroThermalDomain.atReferenceTemperature(fea.Temperature.kelvin(293.15));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(domain), 'fea.ElectroThermalDomain'));" },
    }],
    faqs: &[],
    links: &[crate::fea_contract::doc_links::GET_STARTED],
    implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/fea/domains.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/domains.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed domain constructor tests", location: "builtins::fea::tests::fea_domain_constructors_preserve_typed_physics_boundaries" }],
    notes: &[],
});
pub const ELECTROTHERMAL_DOMAIN_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::entry::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fea.ElectroThermalDomain.atReferenceTemperature",
    DOCUMENTATION,
    &contract::ELECTROTHERMAL_DOMAIN_DESCRIPTOR,
    &[],
    Some(&contract::TYPED_DOMAIN_INTEGER_AUDIT),
);
