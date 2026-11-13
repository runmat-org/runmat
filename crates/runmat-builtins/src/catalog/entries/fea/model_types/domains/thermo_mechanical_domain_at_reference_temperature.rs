//! Canonical documentation for fea.ThermoMechanicalDomain.atReferenceTemperature; original sidecar prose is preserved.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ThermoMechanicalDomain.atReferenceTemperature",
    summary: "Create a thermo-mechanical domain with a typed constitutive reference temperature.",
    description: "`domain = fea.ThermoMechanicalDomain.atReferenceTemperature(temperature)` creates the constitutive domain for thermo-mechanical procedures.",
    keywords: &["fea", "domain", "thermo-mechanical", "temperature"],
    related: &[],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["`temperature` must be an `fea.Temperature`; bare numeric values are rejected.", "Temperature histories and spatial fields are authored as signals and excitations, not stored in the domain.", "A model accepts at most one thermo-mechanical domain."] }],
    examples: &[BuiltinExample {
        id: "author-thermo-mechanical-domain-at-reference-temperature",
        title: "Create a room-temperature domain",
        program: "domain = fea.ThermoMechanicalDomain.atReferenceTemperature(fea.Temperature.celsius(20));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(domain), 'fea.ThermoMechanicalDomain'));" },
    }],
    faqs: &[],
    links: &[crate::fea_contract::doc_links::GET_STARTED],
    implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/fea/domains.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/domains.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed domain constructor tests", location: "builtins::fea::tests::fea_domain_constructors_preserve_typed_physics_boundaries" }],
    notes: &[],
});
pub const THERMOMECHANICAL_DOMAIN_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::entry::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fea.ThermoMechanicalDomain.atReferenceTemperature",
    DOCUMENTATION,
    &contract::THERMOMECHANICAL_DOMAIN_DESCRIPTOR,
    &[],
    Some(&contract::TYPED_DOMAIN_INTEGER_AUDIT),
);
