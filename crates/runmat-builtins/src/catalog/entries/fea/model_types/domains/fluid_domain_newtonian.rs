//! Canonical documentation for fea.FluidDomain.newtonian; original sidecar prose is preserved.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.FluidDomain.newtonian",
    summary: "Create a Newtonian fluid domain from typed constitutive properties.",
    description: "`domain = fea.FluidDomain.newtonian(density, viscosity, turbulenceIntensity)` creates a fluid domain without embedding a solver procedure or inlet condition.",
    keywords: &["fea", "domain", "fluid", "cfd", "newtonian"],
    related: &[],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Density and dynamic viscosity must be positive typed values; turbulence intensity must be a typed ratio between zero and one.", "Steady versus transient execution is selected by the procedure step.", "Inlet velocity and wall motion are authored as constraints or excitations, not stored in the domain."] }],
    examples: &[BuiltinExample {
        id: "author-fluid-domain-newtonian",
        title: "Create an air domain",
        program: "air = fea.FluidDomain.newtonian(fea.MassDensity.kilogramsPerCubicMeter(1.225), fea.DynamicViscosity.pascalSeconds(1.81e-5), fea.TurbulenceIntensity.ratio(0.05));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(air), 'fea.FluidDomain'));" },
    }],
    faqs: &[],
    links: &[crate::fea_contract::doc_links::GET_STARTED],
    implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/fea/domains.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/domains.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed domain constructor tests", location: "builtins::fea::tests::fea_domain_constructors_preserve_typed_physics_boundaries" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Analysis integration tests", location: "tests/analysis" }],
    notes: &[],
});
pub const FLUID_DOMAIN_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::entry::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fea.FluidDomain.newtonian",
    DOCUMENTATION,
    &contract::FLUID_DOMAIN_DESCRIPTOR,
    &[],
    Some(&contract::TYPED_DOMAIN_INTEGER_AUDIT),
);
