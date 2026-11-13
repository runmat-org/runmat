//! Canonical documentation for fea.MassDensity.kilogramsPerCubicMeter; original sidecar prose is preserved.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.MassDensity.kilogramsPerCubicMeter",
    summary: "Create a positive typed mass density in kilograms per cubic meter.",
    description: "`density = fea.MassDensity.kilogramsPerCubicMeter(value)` creates an `fea.MassDensity` for material and fluid-domain definitions.",
    keywords: &["fea", "quantity", "density", "fluid"],
    related: &[],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The input must be a finite positive real scalar.", "Host integer inputs cross one checked binary64 storage boundary; resident values are rejected."] }],
    examples: &[BuiltinExample {
        id: "author-mass-density-kilograms-per-cubic-meter",
        title: "Create the density of air",
        program: "density = fea.MassDensity.kilogramsPerCubicMeter(1.225);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(density), 'fea.MassDensity')); assert(density.kilograms_per_cubic_meter == 1.225);" },
    }],
    faqs: &[],
    links: &[],
    implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/fea/values.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/values.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed quantity constructor tests", location: "builtins::fea::tests::fea_domain_constructors_preserve_typed_physics_boundaries" }],
    notes: &[],
});
pub const MASS_DENSITY_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::entry::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fea.MassDensity.kilogramsPerCubicMeter",
    DOCUMENTATION,
    &contract::MASS_DENSITY_DESCRIPTOR,
    &contract::QUANTITY_INTEGER_CAPABILITIES,
    None,
);
