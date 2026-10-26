//! Canonical documentation for fea.TurbulenceIntensity.ratio; original sidecar prose is preserved.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.TurbulenceIntensity.ratio",
    summary: "Create a typed turbulence-intensity ratio.",
    description: "`intensity = fea.TurbulenceIntensity.ratio(value)` creates an `fea.TurbulenceIntensity` between zero and one.",
    keywords: &["fea", "quantity", "turbulence", "fluid", "ratio"],
    related: &[],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The input must be a finite scalar in the closed interval from zero to one.", "Host integer inputs cross one checked binary64 storage boundary; resident values are rejected."] }],
    examples: &[BuiltinExample {
        id: "author-turbulence-intensity-ratio",
        title: "Create five percent turbulence intensity",
        program: "intensity = fea.TurbulenceIntensity.ratio(0.05);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(intensity), 'fea.TurbulenceIntensity')); assert(intensity.ratio == 0.05);" },
    }],
    faqs: &[],
    links: &[],
    implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/fea/values.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/values.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed quantity constructor tests", location: "builtins::fea::tests::fea_domain_constructors_preserve_typed_physics_boundaries" }],
    notes: &[],
});
pub const TURBULENCE_INTENSITY_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::entry::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fea.TurbulenceIntensity.ratio",
    DOCUMENTATION,
    &contract::TURBULENCE_INTENSITY_DESCRIPTOR,
    &contract::QUANTITY_INTEGER_CAPABILITIES,
    None,
);
