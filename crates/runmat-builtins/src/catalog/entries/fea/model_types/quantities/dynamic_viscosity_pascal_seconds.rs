//! Canonical documentation for fea.DynamicViscosity.pascalSeconds; original sidecar prose is preserved.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.DynamicViscosity.pascalSeconds",
    summary: "Create a positive typed dynamic viscosity in pascal-seconds.",
    description: "`viscosity = fea.DynamicViscosity.pascalSeconds(value)` creates an `fea.DynamicViscosity` for a Newtonian fluid domain.",
    keywords: &["fea", "quantity", "viscosity", "fluid"],
    related: &[],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The input must be a finite positive real scalar.", "Host integer inputs cross one checked binary64 storage boundary; resident values are rejected."] }],
    examples: &[BuiltinExample {
        id: "author-dynamic-viscosity-pascal-seconds",
        title: "Create the viscosity of air",
        program: "viscosity = fea.DynamicViscosity.pascalSeconds(1.81e-5);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(viscosity), 'fea.DynamicViscosity')); assert(viscosity.pascal_seconds == 1.81e-5);" },
    }],
    faqs: &[],
    links: &[],
    implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/fea/values.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/values.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed quantity constructor tests", location: "builtins::fea::tests::fea_domain_constructors_preserve_typed_physics_boundaries" }],
    notes: &[],
});
pub const DYNAMIC_VISCOSITY_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::entry::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fea.DynamicViscosity.pascalSeconds",
    DOCUMENTATION,
    &contract::DYNAMIC_VISCOSITY_DESCRIPTOR,
    &contract::QUANTITY_INTEGER_CAPABILITIES,
    None,
);
