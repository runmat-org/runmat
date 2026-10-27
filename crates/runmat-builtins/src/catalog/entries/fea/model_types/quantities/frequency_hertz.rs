//! Canonical documentation for fea.Frequency.hertz; original sidecar prose is preserved.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.Frequency.hertz",
    summary: "Create a positive typed frequency in hertz.",
    description: "`frequency = fea.Frequency.hertz(value)` creates an `fea.Frequency` for frequency-domain procedures and signals.",
    keywords: &["fea", "quantity", "frequency", "hertz"],
    related: &[],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The input must be a finite positive real scalar.", "Host integer inputs cross one checked binary64 storage boundary; resident values are rejected."] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &["Pass a frequency to fea.ElectromagneticDomain.atFrequency, or as the sample argument in fea.field(results, selector, frequency) for a frequency-domain result."] },
    ],
    examples: &[BuiltinExample {
        id: "author-frequency-hertz",
        title: "Create 60 Hz",
        program: "frequency = fea.Frequency.hertz(60);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(frequency), 'fea.Frequency')); assert(frequency.hertz == 60);" },
    }],
    faqs: &[],
    links: &[],
    implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/fea/values.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/values.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed quantity constructor tests", location: "builtins::fea::tests::fea_numeric_constructors_cross_all_integer_classes_once_into_binary64" }],
    notes: &[],
});
pub const FREQUENCY_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::entry::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fea.Frequency.hertz",
    DOCUMENTATION,
    &contract::FREQUENCY_DESCRIPTOR,
    &contract::QUANTITY_INTEGER_CAPABILITIES,
    None,
);
