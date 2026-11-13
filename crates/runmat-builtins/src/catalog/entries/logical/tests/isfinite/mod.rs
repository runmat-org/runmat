mod documentation;

use super::support::define_numeric_classification_entry;
use documentation::ISFINITE_DOCUMENTATION;

define_numeric_classification_entry!(
    entry: ISFINITE_CATALOG_ENTRY,
    descriptor: ISFINITE_DESCRIPTOR,
    invalid_error: ISFINITE_ERROR_INVALID_INPUT,
    internal_error: ISFINITE_ERROR_INTERNAL,
    output_error: ISFINITE_ERROR_TOO_MANY_OUTPUTS,
    name: "isfinite",
    upper: "ISFINITE",
    predicate: crate::NumericClassificationPredicate::Finite,
    documentation: ISFINITE_DOCUMENTATION,
    input_description: "Numeric, logical, character, or string input to classify element by element.",
    output_description: "Same-shaped logical mask that is true where both numeric components are finite.",
    integer_notes: "Every fixed-width integer element is finite, so all eight classes produce an exact same-shaped true mask without numeric conversion."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISFINITE_CATALOG_ENTRY];
