mod documentation;

use super::support::define_numeric_classification_entry;
use documentation::ISNAN_DOCUMENTATION;

define_numeric_classification_entry!(
    entry: ISNAN_CATALOG_ENTRY,
    descriptor: ISNAN_DESCRIPTOR,
    invalid_error: ISNAN_ERROR_INVALID_INPUT,
    internal_error: ISNAN_ERROR_INTERNAL,
    output_error: ISNAN_ERROR_TOO_MANY_OUTPUTS,
    name: "isnan",
    upper: "ISNAN",
    predicate: crate::NumericClassificationPredicate::Nan,
    documentation: ISNAN_DOCUMENTATION,
    input_description: "Numeric, logical, character, or string input to classify element by element.",
    output_description: "Same-shaped logical mask that is true where either numeric component is NaN.",
    integer_notes: "Fixed-width integers cannot represent NaN, so all eight classes produce an exact same-shaped false mask without numeric conversion."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISNAN_CATALOG_ENTRY];
