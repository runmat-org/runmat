mod documentation;

use super::support::define_numeric_classification_entry;
use documentation::ISINF_DOCUMENTATION;

define_numeric_classification_entry!(
    entry: ISINF_CATALOG_ENTRY,
    descriptor: ISINF_DESCRIPTOR,
    invalid_error: ISINF_ERROR_INVALID_INPUT,
    internal_error: ISINF_ERROR_INTERNAL,
    output_error: ISINF_ERROR_TOO_MANY_OUTPUTS,
    name: "isinf",
    upper: "ISINF",
    predicate: crate::NumericClassificationPredicate::Infinite,
    documentation: ISINF_DOCUMENTATION,
    input_description: "Numeric, logical, character, or string input to classify element by element.",
    output_description: "Same-shaped logical mask that is true where either numeric component is infinite.",
    integer_notes: "Fixed-width integers cannot represent infinity, so all eight classes produce an exact same-shaped false mask without numeric conversion."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISINF_CATALOG_ENTRY];
