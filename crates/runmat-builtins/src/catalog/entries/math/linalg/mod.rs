mod matrix_arithmetic;

pub use matrix_arithmetic::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(matrix_arithmetic::ENTRIES.iter().copied());
}
