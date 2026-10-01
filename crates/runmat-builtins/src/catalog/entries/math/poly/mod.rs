mod polyint;
mod roots;

pub use polyint::{
    POLYINT_CATALOG_ENTRY, POLYINT_DESCRIPTOR, POLYINT_ERROR_INVALID_ARGUMENT,
    POLYINT_ERROR_INVALID_INPUT, POLYINT_INTEGER_CAPABILITIES,
};
pub use roots::{
    ROOTS_CATALOG_ENTRY, ROOTS_DESCRIPTOR, ROOTS_ERROR_INTERNAL, ROOTS_ERROR_INVALID_INPUT,
    ROOTS_EXTENSIONS, ROOTS_INTEGER_CAPABILITIES, ROOTS_INTEGER_COEFFICIENTS_EXTENSION,
};

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(polyint::ENTRIES.iter().copied());
    values.extend(roots::ENTRIES.iter().copied());
}
