mod descriptor;
mod documentation;
mod entry;
mod extensions;
mod inference;
mod integer;
mod subsref;
#[cfg(test)]
mod tests;

pub const GRIDDED_INTERPOLANT_CLASS_IDENTITY: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("griddedInterpolant");

pub use descriptor::{
    ERROR_INTERNAL as GRIDDED_INTERPOLANT_ERROR_INTERNAL,
    ERROR_INVALID_INPUT as GRIDDED_INTERPOLANT_ERROR_INVALID_INPUT,
    ERROR_UNSUPPORTED_METHOD as GRIDDED_INTERPOLANT_ERROR_UNSUPPORTED_METHOD,
    GRIDDED_INTERPOLANT_DESCRIPTOR, GRIDDED_INTERPOLANT_SUBSREF_DESCRIPTOR,
};
pub use entry::GRIDDED_INTERPOLANT_CATALOG_ENTRY;
pub use extensions::*;
pub use integer::*;
pub use subsref::GRIDDED_INTERPOLANT_SUBSREF_CATALOG_ENTRY;

const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &GRIDDED_INTERPOLANT_CATALOG_ENTRY,
    &GRIDDED_INTERPOLANT_SUBSREF_CATALOG_ENTRY,
];

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(ENTRIES.iter().copied());
}
