pub(in crate::catalog) mod core;

pub use core::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    core::extend_entries(entries);
}
