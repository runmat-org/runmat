mod all;
mod any;
mod support;

pub use all::*;
pub use any::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(all::ENTRIES.iter().copied());
    values.extend(any::ENTRIES.iter().copied());
}
