mod ppval;

pub use ppval::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(ppval::ENTRIES.iter().copied());
}
