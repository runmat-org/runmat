mod analysis;

pub use self::analysis::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(analysis::ENTRIES.iter().copied());
}
