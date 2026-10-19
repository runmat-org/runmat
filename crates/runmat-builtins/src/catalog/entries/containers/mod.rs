mod map;

pub use self::map::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(map::ENTRIES.iter().copied());
}
