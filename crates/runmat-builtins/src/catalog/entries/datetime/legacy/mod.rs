mod contracts;
mod declarations;
mod documentation;
mod inference;
mod integer;
#[cfg(test)]
mod tests;

pub use self::declarations::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(declarations::ENTRIES.iter().copied());
}
