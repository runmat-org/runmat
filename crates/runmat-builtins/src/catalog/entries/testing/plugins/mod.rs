//! Testing plugin constructors. The generated testing parent owns composition.

mod declarations;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub use declarations::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(declarations::ENTRIES.iter().copied());
}
