mod contract;
mod declarations;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use declarations::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(declarations::ENTRIES.iter().copied());
}
