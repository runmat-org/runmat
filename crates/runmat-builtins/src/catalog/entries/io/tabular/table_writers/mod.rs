mod writetable;
mod writetimetable;
mod entry;
mod inference;
mod options;
mod integer;
#[cfg(test)]
mod tests;
pub use integer::*;
pub use options::*;
pub use writetable::*;
pub use writetimetable::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&WRITETABLE_CATALOG_ENTRY, &WRITETIMETABLE_CATALOG_ENTRY];
