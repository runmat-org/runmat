mod categorical;
mod ordinal;
mod dictionary;
mod table;
mod entry;
mod facts;
mod options;
#[cfg(test)]
mod tests;
pub use categorical::*;
pub use dictionary::*;
pub use options::*;
pub use ordinal::*;
pub use table::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &CATEGORICAL_CATALOG_ENTRY,
    &ORDINAL_CATALOG_ENTRY,
    &DICTIONARY_CATALOG_ENTRY,
    &TABLE_CATALOG_ENTRY,
];
