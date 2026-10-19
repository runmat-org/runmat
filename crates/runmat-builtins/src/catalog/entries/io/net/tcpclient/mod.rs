mod contract;
mod entry;
mod documentation;
mod inference;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&TCPCLIENT_CATALOG_ENTRY];
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::TCPCLIENT_CATALOG_ENTRY;
