mod descriptor;
mod documentation;
mod entry;
mod extensions;
mod inference;
mod integer;
#[cfg(test)]
mod tests;
pub use super::window_facts::{WindowOption, WindowOutputType, WindowSampling};
pub use descriptor::*;
pub use entry::HAMMING_CATALOG_ENTRY;
pub use extensions::*;
pub use integer::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&HAMMING_CATALOG_ENTRY];
