mod contract;
mod documentation;
mod entry;
mod gpu_callback;
mod inference;

#[cfg(test)]
mod inference_tests;

use crate::BuiltinCatalogEntry;

pub use contract::*;
pub use entry::ARRAYFUN_CATALOG_ENTRY;
pub use gpu_callback::*;

pub(in crate::catalog) use inference::infer;
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&ARRAYFUN_CATALOG_ENTRY];
