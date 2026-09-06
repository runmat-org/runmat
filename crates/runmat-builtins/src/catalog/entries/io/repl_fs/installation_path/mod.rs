mod inference;
mod matlabroot;

#[cfg(test)]
mod tests;

use crate::BuiltinCatalogEntry;

pub use matlabroot::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MATLABROOT_CATALOG_ENTRY];
pub(super) use inference::infer;
