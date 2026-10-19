//! `fea.PeriodicPairing` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod geometric;
mod matching;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&matching::ENTRY, &geometric::ENTRY];
