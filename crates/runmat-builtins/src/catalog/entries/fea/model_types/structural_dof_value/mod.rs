//! `fea.StructuralDofValue` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod rotation;
mod translation;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&translation::ENTRY, &rotation::ENTRY];
