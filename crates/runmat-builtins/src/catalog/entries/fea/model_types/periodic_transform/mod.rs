//! `fea.PeriodicTransform` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod identity;
mod rotation;
mod translation;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&identity::ENTRY, &translation::ENTRY, &rotation::ENTRY];
