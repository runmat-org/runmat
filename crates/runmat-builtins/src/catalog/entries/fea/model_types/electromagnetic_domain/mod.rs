//! `fea.ElectromagneticDomain` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod at_frequency;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&at_frequency::ENTRY];
