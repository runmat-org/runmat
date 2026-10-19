//! `fea.ElectricalInitialCondition` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod potential;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&potential::ENTRY];
