//! `fea.MagneticBoundary` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod insulation;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&insulation::ENTRY];
