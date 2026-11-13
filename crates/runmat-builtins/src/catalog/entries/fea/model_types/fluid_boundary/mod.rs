//! `fea.FluidBoundary` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod outlet_pressure;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&outlet_pressure::ENTRY];
