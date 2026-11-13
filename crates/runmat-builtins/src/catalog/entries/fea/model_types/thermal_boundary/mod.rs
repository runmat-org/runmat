//! `fea.ThermalBoundary` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod convection;
mod heat_flux;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&convection::ENTRY, &heat_flux::ENTRY];
