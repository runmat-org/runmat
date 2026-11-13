//! `fea.ThermalDomain` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod at_reference_temperature;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&at_reference_temperature::ENTRY];
