//! `fea.ThermalExcitation` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod ambient_temperature;
mod heat_flux;
mod heat_generation;
mod temperature_field;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &heat_generation::ENTRY,
    &heat_flux::ENTRY,
    &temperature_field::ENTRY,
    &ambient_temperature::ENTRY,
];
