//! `fea.ElectricalExcitation` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod charge;
mod current;
mod current_density;
mod terminal_voltage;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &terminal_voltage::ENTRY,
    &current::ENTRY,
    &charge::ENTRY,
    &current_density::ENTRY,
];
