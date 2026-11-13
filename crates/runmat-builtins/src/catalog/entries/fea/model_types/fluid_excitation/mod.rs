//! `fea.FluidExcitation` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod inlet_velocity;
mod mass_flow;
mod wall_motion;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &inlet_velocity::ENTRY,
    &mass_flow::ENTRY,
    &wall_motion::ENTRY,
];
