//! `fea.FluidConstraint` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod moving_wall;
mod no_slip_wall;
mod periodic;
mod prescribed_velocity;
mod pressure_reference;
mod slip_wall;
mod symmetry;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &no_slip_wall::ENTRY,
    &slip_wall::ENTRY,
    &symmetry::ENTRY,
    &prescribed_velocity::ENTRY,
    &moving_wall::ENTRY,
    &pressure_reference::ENTRY,
    &periodic::ENTRY,
];
