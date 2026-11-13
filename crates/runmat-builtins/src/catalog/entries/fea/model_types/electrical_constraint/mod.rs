//! `fea.ElectricalConstraint` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod equipotential;
mod floating_conductor;
mod integral_current;
mod periodic;
mod potential_reference;
mod prescribed_potential;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &equipotential::ENTRY,
    &floating_conductor::ENTRY,
    &potential_reference::ENTRY,
    &prescribed_potential::ENTRY,
    &integral_current::ENTRY,
    &periodic::ENTRY,
];
