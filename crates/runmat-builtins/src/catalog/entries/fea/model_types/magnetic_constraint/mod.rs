//! `fea.MagneticConstraint` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod coulomb_gauge;
mod periodic;
mod phase_shifted_periodic;
mod prescribed_vector_potential;
mod vector_potential_ground;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &vector_potential_ground::ENTRY,
    &coulomb_gauge::ENTRY,
    &prescribed_vector_potential::ENTRY,
    &periodic::ENTRY,
    &phase_shifted_periodic::ENTRY,
];
