//! `fea.AcousticConstraint` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod periodic;
mod phase_shifted_periodic;
mod prescribed_pressure;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &prescribed_pressure::ENTRY,
    &periodic::ENTRY,
    &phase_shifted_periodic::ENTRY,
];
