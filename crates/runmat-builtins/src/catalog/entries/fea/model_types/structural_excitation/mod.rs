//! `fea.StructuralExcitation` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod body_force;
mod force;
mod moment;
mod pressure;
mod traction;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &force::ENTRY,
    &moment::ENTRY,
    &traction::ENTRY,
    &pressure::ENTRY,
    &body_force::ENTRY,
];
