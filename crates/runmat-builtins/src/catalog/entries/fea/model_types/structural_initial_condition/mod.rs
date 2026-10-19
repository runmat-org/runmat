//! `fea.StructuralInitialCondition` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod angular_velocity;
mod displacement;
mod velocity;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &displacement::ENTRY,
    &velocity::ENTRY,
    &angular_velocity::ENTRY,
];
