//! `fea.AcousticExcitation` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod normal_velocity;
mod pressure;
mod volume_velocity;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &pressure::ENTRY,
    &normal_velocity::ENTRY,
    &volume_velocity::ENTRY,
];
