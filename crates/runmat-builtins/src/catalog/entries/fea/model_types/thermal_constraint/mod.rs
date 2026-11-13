//! `fea.ThermalConstraint` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod affine_jump;
mod periodic;
mod prescribed;
mod tie;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &prescribed::ENTRY,
    &tie::ENTRY,
    &periodic::ENTRY,
    &affine_jump::ENTRY,
];
