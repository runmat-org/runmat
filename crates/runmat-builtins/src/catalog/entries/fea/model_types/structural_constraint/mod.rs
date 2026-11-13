//! `fea.StructuralConstraint` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod antisymmetry;
mod distributing_coupling;
mod fixed;
mod normal_displacement;
mod periodic;
mod prescribed;
mod prescribed_motion;
mod rigid_coupling;
mod symmetry;
mod tie;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &fixed::ENTRY,
    &prescribed::ENTRY,
    &prescribed_motion::ENTRY,
    &symmetry::ENTRY,
    &antisymmetry::ENTRY,
    &normal_displacement::ENTRY,
    &rigid_coupling::ENTRY,
    &distributing_coupling::ENTRY,
    &tie::ENTRY,
    &periodic::ENTRY,
];
