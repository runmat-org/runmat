//! `fea.StructuralDof` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod rx;
mod ry;
mod rz;
mod ux;
mod uy;
mod uz;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &ux::ENTRY,
    &uy::ENTRY,
    &uz::ENTRY,
    &rx::ENTRY,
    &ry::ENTRY,
    &rz::ENTRY,
];
