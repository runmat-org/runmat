//! `fea.StructuralDof` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod rx;
mod ry;
mod rz;
mod ux;
mod uy;
mod uz;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &ux::ENTRY,
    &uy::ENTRY,
    &uz::ENTRY,
    &rx::ENTRY,
    &ry::ENTRY,
    &rz::ENTRY,
];
