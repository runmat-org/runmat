//! `fea.CoordinateFrame` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod boundary_normal_tangent;
mod cartesian;
mod cylindrical;
mod element;
mod global_frame;
mod material;
mod spherical;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &global_frame::ENTRY,
    &material::ENTRY,
    &element::ENTRY,
    &boundary_normal_tangent::ENTRY,
    &cartesian::ENTRY,
    &cylindrical::ENTRY,
    &spherical::ENTRY,
];
