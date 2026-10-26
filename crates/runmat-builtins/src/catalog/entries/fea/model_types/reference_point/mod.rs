//! `fea.ReferencePoint` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod from_coordinates;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&from_coordinates::ENTRY];
