//! `fea.ContactInterface` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod coulomb;
mod frictionless;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&frictionless::ENTRY, &coulomb::ENTRY];
