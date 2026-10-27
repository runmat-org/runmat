//! `fea.ThermoMechanicalDomain` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod at_reference_temperature;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&at_reference_temperature::ENTRY];
