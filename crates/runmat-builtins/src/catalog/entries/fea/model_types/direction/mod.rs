//! `fea.Direction` constructor family: one catalog identity per static method.

use crate::*;

mod vector;

pub use crate::FEA_DIRECTION_VECTOR_INTEGER_CAPABILITIES as FEA_DIRECTION_INTEGER_CAPABILITIES;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&vector::ENTRY];
