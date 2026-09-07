mod matrix_arithmetic;

pub use matrix_arithmetic::*;

use crate::BuiltinCatalogEntry;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = matrix_arithmetic::ENTRIES;
