use crate::*;

mod pivot;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&pivot::ENTRY];

pub use pivot::{PIVOT_DESCRIPTOR, PIVOT_INTEGER_CAPABILITIES};
