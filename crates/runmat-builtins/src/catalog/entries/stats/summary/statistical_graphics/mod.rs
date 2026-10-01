//! Statistical graphics identities. Each entry owns its public contract.
mod reference_lines;

use crate::BuiltinCatalogEntry;

pub use reference_lines::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = reference_lines::ENTRIES;
