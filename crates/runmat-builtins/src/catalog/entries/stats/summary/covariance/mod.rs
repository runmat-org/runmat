mod corrcoef;
mod cov;
mod geometry;

pub use corrcoef::*;
pub use cov::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(corrcoef::ENTRIES.iter().copied());
    values.extend(cov::ENTRIES.iter().copied());
}
