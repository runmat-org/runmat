//! Typed FEA policy constructor contracts.

mod constraint_support;
pub use self::constraint_support::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(constraint_support::ENTRIES.iter().copied());
}
