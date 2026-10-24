//! Typed FEA policy constructor contracts.

mod constraint_support;
mod dof;
mod entry;
mod frame;
mod reference_point;
#[cfg(test)]
mod tests;
pub use self::constraint_support::*;
pub use self::dof::*;
pub use self::frame::*;
pub use self::reference_point::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(constraint_support::ENTRIES.iter().copied());
    values.extend(frame::ENTRIES.iter().copied());
    values.extend(reference_point::ENTRIES.iter().copied());
    values.extend(dof::ENTRIES.iter().copied());
}
