//! Differentiable dlarray arithmetic and traced gradients.

mod array_methods;
mod declarations;
mod dlgradient;

pub use array_methods::*;
pub use dlgradient::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(declarations::ENTRIES.iter().copied());
}
