//! Scalar optimization contracts and shared option identities.
mod fminbnd;
mod fzero;
mod integral;
mod options;
mod quad;
mod scalar_facts;
#[cfg(test)]
mod options_tests;

pub use fminbnd::*;
pub use fzero::*;
pub use integral::*;
pub use options::{OptimAlgorithm, OptimDisplay, OptimOptionField, OptimSolver, OptimSwitch};
pub use quad::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(fminbnd::ENTRIES.iter().copied());
    values.extend(fzero::ENTRIES.iter().copied());
    values.extend(integral::ENTRIES.iter().copied());
    values.extend(quad::ENTRIES.iter().copied());
}
