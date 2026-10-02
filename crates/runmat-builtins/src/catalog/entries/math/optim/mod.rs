//! Shared optimization option identities.
mod fminbnd;
mod fzero;
mod options;
mod scalar_facts;
#[cfg(test)]
mod options_tests;

pub use fminbnd::*;
pub use fzero::*;
pub use options::{OptimAlgorithm, OptimDisplay, OptimOptionField, OptimSolver, OptimSwitch};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&FMINBND_CATALOG_ENTRY, &FZERO_CATALOG_ENTRY];
