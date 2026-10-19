//! Catalog identities of the `fea.SolverOptions` typed policy class.

use crate::*;

mod defaults;
#[cfg(test)]
mod tests;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&defaults::FEA_SOLVER_OPTIONS_DEFAULTS_CATALOG_ENTRY];
