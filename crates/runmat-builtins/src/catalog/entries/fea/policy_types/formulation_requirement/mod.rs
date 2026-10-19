//! Catalog identities of the `fea.FormulationRequirement` typed policy class.

use crate::*;

mod experimental;
mod production;
#[cfg(test)]
mod tests;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &production::FEA_FORMULATION_REQUIREMENT_PRODUCTION_CATALOG_ENTRY,
    &experimental::FEA_FORMULATION_REQUIREMENT_EXPERIMENTAL_CATALOG_ENTRY,
];
