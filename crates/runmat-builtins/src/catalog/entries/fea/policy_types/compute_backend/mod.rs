//! Catalog identities of the `fea.ComputeBackend` typed policy class.

use crate::*;

mod cpu;
mod gpu;
#[cfg(test)]
mod tests;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &cpu::FEA_COMPUTE_BACKEND_CPU_CATALOG_ENTRY,
    &gpu::FEA_COMPUTE_BACKEND_GPU_CATALOG_ENTRY,
];
