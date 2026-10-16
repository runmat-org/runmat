mod entry;
mod inference;
mod options;
mod chol;
mod decomposition;
mod eig;
mod eigs;
mod lu;
mod qr;
mod svd;
#[cfg(test)]
mod tests;

pub use chol::*;
pub use decomposition::*;
pub use eig::*;
pub use eigs::*;
pub use lu::*;
pub use options::FactorOption;
pub use qr::*;
pub use svd::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    entries.extend(decomposition::ENTRIES.iter().copied());
    entries.extend([
        &CHOL_CATALOG_ENTRY,
        &EIG_CATALOG_ENTRY,
        &EIGS_CATALOG_ENTRY,
        &LU_CATALOG_ENTRY,
        &QR_CATALOG_ENTRY,
        &SVD_CATALOG_ENTRY,
    ]);
}
