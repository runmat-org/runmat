mod entry;
mod inference;

mod cond;
mod det;
mod inv;
mod linsolve;
mod norm;
mod null;
mod pinv;
mod rank;
mod rcond;
mod rref;
mod vecnorm;
#[cfg(test)]
mod tests;

pub use cond::*;
pub use det::*;
pub use inv::*;
pub use linsolve::*;
pub use norm::*;
pub use null::*;
pub use pinv::*;
pub use rank::*;
pub use rcond::*;
pub use rref::*;
pub use vecnorm::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    entries.extend([
        &COND_CATALOG_ENTRY,
        &DET_CATALOG_ENTRY,
        &INV_CATALOG_ENTRY,
        &LINSOLVE_CATALOG_ENTRY,
        &NORM_CATALOG_ENTRY,
        &NULL_CATALOG_ENTRY,
        &PINV_CATALOG_ENTRY,
        &RANK_CATALOG_ENTRY,
        &RCOND_CATALOG_ENTRY,
        &RREF_CATALOG_ENTRY,
        &VECNORM_CATALOG_ENTRY,
    ]);
}
