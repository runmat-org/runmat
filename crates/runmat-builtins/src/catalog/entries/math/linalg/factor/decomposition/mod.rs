mod documentation;
mod control;
mod inference;
mod integer_inputs;
mod integer_capabilities;
mod signatures;
mod constructor;
mod subsref;
mod uplus;
mod uminus;
mod ctranspose;
mod times;
mod mtimes;
mod rdivide;
mod mrdivide;
mod mldivide;
pub use constructor::*;
pub use control::{DecompositionKind, DecompositionOption, DecompositionTriangle};
pub use ctranspose::*;
pub use integer_capabilities::*;
pub use integer_inputs::*;
pub use mldivide::*;
pub use mrdivide::*;
pub use mtimes::*;
pub use rdivide::*;
pub use signatures::*;
pub use subsref::*;
pub use times::*;
pub use uminus::*;
pub use uplus::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &DECOMPOSITION_CATALOG_ENTRY,
    &DECOMPOSITION_SUBSREF_CATALOG_ENTRY,
    &DECOMPOSITION_UPLUS_CATALOG_ENTRY,
    &DECOMPOSITION_UMINUS_CATALOG_ENTRY,
    &DECOMPOSITION_CTRANSPOSE_CATALOG_ENTRY,
    &DECOMPOSITION_TIMES_CATALOG_ENTRY,
    &DECOMPOSITION_MTIMES_CATALOG_ENTRY,
    &DECOMPOSITION_RDIVIDE_CATALOG_ENTRY,
    &DECOMPOSITION_MRDIVIDE_CATALOG_ENTRY,
    &DECOMPOSITION_MLDIVIDE_CATALOG_ENTRY,
];
