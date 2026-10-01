mod descriptor;
mod extensions;
mod integer;
mod inference;
mod corrcov;
mod cov2corr;

pub use corrcov::CORRCOV_CATALOG_ENTRY;
pub use cov2corr::COV2CORR_CATALOG_ENTRY;
pub use descriptor::{CORRCOV_DESCRIPTOR, COV2CORR_DESCRIPTOR};
pub use extensions::*;
pub use integer::{CORRCOV_INTEGER_CAPABILITIES, COV2CORR_INTEGER_CAPABILITIES};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&CORRCOV_CATALOG_ENTRY, &COV2CORR_CATALOG_ENTRY];
