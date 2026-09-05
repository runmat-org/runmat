//! Catalog-owned definitions for numeric-limit builtins.

mod contract;
mod evidence;
mod flintmax;
mod intmax;
mod intmin;
mod realmax;
mod realmin;

pub use contract::{
    NUMERIC_LIMIT_ERROR_INTERNAL, NUMERIC_LIMIT_ERROR_INVALID_CLASS,
    NUMERIC_LIMIT_ERROR_INVALID_SYNTAX,
};
pub use flintmax::FLINTMAX_CATALOG_ENTRY;
pub use intmax::INTMAX_CATALOG_ENTRY;
pub use intmin::INTMIN_CATALOG_ENTRY;
pub use realmax::REALMAX_CATALOG_ENTRY;
pub use realmin::REALMIN_CATALOG_ENTRY;

use crate::BuiltinCatalogEntry;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &INTMIN_CATALOG_ENTRY,
    &INTMAX_CATALOG_ENTRY,
    &REALMIN_CATALOG_ENTRY,
    &REALMAX_CATALOG_ENTRY,
    &FLINTMAX_CATALOG_ENTRY,
];
