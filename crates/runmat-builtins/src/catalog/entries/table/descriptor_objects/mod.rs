mod contract;
mod facts;
mod array_datastore;
mod parquet_datastore;
mod rowfilter;
mod timerange;
mod uitable;
mod vartype;
#[cfg(test)]
mod tests;

pub use array_datastore::*;
pub use parquet_datastore::*;
pub use rowfilter::*;
pub use timerange::*;
pub use uitable::*;
pub use vartype::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &TIMERANGE_ENTRY,
    &VARTYPE_ENTRY,
    &ROWFILTER_ENTRY,
    &ARRAY_DATASTORE_ENTRY,
    &PARQUET_DATASTORE_ENTRY,
    &UITABLE_ENTRY,
];
