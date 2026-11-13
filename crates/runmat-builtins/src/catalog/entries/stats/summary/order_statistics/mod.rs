mod contract;
mod inference;
mod options;
mod shape;
mod prctile;
mod quantile;
mod tiedrank;

pub use options::{OrderStatisticOption, QuantileRequestUnit, TiedRankMode};
pub use prctile::{PRCTILE_CATALOG_ENTRY, PRCTILE_DATA_EXTENSION, PRCTILE_PROBABILITY_EXTENSION};
pub use quantile::{
    QUANTILE_CATALOG_ENTRY, QUANTILE_DATA_EXTENSION, QUANTILE_PROBABILITY_EXTENSION,
};
pub use tiedrank::{TIEDRANK_CATALOG_ENTRY, TIEDRANK_DATA_EXTENSION};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &PRCTILE_CATALOG_ENTRY,
    &QUANTILE_CATALOG_ENTRY,
    &TIEDRANK_CATALOG_ENTRY,
];
