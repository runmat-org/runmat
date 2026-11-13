mod contract;
mod documentation;
mod entry;
mod inference;
mod options;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::PERFCURVE_CATALOG_ENTRY;
pub use options::{
    PerfcurveBoolean, PerfcurveCriterion, PerfcurveOptionName, PerfcurvePriorKeyword,
    PerfcurveProcessNaN,
};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&PERFCURVE_CATALOG_ENTRY];
