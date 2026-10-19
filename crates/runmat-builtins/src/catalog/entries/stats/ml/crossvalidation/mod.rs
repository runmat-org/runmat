//! Shared typed cross-validation controls and identity packages.

mod options;
mod crossvalind;
mod cvpartition;

pub use crossvalind::*;
pub use cvpartition::*;
pub use options::{
    parse_stratify_boolean_alias, parse_stratify_numeric_boolean, CrossvalIndMethod,
    CrossvalIndOption, CvPartitionMethod, CvPartitionOption, MaskSelectionKeyword,
    PartitionCountGrammar,
};

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(crossvalind::ENTRIES.iter().copied());
    values.extend(cvpartition::ENTRIES.iter().copied());
}
