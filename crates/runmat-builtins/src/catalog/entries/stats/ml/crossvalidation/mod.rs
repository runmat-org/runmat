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

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &CROSSVALIND_CATALOG_ENTRY,
    &CVPARTITION_CATALOG_ENTRY,
    &TEST_CATALOG_ENTRY,
    &TRAINING_CATALOG_ENTRY,
];
