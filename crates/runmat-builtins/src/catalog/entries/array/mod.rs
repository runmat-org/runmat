mod accumulation;
mod binning;
mod combinatorics;
mod creation;
mod documentation;
mod grouping;
mod introspection;

pub use accumulation::*;
pub use binning::*;
pub use combinatorics::*;
pub use creation::*;
pub use grouping::*;
pub use introspection::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(
        entries,
        &[
            accumulation::ENTRIES,
            binning::ENTRIES,
            combinatorics::ENTRIES,
            creation::ENTRIES,
            grouping::ENTRIES,
        ],
    );
    super::extend_groups(entries, introspection::ENTRY_GROUPS);
}
