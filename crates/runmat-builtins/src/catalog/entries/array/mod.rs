mod binning;
mod combinatorics;
mod creation;
mod documentation;
mod introspection;

pub use binning::*;
pub use combinatorics::*;
pub use creation::*;
pub use introspection::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(
        entries,
        &[binning::ENTRIES, combinatorics::ENTRIES, creation::ENTRIES],
    );
    super::extend_groups(entries, introspection::ENTRY_GROUPS);
}
