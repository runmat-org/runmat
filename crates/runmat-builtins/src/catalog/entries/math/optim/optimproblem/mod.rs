mod descriptor;
mod documentation;
mod entry;
mod inference;
mod solve;
mod exitflag;
pub use exitflag::*;
pub use solve::*;
#[cfg(test)]
mod tests;
pub use descriptor::*;
pub use entry::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &OPTIMPROBLEM_CATALOG_ENTRY,
    &OPTIMPROBLEM_SUBSREF_CATALOG_ENTRY,
    &OPTIMPROBLEM_SUBSASGN_CATALOG_ENTRY,
    &OPTIMPROBLEM_SOLVE_CATALOG_ENTRY,
    &OPTIMPROBLEM_SOLVE_METHOD_CATALOG_ENTRY,
    &OPTIMEXITFLAG_DOUBLE_CATALOG_ENTRY,
    &OPTIMEXITFLAG_STRING_CATALOG_ENTRY,
    &OPTIMEXITFLAG_EQ_CATALOG_ENTRY,
    &OPTIMEXITFLAG_NE_CATALOG_ENTRY,
];
