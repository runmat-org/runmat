mod groups;
#[cfg(test)]
mod tests;

pub(in crate::catalog::entries::io) fn extend_entries(
    entries: &mut Vec<&'static crate::BuiltinCatalogEntry>,
) {
    super::super::super::extend_groups(entries, groups::ENTRY_GROUPS);
}
