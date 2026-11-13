pub(super) fn extend_groups(
    entries: &mut Vec<&'static crate::BuiltinCatalogEntry>,
    groups: &[&[&'static crate::BuiltinCatalogEntry]],
) {
    entries.extend(groups.iter().flat_map(|group| group.iter().copied()));
}
