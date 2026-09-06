pub(in crate::catalog::entries::io) fn extend_entries(
    entries: &mut Vec<&'static crate::BuiltinCatalogEntry>,
) {
    super::super::super::extend_groups(
        entries,
        &[
            super::addpath::ENTRIES,
            super::cd::ENTRIES,
            super::directory_lifecycle::ENTRIES,
            super::environment::ENTRIES,
            super::genpath::ENTRIES,
            super::path::ENTRIES,
            super::path_syntax::ENTRIES,
            super::pwd::ENTRIES,
            super::rmpath::ENTRIES,
            super::savepath::ENTRIES,
            super::temporary_path::ENTRIES,
        ],
    );
}
