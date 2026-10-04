#[test]
fn registry_preserves_reviewed_family_order_and_entry_identity() {
    let reviewed_groups = [
        super::super::addpath::ENTRIES,
        super::super::cd::ENTRIES,
        super::super::directory_lifecycle::ENTRIES,
        super::super::directory_listing::ENTRIES,
        super::super::environment::ENTRIES,
        super::super::exist::ENTRIES,
        &[&super::super::fileattrib::FILEATTRIB_CATALOG_ENTRY],
        super::super::file_transfer::ENTRIES,
        super::super::genpath::ENTRIES,
        super::super::installation_path::ENTRIES,
        &[&super::super::memmapfile::MEMMAPFILE_CATALOG_ENTRY],
        super::super::open::ENTRIES,
        super::super::opentoline::ENTRIES,
        super::super::path::ENTRIES,
        super::super::path_predicate::ENTRIES,
        super::super::path_syntax::ENTRIES,
        super::super::pcode::ENTRIES,
        super::super::preferences::ENTRIES,
        super::super::pwd::ENTRIES,
        &[&super::super::rehash::REHASH_CATALOG_ENTRY],
        &[&super::super::restoredefaultpath::RESTOREDEFAULTPATH_CATALOG_ENTRY],
        super::super::rmpath::ENTRIES,
        super::super::run::ENTRIES,
        super::super::savepath::ENTRIES,
        super::super::source_inventory::ENTRIES,
        super::super::system::ENTRIES,
        super::super::temporary_path::ENTRIES,
        super::super::uigetdir::ENTRIES,
        super::super::uigetfile::ENTRIES,
        super::super::uiputfile::ENTRIES,
        &[&super::super::userpath::USERPATH_CATALOG_ENTRY],
        super::super::filesystem::ENTRIES,
    ];
    let expected: Vec<_> = reviewed_groups.into_iter().flatten().copied().collect();
    let mut actual = Vec::new();
    super::extend_entries(&mut actual);
    assert_eq!(actual.len(), expected.len());
    // Const declarations may have distinct addresses; their typed identity
    // and source provenance define which entry the registry contributes.
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_eq!(actual.identity, expected.identity);
        assert_eq!(actual.provenance, expected.provenance);
    }
}
