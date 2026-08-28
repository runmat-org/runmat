use runmat_package_cache::{validate_archive, ArchiveEntryHeader, ArchiveEntryKind, ArchiveLimits};

fn file(path: &str, expanded_bytes: u64, compressed_bytes: u64) -> ArchiveEntryHeader {
    ArchiveEntryHeader {
        path: path.to_string(),
        kind: ArchiveEntryKind::File,
        expanded_bytes,
        compressed_bytes,
        link_target: None,
        executable: false,
    }
}

fn link(path: &str, target: &str) -> ArchiveEntryHeader {
    ArchiveEntryHeader {
        path: path.to_string(),
        kind: ArchiveEntryKind::Symlink,
        expanded_bytes: 0,
        compressed_bytes: 0,
        link_target: Some(target.to_string()),
        executable: false,
    }
}

#[test]
fn valid_headers_canonicalize_paths_and_links() {
    let archive = validate_archive(
        [
            file("src\\main.m", 4, 4),
            ArchiveEntryHeader {
                path: "bin".to_string(),
                kind: ArchiveEntryKind::Directory,
                expanded_bytes: 0,
                compressed_bytes: 0,
                link_target: None,
                executable: false,
            },
            link("bin/main", "../src/main.m"),
        ],
        ArchiveLimits::default(),
    )
    .unwrap();
    assert_eq!(archive.file_count, 1);
    assert_eq!(archive.expanded_bytes, 4);
    assert_eq!(archive.entries[2].path.as_str(), "src/main.m");
    assert_eq!(
        archive.entries[1].link_target.as_ref().unwrap().as_str(),
        "src/main.m"
    );
}

#[test]
fn traversal_absolute_and_escaping_links_are_rejected() {
    for header in [
        file("../outside", 1, 1),
        file("/absolute", 1, 1),
        file(r"C:\outside", 1, 1),
        link("src/link", "../../outside"),
    ] {
        assert!(validate_archive([header], ArchiveLimits::default()).is_err());
    }
}

#[test]
fn special_files_reserved_names_and_invalid_components_are_rejected() {
    let special = ArchiveEntryHeader {
        path: "device".to_string(),
        kind: ArchiveEntryKind::CharacterDevice,
        expanded_bytes: 0,
        compressed_bytes: 0,
        link_target: None,
        executable: false,
    };
    assert!(validate_archive([special], ArchiveLimits::default()).is_err());
    for path in [
        "CON",
        "aux.txt",
        "src/trailing.",
        "src/bad:name",
        "src//file",
    ] {
        assert!(validate_archive([file(path, 1, 1)], ArchiveLimits::default()).is_err());
    }
}

#[test]
fn case_unicode_and_ancestor_collisions_are_rejected() {
    assert!(validate_archive(
        [file("Src/Main.m", 1, 1), file("src/main.m", 1, 1)],
        ArchiveLimits::default()
    )
    .is_err());
    assert!(validate_archive(
        [file("café.m", 1, 1), file("cafe\u{301}.m", 1, 1)],
        ArchiveLimits::default()
    )
    .is_err());
    assert!(validate_archive(
        [file("straße.m", 1, 1), file("STRASSE.m", 1, 1)],
        ArchiveLimits::default()
    )
    .is_err());
    assert!(validate_archive(
        [file("src", 1, 1), file("src/main.m", 1, 1)],
        ArchiveLimits::default()
    )
    .is_err());
}

#[test]
fn expanded_size_count_and_compression_limits_are_enforced() {
    let limits = ArchiveLimits {
        max_entries: 2,
        max_files: 1,
        max_expanded_bytes: 10,
        max_file_bytes: 8,
        max_compression_ratio: 4,
        ..ArchiveLimits::default()
    };
    assert!(validate_archive([file("large", 9, 9)], limits).is_err());
    assert!(validate_archive([file("one", 1, 1), file("two", 1, 1)], limits).is_err());
    assert!(validate_archive([file("bomb", 8, 1)], limits).is_err());
}

#[test]
fn hardlinks_must_resolve_to_regular_files() {
    let hardlink = ArchiveEntryHeader {
        path: "copy".to_string(),
        kind: ArchiveEntryKind::Hardlink,
        expanded_bytes: 0,
        compressed_bytes: 0,
        link_target: Some("missing".to_string()),
        executable: false,
    };
    assert!(validate_archive([hardlink], ArchiveLimits::default()).is_err());
}

#[test]
fn broken_and_cyclic_symlinks_are_rejected() {
    assert!(validate_archive([link("src/link", "missing")], ArchiveLimits::default()).is_err());
    assert!(validate_archive([link("a", "b"), link("b", "a")], ArchiveLimits::default()).is_err());
}
