use super::*;

fn request(max_bytes: u64, max_files: u32) -> WorkspaceCaptureRequest {
    WorkspaceCaptureRequest {
        root: "/".into(),
        excluded_subtrees: vec!["excluded".into()],
        max_files,
        max_bytes,
    }
}

#[test]
fn complete_inventory_admission_precedes_content_materialization() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("small"), b"original").unwrap();
    let large = fs::File::create(root.path().join("large")).unwrap();
    large.set_len(4096).unwrap();
    assert!(inspection::inspect(root.path(), request(4096, 10)).is_err());
    let plan = inspection::inspect(root.path(), request(4104, 10)).unwrap();
    assert_eq!(plan.files.iter().map(|file| file.length).sum::<u64>(), 4104);
    let captured = materialize(root.path(), plan).unwrap();
    assert_eq!(
        captured
            .snapshot
            .files
            .iter()
            .map(|file| file.bytes.len())
            .sum::<usize>(),
        4104
    );
    assert_eq!(
        captured
            .snapshot
            .files
            .iter()
            .find(|file| file.path == Path::new("small"))
            .unwrap()
            .bytes,
        b"original"
    );
}

#[test]
fn resident_capture_boundary_includes_metadata_and_preserves_original_bytes() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("original"), b"original content").unwrap();
    let plan = inspection::inspect(root.path(), request(16, 10)).unwrap();
    let required = super::super::admission::capture(&plan, u64::MAX).unwrap();
    assert!(required > 16);
    assert!(capture_locked_admitted(root.path(), request(16, 10), required - 1).is_err());
    let captured = capture_locked_admitted(root.path(), request(16, 10), required).unwrap();
    assert_eq!(captured.files[0].bytes, b"original content");
}

#[test]
fn inspection_bounds_directory_metadata_even_without_file_contents() {
    let root = tempfile::tempdir().unwrap();
    for ordinal in 0..24 {
        fs::create_dir(root.path().join(format!("directory{ordinal}"))).unwrap();
    }
    assert!(inspection::inspect_admitted(root.path(), request(0, 100), 128).is_err());
    // Path storage/capacity differs across hosts. Use the structural admission
    // quote rather than assuming a POSIX-sized directory inventory fits 4KiB.
    let inventory = inspection::inspect(root.path(), request(0, 100)).unwrap();
    let required = super::super::admission::capture(&inventory, u64::MAX).unwrap();
    let plan = inspection::inspect_admitted(root.path(), request(0, 100), required).unwrap();
    assert_eq!(plan.directories.len(), 25);
    assert!(plan.files.is_empty());
}

#[test]
fn materialization_rejects_growth_and_shrinkage_after_metadata_admission() {
    for changed in [b"longer".as_slice(), b"x".as_slice()] {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("original");
        fs::write(&path, b"old").unwrap();
        let plan = inspection::inspect(root.path(), request(3, 10)).unwrap();
        fs::write(&path, changed).unwrap();
        assert!(materialize(root.path(), plan).is_err());
    }
}

#[test]
fn directory_count_and_exclusions_are_admitted_without_reading_contents() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("nested")).unwrap();
    fs::write(root.path().join("nested/file"), b"original").unwrap();
    fs::create_dir(root.path().join("excluded")).unwrap();
    fs::File::create(root.path().join("excluded/large"))
        .unwrap()
        .set_len(1 << 30)
        .unwrap();
    assert!(inspection::inspect(root.path(), request(8, 1)).is_err());
    let plan = inspection::inspect(root.path(), request(8, 2)).unwrap();
    assert_eq!(plan.files.len(), 1);
    assert_eq!(plan.directories.len(), 2);
    let captured = materialize(root.path(), plan).unwrap();
    assert_eq!(captured.snapshot.files[0].bytes, b"original");
}

#[cfg(unix)]
#[test]
fn replacement_with_symlink_between_inspection_and_read_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("file"), b"old").unwrap();
    let plan = inspection::inspect(root.path(), request(3, 10)).unwrap();
    fs::remove_file(root.path().join("file")).unwrap();
    std::os::unix::fs::symlink("/etc/hosts", root.path().join("file")).unwrap();
    assert!(materialize(root.path(), plan).is_err());
}
