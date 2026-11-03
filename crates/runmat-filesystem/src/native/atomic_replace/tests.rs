use super::*;

#[test]
fn replacement_failure_retains_exact_stage_path_and_os_cause() {
    let directory = tempfile::tempdir().unwrap();
    let parent = directory.path().join("missing");
    let path = parent.join("head.json");
    let original = fs::metadata(&parent).unwrap_err();
    let error = publish(&path, b"replacement").unwrap_err();
    assert_eq!(error.kind(), original.kind());
    let context = error
        .get_ref()
        .unwrap()
        .downcast_ref::<ReplacementFailure>()
        .unwrap();
    assert!(matches!(context.operation, Operation::InspectParent));
    assert_eq!(context.path, parent);
    assert_eq!(context.source.raw_os_error(), original.raw_os_error());
    assert!(context.cleanup.is_none());
    assert!(std::error::Error::source(context)
        .unwrap()
        .is::<io::Error>());
}

#[test]
fn replacement_is_complete_and_flush_uncertainty_does_not_undo_visibility() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("current");
    let first = publish(&path, b"original").unwrap();
    assert_eq!(first.durability, PublicationDurability::Durable);
    assert_eq!(first.durability_diagnostic, None);
    let next = publish_with_directory_sync(&path, b"replacement", |_| {
        Err(io::Error::other("injected directory flush failure"))
    })
    .unwrap();
    assert_eq!(fs::read(&path).unwrap(), b"replacement");
    assert_eq!(next.durability, PublicationDurability::AtomicVisibility);
    assert_eq!(
        next.durability_diagnostic.as_deref(),
        Some("injected directory flush failure")
    );
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn competing_replacements_never_expose_partial_payloads() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("current");
    let payloads: Vec<Vec<u8>> = (0..4).map(|i| vec![i; 128 * 1024]).collect();
    publish(&path, &payloads[0]).unwrap();
    std::thread::scope(|scope| {
        let mut writers = Vec::new();
        for payload in &payloads {
            let path = &path;
            writers.push(scope.spawn(move || {
                for _ in 0..8 {
                    publish(path, payload).unwrap();
                }
            }));
        }
        for _ in 0..64 {
            let observed = fs::read(&path).unwrap();
            assert!(payloads.contains(&observed));
        }
        for writer in writers {
            writer.join().unwrap();
        }
    });
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn invalid_destinations_reject_without_replacing_or_leaking_staging() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("directory");
    fs::create_dir(&path).unwrap();
    assert!(publish(&path, b"replacement").is_err());
    assert!(path.is_dir());
    assert!(publish(&path.join("missing").join("current"), b"replacement").is_err());
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    #[cfg(unix)]
    {
        let target = directory.path().join("target");
        fs::write(&target, b"original").unwrap();
        let link = directory.path().join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(publish(&link, b"replacement").is_err());
        assert_eq!(fs::read(target).unwrap(), b"original");
        assert!(fs::symlink_metadata(link).unwrap().file_type().is_symlink());
    }
}
