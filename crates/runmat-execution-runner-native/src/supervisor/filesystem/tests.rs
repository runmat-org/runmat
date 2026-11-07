use super::*;
use runmat_canonical_codec::CanonicalCodecErrorKind;
use serde::ser::SerializeSeq;
use std::cell::Cell;

#[test]
fn failed_metadata_mutation_retains_operation_path_and_io_cause() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("authority");
    secure_directory(&root).unwrap();
    let path = root.join("head.json");
    std::fs::create_dir(&path).unwrap();
    let error = atomic_json(&path, &serde_json::json!({"head": 1})).unwrap_err();
    let NativeExecutionError::MetadataIo {
        operation,
        path: failed_path,
        source,
    } = &error
    else {
        panic!("expected typed metadata I/O cause, got {error}");
    };
    assert_eq!(*operation, crate::MetadataIoOperation::AtomicReplacement);
    assert_eq!(failed_path, &path);
    assert_eq!(source.kind(), std::io::ErrorKind::InvalidInput);
    assert!(std::error::Error::source(&error)
        .unwrap()
        .is::<std::io::Error>());
    assert!(path.is_dir());
}

/// A small borrowed inventory that would expand far beyond the real wire ceiling.
/// Early rejection must stop traversal, not finish encoding and then measure it.
struct Expanding<'a> {
    visited: &'a Cell<usize>,
    item: &'a str,
}
impl Serialize for Expanding<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(8192))?;
        for _ in 0..8192 {
            self.visited.set(self.visited.get() + 1);
            sequence.serialize_element(self.item)?;
        }
        sequence.end()
    }
}

#[test]
fn over_limit_metadata_stops_before_visibility_and_preserves_the_existing_winner() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("authority");
    let existing = root.join("retained.json");
    let original = serde_json::json!({"original": "immutable evidence"});
    atomic_json(&existing, &original).unwrap();
    let original_bytes = std::fs::read(&existing).unwrap();
    let inventory = || {
        let mut entries = std::fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>();
        entries.sort();
        entries
    };
    let original_inventory = inventory();
    let item = "x".repeat(64 * 1024);
    let visited = Cell::new(0);
    let expanding = Expanding {
        visited: &visited,
        item: &item,
    };
    for immutable in [false, true] {
        visited.set(0);
        let error = if immutable {
            immutable_json(&existing, &expanding).unwrap_err()
        } else {
            atomic_json(&existing, &expanding).unwrap_err()
        };
        assert!(matches!(error, NativeExecutionError::Encoding(error)
            if error.kind == CanonicalCodecErrorKind::ResourceLimit));
        assert!(
            visited.get() < 8192,
            "the encoder traversed the entire rejected inventory"
        );
        assert_eq!(std::fs::read(&existing).unwrap(), original_bytes);
    }
    let absent = root.join("must-not-create").join("rejected.json");
    assert!(immutable_json(&absent, &expanding).is_err());
    assert!(!absent.parent().unwrap().exists());
    assert_eq!(inventory(), original_inventory);
}
