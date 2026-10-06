use super::*;
use runmat_canonical_codec::CanonicalCodecErrorKind;
use serde::ser::SerializeSeq;
use std::cell::Cell;

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
    let existing = temp.path().join("retained.json");
    let original = serde_json::json!({"original": "immutable evidence"});
    atomic_json(&existing, &original).unwrap();
    let original_bytes = std::fs::read(&existing).unwrap();
    let inventory = || {
        let mut entries = std::fs::read_dir(temp.path())
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
    let absent = temp.path().join("must-not-create").join("rejected.json");
    assert!(immutable_json(&absent, &expanding).is_err());
    assert!(!absent.parent().unwrap().exists());
    assert_eq!(inventory(), original_inventory);
}
