use super::*;

#[test]
fn original_entries_are_canonical_and_identity_distinguishes_absent_from_empty() {
    let absent = EnvironmentSnapshot::default();
    let empty = EnvironmentSnapshot::new(vec![("NAME".into(), "".into())]).unwrap();
    let first = EnvironmentSnapshot::new(vec![("NAME".into(), "first".into())]).unwrap();
    let decoded: EnvironmentSnapshot =
        serde_json::from_slice(&serde_json::to_vec(&first).unwrap()).unwrap();
    assert_eq!(decoded, first);
    assert_ne!(absent.identity().unwrap(), empty.identity().unwrap());
    assert_ne!(empty.identity().unwrap(), first.identity().unwrap());
    assert!(!format!("{first:?}").contains("first"));
}

#[test]
fn unordered_duplicate_invalid_and_oversized_decoded_entries_reject() {
    for entries in [
        vec![("B".into(), "1".into()), ("A".into(), "2".into())],
        vec![("A".into(), "1".into()), ("A".into(), "2".into())],
        vec![("bad=name".into(), "1".into())],
        vec![("A".into(), "bad\0value".into())],
        vec![("A".into(), "x".repeat(MAX_ENVIRONMENT_BYTES))],
    ] {
        assert!(EnvironmentSnapshot::new(entries.clone()).is_err());
        let decoded: EnvironmentSnapshot =
            serde_json::from_value(serde_json::json!({"entries":entries})).unwrap();
        assert!(decoded.validate().is_err());
        assert!(decoded.identity().is_err());
    }
}

#[test]
fn environment_storage_profiles_original_utf8_and_escaped_transport_without_copying() {
    let snapshot = EnvironmentSnapshot::new(vec![
        ("A".into(), "quoted\"\n".into()),
        ("B".into(), "λ".into()),
    ])
    .unwrap();
    let profile = snapshot.storage_profile().unwrap();
    assert_eq!(profile.entries, 2);
    assert_eq!(profile.utf8_bytes, 12);
    assert_eq!(
        profile.json_bytes,
        serde_json::to_vec(&snapshot).unwrap().len() as u64
    );
    assert_eq!(
        profile.minimum_memory_bytes().unwrap(),
        profile.typed_data_bytes + profile.json_bytes
    );
    assert!(profile.json_bytes > profile.utf8_bytes);
}
