use super::*;

#[test]
fn portable_snapshot_rejects_duplicate_invalid_secret_and_oversized_inventory() {
    assert!(validate(&[("RUNMAT_MODE".into(), "frozen".into())]).is_ok());
    for entries in [
        vec![("name".into(), "1".into()), ("name".into(), "2".into())],
        vec![("bad=name".into(), "1".into())],
        vec![("API_TOKEN".into(), "redacted".into())],
        vec![("OTEL_EXPORTER_OTLP_HEADERS".into(), "redacted".into())],
        vec![(
            "OTEL_EXPORTER_OTLP_ENDPOINT".into(),
            "https://submitter.invalid".into(),
        )],
        vec![(
            "RUNMAT_OTEL_ENDPOINT".into(),
            "https://submitter.invalid".into(),
        )],
        vec![(
            "RUNMAT_RUNTIME__LOGGING__OTEL__ENABLED".into(),
            "true".into(),
        )],
        vec![("name".into(), "x".repeat(4097))],
        (0..257)
            .map(|i| (format!("VAR{i:03}"), "1".into()))
            .collect(),
        (0..256)
            .map(|i| (format!("VAR{i:03}"), "x".repeat(4096)))
            .collect(),
    ] {
        assert!(validate(&entries).is_err());
    }
}
