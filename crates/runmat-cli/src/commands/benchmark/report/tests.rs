use super::*;

#[test]
fn streamed_reports_are_exclusive_and_abandonment_is_not_success() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("samples.jsonl");
    {
        let _report = Report::create(&path).unwrap();
        assert!(Report::create(&path).is_err());
    }
    let read = || {
        std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>()
    };
    let rows = read();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["record"], "begin");
    assert_eq!(rows[1]["status"], "failed_or_abandoned");
    assert_eq!(rows[1]["samples"], 0);
    let completed = directory.path().join("completed.jsonl");
    {
        let mut report = Report::create(&completed).unwrap();
        report.finish().unwrap();
    }
    let text = std::fs::read_to_string(completed).unwrap();
    assert_eq!(text.lines().count(), 2);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(text.lines().nth(1).unwrap()).unwrap()["status"],
        "completed"
    );
}

#[test]
fn command_projection_does_not_promote_partial_physical_counts() {
    let evidence = runmat_core::CommandExecutionEvidence {
        physical_submissions: Some(runmat_execution::profiling::PhysicalSubmissionCounts {
            pending: 1,
            ..Default::default()
        }),
        deferred_work: None,
        route_coverage: runmat_execution::profiling::CommandRouteCoverage::Incomplete,
        canonical_device_route_admitted: true,
        unproven_device_route: false,
        unproven_device_time: true,
    };
    let value = serde_json::to_value(Command::from(evidence)).unwrap();
    assert_eq!(value["coverage"], "incomplete");
    assert_eq!(value["physical"]["pending"], 1);
    assert!(value["deferred"].is_null());
    assert_eq!(value["unproven_device_time"], true);
}
