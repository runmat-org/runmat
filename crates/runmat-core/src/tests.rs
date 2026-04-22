use crate::*;
use futures::executor::block_on;

#[test]
fn captures_basic_workspace_assignments() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let result = block_on(session.execute("x = 42;")).expect("exec succeeds");
    assert!(
        result
            .workspace
            .values
            .iter()
            .any(|entry| entry.name == "x"),
        "workspace snapshot should include assigned variable"
    );
}

#[test]
fn workspace_reports_datetime_array_shape() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let result =
        block_on(session.execute("d = datetime([739351; 739352], 'ConvertFrom', 'datenum');"))
            .expect("exec succeeds");
    let entry = result
        .workspace
        .values
        .iter()
        .find(|entry| entry.name == "d")
        .expect("workspace entry for d");
    assert_eq!(entry.class_name, "datetime");
    assert_eq!(entry.shape, vec![2, 1]);
}

#[test]
fn workspace_state_roundtrip_replace_only() {
    let mut source_session =
        RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _ = block_on(source_session.execute("x = 42; y = [1, 2, 3];")).expect("exec succeeds");

    let bytes = block_on(source_session.export_workspace_state(WorkspaceExportMode::Force))
        .expect("workspace export")
        .expect("workspace bytes");

    let mut restore_session =
        RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _ = block_on(restore_session.execute("z = 99;")).expect("seed workspace");
    restore_session
        .import_workspace_state(&bytes)
        .expect("workspace import");

    let _restored = block_on(restore_session.execute("r = x + y(2);")).expect("post-import exec");

    let x = block_on(restore_session.materialize_variable(
        WorkspaceMaterializeTarget::Name("x".to_string()),
        WorkspaceMaterializeOptions::default(),
    ))
    .expect("x should exist after import");
    assert_eq!(x.name, "x");

    let y = block_on(restore_session.materialize_variable(
        WorkspaceMaterializeTarget::Name("y".to_string()),
        WorkspaceMaterializeOptions::default(),
    ))
    .expect("y should exist after import");
    assert_eq!(y.name, "y");

    let z = block_on(restore_session.materialize_variable(
        WorkspaceMaterializeTarget::Name("z".to_string()),
        WorkspaceMaterializeOptions::default(),
    ));
    assert!(
        z.is_err(),
        "replace-only import should drop stale z variable"
    );
}

#[test]
fn workspace_state_import_rejects_invalid_payload() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let err = session
        .import_workspace_state(&[1, 2, 3, 4])
        .expect_err("invalid payload should be rejected");
    let runtime_err = err
        .downcast_ref::<runmat_runtime::RuntimeError>()
        .expect("error should preserve runtime replay details");
    assert_eq!(
        runtime_err.identifier(),
        Some("RunMat:ReplayDecodeFailed"),
        "invalid payload should map to replay decode identifier"
    );
}
