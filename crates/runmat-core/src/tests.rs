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
fn execute_outcome_exposes_runtime_flow() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = block_on(session.execute_outcome("1 + 1")).expect("exec succeeds");
    assert!(
        matches!(outcome.flow, abi::RuntimeFlow::Single(_)),
        "unsuppressed expression should adapt to a single-value ABI flow"
    );
    assert_eq!(outcome.display_events.len(), 1);
    assert!(outcome.diagnostics.is_empty());
}

#[test]
fn execute_outcome_exposes_workspace_upserts() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = block_on(session.execute_outcome("x = 42;")).expect("exec succeeds");
    let upsert = outcome
        .workspace_delta
        .upserts
        .iter()
        .find(|upsert| {
            matches!(
                &upsert.key,
                abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "x"
            )
        })
        .expect("ABI workspace delta should expose assigned x");
    assert_eq!(upsert.value, runmat_builtins::Value::Num(42.0));
}

#[test]
fn execute_outcome_exposes_workspace_removals_and_effects() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(session.execute_outcome("x = 1; y = 2;")).expect("seed workspace");
    let outcome = block_on(session.execute_outcome("clear x;")).expect("clear succeeds");

    assert!(outcome.workspace_delta.removals.iter().any(|key| {
        matches!(key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "x")
    }));
    assert!(!outcome.workspace_delta.removals.iter().any(|key| {
        matches!(key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
    }));
    assert!(outcome.effects.iter().any(|effect| {
        matches!(
            effect,
            abi::ObservedEffect::Workspace(abi::WorkspaceEffectKind::Clear)
        )
    }));
}

#[test]
fn execute_request_uses_request_workspace_handle() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let workspace = abi::WorkspaceHandle(uuid::Uuid::from_u128(7));
    let outcome = block_on(session.execute_request(abi::ExecutionRequest {
        source: abi::SourceInput::Text {
            name: "request-test.m".to_string(),
            text: "requested = 7;".to_string(),
        },
        entrypoint: abi::EntrypointSelector::SourcePath("request-test.m".to_string()),
        compatibility: runmat_hir::CompatibilityMode::Interactive,
        host_policy: abi::HostExecutionPolicy::default(),
        inputs: abi::RuntimeFlow::NoValue,
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace,
        resolver: abi::ResolverHandle(uuid::Uuid::from_u128(8)),
    }))
    .expect("exec succeeds");

    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(
            &upsert.key,
            abi::WorkspaceBindingKey::Interactive { session, name }
                if *session == workspace.0 && name.0 == "requested"
        )
    }));
}

#[test]
fn execute_request_honors_zero_requested_outputs() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = block_on(session.execute_request(abi::ExecutionRequest {
        source: abi::SourceInput::Text {
            name: "request-zero-output.m".to_string(),
            text: "1 + 1".to_string(),
        },
        entrypoint: abi::EntrypointSelector::SourcePath("request-zero-output.m".to_string()),
        compatibility: runmat_hir::CompatibilityMode::Interactive,
        host_policy: abi::HostExecutionPolicy::default(),
        inputs: abi::RuntimeFlow::NoValue,
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace: abi::WorkspaceHandle(uuid::Uuid::from_u128(9)),
        resolver: abi::ResolverHandle(uuid::Uuid::from_u128(10)),
    }))
    .expect("exec succeeds");

    assert!(outcome.flow.is_no_value());
    assert_eq!(outcome.display_events.len(), 1);
}

#[test]
fn compile_input_uses_semantic_vm_when_supported() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let prepared = session
        .compile_input("x = [1 2; 3 4]; y = x(:, 2);")
        .expect("compile");
    assert!(
        prepared.bytecode.layout.is_some(),
        "supported straight-line scripts should compile through semantic HIR/MIR/VM"
    );
}

#[test]
fn workspace_read_across_submissions_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(session.execute_outcome("x = 42;")).expect("seed workspace");

    let prepared = session.compile_input("x").expect("compile workspace read");
    assert!(
        prepared.bytecode.layout.is_some(),
        "workspace read should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome("x")).expect("read workspace value");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("workspace read should return a durable value");
    assert_eq!(value.to_string(), "42");
}

#[test]
fn char_literal_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let prepared = session
        .compile_input("w = 'hello';")
        .expect("compile char literal assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "char literal assignment should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome("w = 'hello';")).expect("exec succeeds");
    assert!(outcome.flow.is_no_value());
    assert_eq!(outcome.type_info, Some("1x5 char array".to_string()));
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
