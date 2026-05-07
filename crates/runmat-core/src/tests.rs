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
fn end_offset_indexing_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [1, 2, 3]; y = A(end-1);";
    let prepared = session.compile_input(source).expect("compile end indexing");
    assert!(
        prepared.bytecode.layout.is_some(),
        "end-offset indexing should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "2");
}

#[test]
fn end_offset_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [1, 2, 3]; A(end-1) = 9; y = A(2);";
    let prepared = session
        .compile_input(source)
        .expect("compile end assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "end-offset assignment should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "9");
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
fn direct_display_builtins_use_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let prepared = session
        .compile_input("disp('alpha')")
        .expect("compile disp call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "disp should compile through semantic HIR/MIR/VM"
    );
    let outcome = block_on(session.execute_outcome("disp('alpha')")).expect("exec disp");
    let stdout = outcome
        .streams
        .iter()
        .filter(|entry| entry.stream == ExecutionStreamKind::Stdout)
        .map(|entry| entry.text.as_str())
        .collect::<String>();
    assert_eq!(stdout, "alpha\n");

    let prepared = session
        .compile_input("fprintf('foo')")
        .expect("compile fprintf call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "fprintf should compile through semantic HIR/MIR/VM"
    );
    let outcome = block_on(session.execute_outcome("fprintf('foo')")).expect("exec fprintf");
    let stdout = outcome
        .streams
        .iter()
        .filter(|entry| entry.stream == ExecutionStreamKind::Stdout)
        .map(|entry| entry.text.as_str())
        .collect::<String>();
    assert_eq!(stdout, "foo");
}

#[test]
fn simple_builtin_call_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let prepared = session.compile_input("sin(0)").expect("compile sin call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "simple builtin call should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome("sin(0)")).expect("exec sin");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("sin should return a value");
    assert_eq!(value.to_string(), "0");
}

#[test]
fn multi_assign_deal_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let prepared = session
        .compile_input("[a, b] = deal(1, 2)")
        .expect("compile multi-assign");
    assert!(
        prepared.bytecode.layout.is_some(),
        "multi-assign should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome("[a, b] = deal(1, 2)")).expect("exec succeeds");
    let stdout = outcome
        .streams
        .iter()
        .filter(|entry| entry.stream == ExecutionStreamKind::Stdout)
        .map(|entry| entry.text.trim().to_string())
        .collect::<Vec<_>>();
    assert_eq!(stdout, vec!["a = 1", "b = 2"]);
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "a")
            && upsert.value.to_string() == "1"
    }));
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "b")
            && upsert.value.to_string() == "2"
    }));
}

#[test]
fn builtin_function_handle_call_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [0, pi/2]; B = arrayfun(@sin, A);";
    let prepared = session
        .compile_input(source)
        .expect("compile builtin handle call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "builtin function handle call should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "B")
    }));
}

#[test]
fn anonymous_function_handle_call_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [1, 2, 3]; B = arrayfun(@(x) x.^2, A);";
    let prepared = session
        .compile_input(source)
        .expect("compile anonymous handle call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "anonymous function handle call should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    let b = outcome
        .workspace_delta
        .upserts
        .iter()
        .find(|upsert| {
            matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "B")
        })
        .expect("B should be exported");
    assert_eq!(
        b.value.to_string().split_whitespace().collect::<Vec<_>>(),
        vec!["1", "4", "9"]
    );
}

#[test]
fn local_function_call_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "y = inc(2);\nfunction z = inc(x)\n  z = x + 1;\nend";
    let prepared = session
        .compile_input(source)
        .expect("compile local function call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "local function call should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn builtin_call_with_cell_expansion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2}; y = plus(C{:});";
    let prepared = session
        .compile_input(source)
        .expect("compile builtin expansion call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "builtin call with cell expansion should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn multi_assign_builtin_with_cell_expansion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2}; [a, b] = deal(C{:});";
    let prepared = session
        .compile_input(source)
        .expect("compile multi-assign builtin expansion call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "multi-assign builtin expansion should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "a")
            && upsert.value.to_string() == "1"
    }));
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "b")
            && upsert.value.to_string() == "2"
    }));
}

#[test]
fn multi_assign_cell_expansion_rhs_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2}; [a, b] = C{:};";
    let prepared = session
        .compile_input(source)
        .expect("compile multi-assign cell expansion rhs");
    assert!(
        prepared.bytecode.layout.is_some(),
        "multi-assign cell expansion RHS should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "a")
            && upsert.value.to_string() == "1"
    }));
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "b")
            && upsert.value.to_string() == "2"
    }));
}

#[test]
fn cell_end_indexing_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2}; y = C{end};";
    let prepared = session
        .compile_input(source)
        .expect("compile cell end indexing");
    assert!(
        prepared.bytecode.layout.is_some(),
        "cell end indexing should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "2"
    }));
}

#[test]
fn feval_anonymous_handle_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "f = @(x) x + 1; y = feval(f, 2);";
    let prepared = session.compile_input(source).expect("compile feval call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "feval over anonymous handle should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn dynamic_function_handle_call_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "f = @sin; y = f(0);";
    let prepared = session
        .compile_input(source)
        .expect("compile dynamic function handle call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "dynamic function handle call should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "0");
}

#[test]
fn dynamic_anonymous_handle_call_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "f = @(x) x + 1; y = f(2);";
    let prepared = session
        .compile_input(source)
        .expect("compile dynamic anonymous handle call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "dynamic anonymous handle call should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "3");
}

#[test]
fn local_function_multi_output_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "[a, b] = pair(2);\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend";
    let prepared = session
        .compile_input(source)
        .expect("compile local multi-output function call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "local multi-output function call should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "a")
            && upsert.value.to_string() == "2"
    }));
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "b")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn feval_with_cell_expansion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "f = @(x) x + 1; C = {2}; y = feval(f, C{:});";
    let prepared = session
        .compile_input(source)
        .expect("compile feval expansion call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "feval with cell expansion should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalExpandMulti(_))),
        "expanded feval should use the VM feval ABI instruction"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn local_function_with_cell_expansion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {2}; y = inc(C{:});\nfunction z = inc(x)\n  z = x + 1;\nend";
    let prepared = session
        .compile_input(source)
        .expect("compile local function expansion call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "local function with cell expansion should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn local_function_multi_output_with_cell_expansion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source =
        "C = {2}; [a, b] = pair(C{:});\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend";
    let prepared = session
        .compile_input(source)
        .expect("compile local multi-output expansion call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "local multi-output expansion should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "a")
            && upsert.value.to_string() == "2"
    }));
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "b")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn struct_member_access_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "s = struct(); s.a = 3; y = s.a;";
    let prepared = session
        .compile_input(source)
        .expect("compile struct member access");
    assert!(
        prepared.bytecode.layout.is_some(),
        "struct member access should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
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
