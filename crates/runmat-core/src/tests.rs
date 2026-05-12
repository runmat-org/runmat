use crate::*;
use futures::executor::block_on;

fn reset_legacy_user_dispatch_fallback_count() {
    runmat_vm::reset_legacy_user_dispatch_fallback_count();
}

fn assert_no_legacy_user_dispatch_fallback() {
    assert_eq!(
        runmat_vm::legacy_user_dispatch_fallback_count(),
        0,
        "semantic session call should not use legacy user dispatch fallback"
    );
}

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
fn indexed_deletion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [1, 2, 3]; A(2) = []; y = A(2);";
    let prepared = session
        .compile_input(source)
        .expect("compile indexed deletion");
    assert!(
        prepared.bytecode.layout.is_some(),
        "indexed deletion should compile through semantic HIR/MIR/VM"
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
fn complex_matrix_literal_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [1+1i, 2+2i, 3+3i]; y = A(2);";
    let prepared = session
        .compile_input(source)
        .expect("compile complex matrix literal");
    assert!(
        prepared.bytecode.layout.is_some(),
        "complex matrix literal should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "2+2i");
}

#[test]
fn complex_indexed_deletion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [1+1i, 2+2i, 3+3i]; A(2) = []; y = A(2);";
    let prepared = session
        .compile_input(source)
        .expect("compile complex indexed deletion");
    assert!(
        prepared.bytecode.layout.is_some(),
        "complex indexed deletion should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "3+3i");
}

#[test]
fn real_tensor_complex_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [1, 2, 3]; A(2) = 4+5i; y = A(2);";
    let prepared = session
        .compile_input(source)
        .expect("compile real tensor complex assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "real tensor complex assignment should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "4+5i");
}

#[test]
fn real_tensor_2d_complex_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [1, 2; 3, 4]; A(1, 2) = 5+6i; y = A(1, 2);";
    let prepared = session
        .compile_input(source)
        .expect("compile real tensor 2d complex assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "real tensor 2d complex assignment should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "5+6i");
}

#[test]
fn cell_paren_deletion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2, 3}; C(2) = []; y = C{2};";
    let prepared = session
        .compile_input(source)
        .expect("compile cell paren deletion");
    assert!(
        prepared.bytecode.layout.is_some(),
        "cell paren deletion should compile through semantic HIR/MIR/VM"
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
fn cell_paren_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2, 3}; C(2) = {4}; y = C{2};";
    let prepared = session
        .compile_input(source)
        .expect("compile cell paren assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "cell paren assignment should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "4");
}

#[test]
fn cell_2d_paren_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2; 3, 4}; C(1, 2) = {9}; y = C{1, 2};";
    let prepared = session
        .compile_input(source)
        .expect("compile 2d cell paren assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "2d cell paren assignment should compile through semantic HIR/MIR/VM"
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
fn cell_2d_linear_indexing_is_column_major_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2; 3, 4}; y = C{3};";
    let prepared = session
        .compile_input(source)
        .expect("compile 2d cell linear indexing");
    assert!(
        prepared.bytecode.layout.is_some(),
        "2d cell linear indexing should compile through semantic HIR/MIR/VM"
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
fn cell_2d_expansion_is_column_major_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2; 3, 4}; [a, b, c, d] = C{:}; y = c;";
    let prepared = session
        .compile_input(source)
        .expect("compile 2d cell expansion");
    assert!(
        prepared.bytecode.layout.is_some(),
        "2d cell expansion should compile through semantic HIR/MIR/VM"
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
fn if_else_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "x = 2; if x > 1; y = 10; else; y = 20; end";
    let prepared = session.compile_input(source).expect("compile if else");
    assert!(
        prepared.bytecode.layout.is_some(),
        "if/else should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "10");
}

#[test]
fn switch_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "x = 2; switch x; case 1; y = 10; case 2; y = 20; otherwise; y = 30; end";
    let prepared = session.compile_input(source).expect("compile switch");
    assert!(
        prepared.bytecode.layout.is_some(),
        "switch should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "20");
}

#[test]
fn global_statement_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "global g; g = 7; y = g;";
    let prepared = session.compile_input(source).expect("compile global");
    assert!(
        prepared.bytecode.layout.is_some(),
        "global statement should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "7");
}

#[test]
fn range_slice_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [1, 2, 3, 4]; B = A(2:3); y = B(2);";
    let prepared = session.compile_input(source).expect("compile range slice");
    assert!(
        prepared.bytecode.layout.is_some(),
        "range slice should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::IndexSlice(..))),
        "range indexing should lower to slice bytecode"
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
fn end_expression_user_function_call_uses_semantic_identity() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source =
        "function y = pick(n)\n  y = n;\nend\nx = [10 20 30 40 50 60 70 80]; a = x(pick(end-3));";
    let prepared = session
        .compile_input(source)
        .expect("compile end-expression function call");
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::IndexSliceExpr { end_numeric_exprs, .. }
                if end_numeric_exprs.iter().any(|(_, expr)| matches!(
                    expr,
                    runmat_vm::EndExpr::SemanticCall(_, name, _) if name == "pick"
                ))
        )),
        "end-expression user calls should carry semantic function identity"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "a")
            && upsert.value.to_string() == "50"
    }));
}

#[test]
fn end_expression_session_function_call_uses_semantic_identity() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(session.execute_outcome("seed = 0;\nfunction y = pick(n)\n  y = n;\nend"))
        .expect("define session function");
    let source = "x = [10 20 30 40 50 60 70 80]; a = x(pick(end-3));";
    let prepared = session
        .compile_input(source)
        .expect("compile session end-expression function call");
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::IndexSliceExpr { end_numeric_exprs, .. }
                if end_numeric_exprs.iter().any(|(_, expr)| matches!(
                    expr,
                    runmat_vm::EndExpr::SemanticCall(_, name, _) if name == "pick"
                ))
        )),
        "session end-expression user calls should carry semantic function identity"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "a")
            && upsert.value.to_string() == "50"
    }));
}

#[test]
fn for_range_loop_uses_semantic_vm_without_rerunning_prefix() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(session.execute_outcome("prefix = 0;")).expect("seed prefix");
    let source = "prefix = prefix + 1; s = 0; for i = 1:3; s = s + i; end; y = s + prefix;";
    let prepared = session.compile_input(source).expect("compile for loop");
    assert!(
        prepared.bytecode.layout.is_some(),
        "for range loops should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let y_outcome = block_on(session.execute_outcome("y")).expect("read y");
    let y = y_outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(y.to_string(), "7");
    let s_outcome = block_on(session.execute_outcome("s")).expect("read s");
    let s = s_outcome
        .flow
        .durable_workspace_value()
        .expect("s should be readable from workspace");
    assert_eq!(s.to_string(), "6");
    let prefix_outcome = block_on(session.execute_outcome("prefix")).expect("read prefix");
    let prefix = prefix_outcome
        .flow
        .durable_workspace_value()
        .expect("prefix should be readable from workspace");
    assert_eq!(prefix.to_string(), "1");
}

#[test]
fn while_loop_uses_semantic_vm_without_rerunning_prefix() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "x = 0; while x < 3; x = x + 1; end; y = x;";
    let prepared = session.compile_input(source).expect("compile while loop");
    assert!(
        prepared.bytecode.layout.is_some(),
        "while loops should compile through semantic HIR/MIR/VM"
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
fn range_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [1, 2, 3, 4]; A(2:3) = 9; y = A(3);";
    let prepared = session
        .compile_input(source)
        .expect("compile range assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "range assignment should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::StoreSlice(..))),
        "range assignment should lower to slice store bytecode"
    );
    assert!(
        !prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::StoreIndex(1))),
        "range assignment should not rely on StoreIndex fallback"
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
fn range_assignment_vector_rhs_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [1, 2, 3, 4]; A(2:3) = [8, 9]; y = A(3);";
    let prepared = session
        .compile_input(source)
        .expect("compile range vector assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "range vector assignment should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::StoreSlice(..))),
        "range vector assignment should lower to slice store bytecode"
    );
    assert!(
        !prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::StoreIndex(1))),
        "range vector assignment should not rely on StoreIndex fallback"
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
fn range_complex_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [1, 2, 3, 4]; A(2:3) = 8+9i; y = A(3);";
    let prepared = session
        .compile_input(source)
        .expect("compile complex range assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "complex range assignment should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "8+9i");
}

#[test]
fn logical_indexing_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [10, 20, 30]; B = A([true, false, true]); y = B(2);";
    let prepared = session
        .compile_input(source)
        .expect("compile logical indexing");
    assert!(
        prepared.bytecode.layout.is_some(),
        "logical indexing should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::IndexSlice(..))),
        "logical indexing should lower to slice bytecode"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "30");
}

#[test]
fn logical_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [10, 20, 30]; A([true, false, true]) = 5; y = A(3);";
    let prepared = session
        .compile_input(source)
        .expect("compile logical assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "logical assignment should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::StoreSlice(..))),
        "logical assignment should lower to slice store bytecode"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "5");
}

#[test]
fn mixed_logical_numeric_matrix_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [true, 2, false]; y = A(2);";
    let prepared = session
        .compile_input(source)
        .expect("compile mixed logical numeric matrix");
    assert!(
        prepared.bytecode.layout.is_some(),
        "mixed logical numeric matrix should compile through semantic HIR/MIR/VM"
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
fn mixed_logical_complex_matrix_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [true, 2+3i, false]; y = A(1);";
    let prepared = session
        .compile_input(source)
        .expect("compile mixed logical complex matrix");
    assert!(
        prepared.bytecode.layout.is_some(),
        "mixed logical complex matrix should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "1");
}

#[test]
fn cell_2d_range_expansion_is_column_major_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2; 3, 4}; [a, b] = C{2:3}; y = b;";
    let prepared = session
        .compile_input(source)
        .expect("compile 2d cell range expansion");
    assert!(
        prepared.bytecode.layout.is_some(),
        "2d cell range expansion should compile through semantic HIR/MIR/VM"
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
fn cell_range_paren_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2, 3, 4}; C(2:3) = {8, 9}; y = C{3};";
    let prepared = session
        .compile_input(source)
        .expect("compile cell range paren assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "cell range paren assignment should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::StoreSlice(..))),
        "cell range paren assignment should lower to slice store bytecode"
    );
    assert!(
        !prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::StoreIndex(1))),
        "cell range paren assignment should not rely on StoreIndex fallback"
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
fn cell_range_deletion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2, 3, 4}; C(2:3) = []; y = C{2};";
    let prepared = session
        .compile_input(source)
        .expect("compile cell range deletion");
    assert!(
        prepared.bytecode.layout.is_some(),
        "cell range deletion should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("y")).expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "4");
}

#[test]
fn cell_colon_paren_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2}; C(:) = {8, 9}; y = C{2};";
    let prepared = session
        .compile_input(source)
        .expect("compile cell colon paren assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "cell colon paren assignment should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::StoreSlice(..))),
        "cell colon paren assignment should lower to slice store bytecode"
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
fn elementwise_logical_ops_use_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "a = 1 & 0; b = 1 | 0; c = ~0;";
    let prepared = session
        .compile_input(source)
        .expect("compile elementwise logical ops");
    assert!(
        prepared.bytecode.layout.is_some(),
        "elementwise logical ops should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::LogicalAnd)),
        "elementwise and should lower to typed logical bytecode"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::LogicalOr)),
        "elementwise or should lower to typed logical bytecode"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::LogicalNot)),
        "logical not should lower to typed logical bytecode"
    );
    assert!(
        !prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallBuiltin(name, _) if matches!(name.as_str(), "and" | "or" | "not")
        )),
        "semantic logical ops should not lower through string-keyed builtin calls"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "a")
            && upsert.value.to_string() == "0"
    }));
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "b")
            && upsert.value.to_string() == "1"
    }));
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "c")
            && upsert.value.to_string() == "1"
    }));
}

#[test]
fn short_circuit_logical_ops_use_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "a = 0 && 1; b = 1 || 0;";
    let prepared = session
        .compile_input(source)
        .expect("compile short-circuit logical ops");
    assert!(
        prepared.bytecode.layout.is_some(),
        "short-circuit logical ops should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "a")
            && upsert.value.to_string() == "0"
    }));
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "b")
            && upsert.value.to_string() == "1"
    }));
}

#[test]
fn metaclass_literal_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "c = ?Point;";
    let prepared = session
        .compile_input(source)
        .expect("compile metaclass literal");
    assert!(
        prepared.bytecode.layout.is_some(),
        "metaclass literals should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "c")
            && upsert.value.to_string() == "'Point'"
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

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
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
fn builtin_call_with_cell_end_expansion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2}; y = plus(C{end}, 1);";
    let prepared = session
        .compile_input(source)
        .expect("compile builtin cell end expansion call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "builtin call with cell end expansion should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn builtin_call_with_cell_end_offset_expansion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2, 3}; y = plus(C{end-1}, 1);";
    let prepared = session
        .compile_input(source)
        .expect("compile builtin cell end-offset expansion call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "builtin call with cell end-offset expansion should compile through semantic HIR/MIR/VM"
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
fn multi_assign_cell_end_rhs_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2}; [a] = C{end};";
    let prepared = session
        .compile_input(source)
        .expect("compile multi-assign cell end rhs");
    assert!(
        prepared.bytecode.layout.is_some(),
        "multi-assign cell end rhs should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("a")).expect("read a");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("a should be readable from workspace");
    assert_eq!(value.to_string(), "2");
}

#[test]
fn multi_assign_cell_end_offset_rhs_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2, 3}; [a] = C{end-1};";
    let prepared = session
        .compile_input(source)
        .expect("compile multi-assign cell end-offset rhs");
    assert!(
        prepared.bytecode.layout.is_some(),
        "multi-assign cell end-offset rhs should compile through semantic HIR/MIR/VM"
    );

    block_on(session.execute_outcome(source)).expect("exec succeeds");
    let outcome = block_on(session.execute_outcome("a")).expect("read a");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("a should be readable from workspace");
    assert_eq!(value.to_string(), "2");
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
fn cell_end_offset_indexing_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2, 3}; y = C{end-1};";
    let prepared = session
        .compile_input(source)
        .expect("compile cell end-offset indexing");
    assert!(
        prepared.bytecode.layout.is_some(),
        "cell end-offset indexing should compile through semantic HIR/MIR/VM"
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
fn cell_end_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2}; C{end} = 9; y = C{2};";
    let prepared = session
        .compile_input(source)
        .expect("compile cell end assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "cell end assignment should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "9"
    }));
}

#[test]
fn cell_end_offset_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2, 3}; C{end-1} = 9; y = C{2};";
    let prepared = session
        .compile_input(source)
        .expect("compile cell end-offset assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "cell end-offset assignment should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "9"
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
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFeval(1))),
        "feval should lower from dynamic MIR callee to the VM feval ABI instruction"
    );
    assert!(
        !prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallBuiltin(name, _) if name == "feval"
        )),
        "feval should not lower as a string-keyed builtin call"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn feval_string_local_function_uses_semantic_handle() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "y = feval('inc', 2);\nfunction z = inc(x)\n  z = x + 1;\nend";
    let prepared = session
        .compile_input(source)
        .expect("compile feval string local function call");
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateSemanticFunctionHandle(_, name) if name == "inc"
        )),
        "local feval string callee should lower to a semantic function handle"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
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
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateFunctionHandle(name) if name == "sin"
        )),
        "function handle literals should lower to typed function-handle bytecode"
    );
    assert!(
        !prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallBuiltin(name, 1) if name == "make_handle"
        )),
        "function handle literals should not lower through the internal make_handle builtin"
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
fn local_function_handle_uses_semantic_handle() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "f = @inc; y = f(2);\nfunction z = inc(x)\n  z = x + 1;\nend";
    let prepared = session
        .compile_input(source)
        .expect("compile local function handle call");
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateSemanticFunctionHandle(_, name) if name == "inc"
        )),
        "local function handles should lower to semantic function handles"
    );
    assert!(
        !prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateSemanticClosure(_, name, 0) if name == "inc"
        )),
        "zero-capture local function handles should not lower as closures"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
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
fn dynamic_function_handle_multi_output_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source =
        "f = @pair; [a, b] = feval(f, 2);\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend";
    let prepared = session
        .compile_input(source)
        .expect("compile dynamic multi-output function handle call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "dynamic multi-output function handle call should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallFevalMulti(_, 2)
        )),
        "dynamic multi-output function handle call should lower to typed feval multi-output bytecode"
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
fn dynamic_function_handle_multi_output_with_expansion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source =
        "f = @pair; C = {2}; [a, b] = feval(f, C{:});\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend";
    let prepared = session
        .compile_input(source)
        .expect("compile dynamic multi-output expansion call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "dynamic multi-output expansion call should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalExpandMultiOutput(_, 2))),
        "dynamic multi-output expansion call should lower to typed feval expansion bytecode"
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
fn try_catch_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "try\n  error('boom');\n  y = 1;\ncatch\n  y = 2;\nend";
    let prepared = session.compile_input(source).expect("compile try/catch");
    assert!(
        prepared.bytecode.layout.is_some(),
        "try/catch should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::EnterTry(_, None))),
        "try/catch should lower to typed exception bytecode"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "2"
    }));
}

#[test]
fn try_catch_binding_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "try\n  error('boom');\ncatch err\n  y = err.message;\nend";
    let prepared = session
        .compile_input(source)
        .expect("compile try/catch binding");
    assert!(
        prepared.bytecode.layout.is_some(),
        "try/catch binding should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::EnterTry(_, Some(_)))),
        "try/catch binding should lower the catch binding into exception bytecode"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string().contains("boom")
    }));
}

#[test]
fn indexed_member_slice_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "s = struct(); s.a = [1 2 3]; s.a(2:3) = [4 5]; y = s.a(3);";
    let prepared = session
        .compile_input(source)
        .expect("compile indexed member slice assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "indexed member slice assignment should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::StoreSlice(..))),
        "indexed member slice assignment should lower to typed slice store bytecode"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "5"
    }));
}

#[test]
fn dotted_member_index_call_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "s = struct(); s.a = [1 2 3]; y = s.a(2);";
    let prepared = session
        .compile_input(source)
        .expect("compile dotted member index call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "dotted member index call should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallMethodOrMemberIndex(name, 1) if name == "a"
        )),
        "dotted member index call should lower to typed method/member-index bytecode"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "2"
    }));
}

#[test]
fn dotted_member_index_expansion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "s = struct(); s.a = [10 20 30]; C = {2}; y = s.a(C{:});";
    let prepared = session
        .compile_input(source)
        .expect("compile expanded dotted member index call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "expanded dotted member index call should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallMethodOrMemberIndexExpandMulti(name, _) if name == "a"
        )),
        "expanded dotted member index call should lower to typed expansion bytecode"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "20"
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
    assert!(
        !prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallBuiltin(name, _) if name == "feval"
        )),
        "expanded feval should not lower as a string-keyed builtin call"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn feval_with_2d_cell_expansion_is_column_major_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "f = @(a, b, c, d) c; C = {1, 2; 3, 4}; y = feval(f, C{:});";
    let prepared = session
        .compile_input(source)
        .expect("compile 2d feval expansion call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "2d feval expansion should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalExpandMulti(_))),
        "2d expanded feval should use the VM feval ABI instruction"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "2"
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

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn cellfun_named_local_function_uses_semantic_callback() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source =
        "C = {2}; B = cellfun('inc', C); y = B(1);\nfunction z = inc(x)\n  z = x + 1;\nend";
    let prepared = session
        .compile_input(source)
        .expect("compile named local function callback");
    assert!(
        prepared.bytecode.layout.is_some(),
        "named local callback should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .semantic_function_registry
            .resolve_name("inc")
            .is_some(),
        "local callback target should be available as semantic function bytecode"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateSemanticFunctionHandle(_, name) if name == "inc"
        )),
        "local string callback should be bound to a semantic function handle"
    );
    assert!(
        prepared.bytecode.functions.is_empty(),
        "semantic compile should not require legacy user-function bytecode entries"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn cellfun_session_function_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(session.execute_outcome("seed = 0;\nfunction z = inc(x)\n  z = x + 1;\nend"))
        .expect("define session function");

    let source = "C = {2}; B = cellfun('inc', C); y = B(1);";
    let prepared = session
        .compile_input(source)
        .expect("compile callback using session function");
    let inc_id = prepared
        .bytecode
        .semantic_function_registry
        .resolve_name("inc")
        .expect("session callback target should resolve through semantic function registry");
    assert!(
        prepared
            .bytecode
            .semantic_function_registry
            .get(inc_id)
            .and_then(|function| function.source_id)
            .is_some(),
        "session semantic callback target should retain source metadata"
    );
    let inc_source = prepared
        .bytecode
        .semantic_function_registry
        .get(inc_id)
        .and_then(|function| function.source_id)
        .expect("inc source metadata");
    assert!(
        prepared
            .bytecode
            .semantic_function_registry
            .functions_for_source(inc_source)
            .contains(&inc_id),
        "session semantic callback target should be indexed by source ownership"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateSemanticFunctionHandle(_, name) if name == "inc"
        )),
        "session string callback should be bound to a semantic function handle"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn arrayfun_named_local_function_uses_semantic_callback() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source =
        "A = [2, 3]; B = arrayfun('inc', A); y = B(2);\nfunction z = inc(x)\n  z = x + 1;\nend";
    let prepared = session
        .compile_input(source)
        .expect("compile named local arrayfun callback");
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateSemanticFunctionHandle(_, name) if name == "inc"
        )),
        "local arrayfun string callback should be bound to a semantic function handle"
    );
    assert!(
        prepared.bytecode.functions.is_empty(),
        "semantic arrayfun callback should not require legacy user-function bytecode entries"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "4"
    }));
}

#[test]
fn arrayfun_session_function_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(session.execute_outcome("seed = 0;\nfunction z = inc(x)\n  z = x + 1;\nend"))
        .expect("define session function");

    let source = "A = [2, 3]; B = arrayfun('inc', A); y = B(2);";
    let prepared = session
        .compile_input(source)
        .expect("compile session arrayfun callback");
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateSemanticFunctionHandle(_, name) if name == "inc"
        )),
        "session arrayfun string callback should be bound to a semantic function handle"
    );
    assert!(
        prepared.bytecode.functions.is_empty(),
        "session semantic arrayfun callback should not require legacy bytecode functions"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "4"
    }));
}

#[test]
fn direct_session_function_call_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(session.execute_outcome("seed = 0;\nfunction z = inc(x)\n  z = x + 1;\nend"))
        .expect("define session function");

    let prepared = session
        .compile_input("y = inc(2);")
        .expect("compile direct session function call");
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallSemanticFunction(_, 1))),
        "direct call should lower to semantic function bytecode"
    );
    assert!(
        prepared.bytecode.functions.is_empty(),
        "direct session semantic call should not require legacy bytecode functions"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome("y = inc(2);")).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn direct_session_function_multi_output_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(
        session
            .execute_outcome("seed = 0;\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend"),
    )
    .expect("define session function");

    let prepared = session
        .compile_input("[a, b] = pair(2);")
        .expect("compile direct session multi-output function call");
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallSemanticFunctionMulti(_, 1, 2))),
        "direct multi-output call should lower to semantic function bytecode"
    );
    assert!(
        prepared.bytecode.functions.is_empty(),
        "direct session semantic multi-output call should not require legacy bytecode functions"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome("[a, b] = pair(2);")).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
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
fn direct_session_function_cell_expansion_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(session.execute_outcome("seed = 0;\nfunction z = inc(x)\n  z = x + 1;\nend"))
        .expect("define session function");

    let prepared = session
        .compile_input("C = {2}; y = inc(C{:});")
        .expect("compile direct session expansion function call");
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallSemanticFunctionExpandMulti(_, _)
        )),
        "direct expansion call should lower to semantic function bytecode"
    );
    assert!(
        prepared.bytecode.functions.is_empty(),
        "direct session semantic expansion call should not require legacy bytecode functions"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome =
        block_on(session.execute_outcome("C = {2}; y = inc(C{:});")).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn direct_session_function_expansion_multi_output_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(
        session
            .execute_outcome("seed = 0;\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend"),
    )
    .expect("define session function");

    let prepared = session
        .compile_input("C = {2}; [a, b] = pair(C{:});")
        .expect("compile direct session expansion multi-output function call");
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallSemanticFunctionExpandMultiOutput(_, _, 2)
        )),
        "direct expansion multi-output call should lower to semantic function bytecode"
    );
    assert!(
        prepared.bytecode.functions.is_empty(),
        "direct session semantic expansion multi-output call should not require legacy bytecode functions"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome =
        block_on(session.execute_outcome("C = {2}; [a, b] = pair(C{:});")).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
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
fn session_function_handle_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(session.execute_outcome("seed = 0;\nfunction z = inc(x)\n  z = x + 1;\nend"))
        .expect("define session function");

    let prepared = session
        .compile_input("f = @inc; y = f(2);")
        .expect("compile function handle session call");
    assert!(
        prepared
            .bytecode
            .semantic_function_registry
            .resolve_name("inc")
            .is_some(),
        "function handle target should be present in semantic registry"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateSemanticFunctionHandle(_, name) if name == "inc"
        )),
        "session function handles should carry semantic identity"
    );
    assert!(
        !prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateFunctionHandle(name) if name == "inc"
        )),
        "session function handles should not remain name-only handles"
    );
    assert!(
        prepared.bytecode.functions.is_empty(),
        "function handle session semantic call should not require legacy bytecode functions"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome("f = @inc; y = f(2);"))
        .expect("function handle call succeeds");
    assert_no_legacy_user_dispatch_fallback();
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn session_function_handle_feval_multi_output_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(
        session
            .execute_outcome("seed = 0;\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend"),
    )
    .expect("define session function");

    let prepared = session
        .compile_input("f = @pair; [a, b] = feval(f, 2);")
        .expect("compile feval multi-output session handle call");
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalMulti(_, 2))),
        "function handle feval multi-output call should use typed feval bytecode"
    );
    assert!(
        prepared.bytecode.functions.is_empty(),
        "function handle feval session semantic call should not require legacy bytecode functions"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome("f = @pair; [a, b] = feval(f, 2);"))
        .expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
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
fn session_feval_string_multi_output_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(
        session
            .execute_outcome("seed = 0;\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend"),
    )
    .expect("define session function");

    let prepared = session
        .compile_input("[a, b] = feval('pair', 2);")
        .expect("compile feval string multi-output session call");
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateSemanticFunctionHandle(_, name) if name == "pair"
        )),
        "session feval string callee should carry semantic identity"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalMulti(_, 2))),
        "session feval string multi-output call should use typed feval bytecode"
    );
    assert!(
        prepared.bytecode.functions.is_empty(),
        "session feval string semantic call should not require legacy bytecode functions"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome =
        block_on(session.execute_outcome("[a, b] = feval('pair', 2);")).expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
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
fn session_function_handle_feval_expansion_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(session.execute_outcome("seed = 0;\nfunction z = add2(a, b)\n  z = a + b;\nend"))
        .expect("define session function");

    let prepared = session
        .compile_input("f = @add2; C = {2, 3}; y = feval(f, C{:});")
        .expect("compile feval expansion session handle call");
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalExpandMulti(_))),
        "function handle feval expansion call should use typed feval expansion bytecode"
    );
    assert!(
        prepared.bytecode.functions.is_empty(),
        "function handle feval expansion session semantic call should not require legacy bytecode functions"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome("f = @add2; C = {2, 3}; y = feval(f, C{:});"))
        .expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "5"
    }));
}

#[test]
fn session_function_handle_feval_expansion_multi_output_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(
        session
            .execute_outcome("seed = 0;\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend"),
    )
    .expect("define session function");

    let prepared = session
        .compile_input("f = @pair; C = {2}; [a, b] = feval(f, C{:});")
        .expect("compile feval expansion multi-output session handle call");
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallFevalExpandMultiOutput(_, 2)
        )),
        "function handle feval expansion multi-output call should use typed feval expansion bytecode"
    );
    assert!(
        prepared.bytecode.functions.is_empty(),
        "function handle feval expansion multi-output session semantic call should not require legacy bytecode functions"
    );

    reset_legacy_user_dispatch_fallback_count();
    let outcome = block_on(session.execute_outcome("f = @pair; C = {2}; [a, b] = feval(f, C{:});"))
        .expect("exec succeeds");
    assert_no_legacy_user_dispatch_fallback();
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
fn session_semantic_registry_replaces_redefined_function() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(session.execute_outcome("seed = 0;\nfunction z = inc(x)\n  z = x + 1;\nend"))
        .expect("define first function");
    block_on(session.execute_outcome("seed = 0;\nfunction z = inc(x)\n  z = x + 10;\nend"))
        .expect("redefine function");

    let source = "C = {2}; B = cellfun('inc', C); y = B(1);";
    let prepared = session
        .compile_input(source)
        .expect("compile callback using redefined session function");
    assert_eq!(
        prepared
            .bytecode
            .semantic_function_registry
            .functions
            .values()
            .filter(|function| function.display_name == "inc")
            .count(),
        1,
        "semantic registry should retire the old definition"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "12"
    }));
}

#[test]
fn session_semantic_registry_retires_replaced_source_group() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    block_on(session.execute_outcome(
        "seed = 0;\nfunction z = inc(x)\n  z = x + 1;\nend\nfunction z = dec(x)\n  z = x - 1;\nend",
    ))
    .expect("define source function group");
    block_on(session.execute_outcome("seed = 0;\nfunction z = inc(x)\n  z = x + 10;\nend"))
        .expect("replace one function from group");

    let prepared = session
        .compile_input("y = inc(2);")
        .expect("compile using replacement function");
    assert!(
        prepared
            .bytecode
            .semantic_function_registry
            .resolve_name("inc")
            .is_some(),
        "replacement function should remain in semantic registry"
    );
    assert!(
        prepared
            .bytecode
            .semantic_function_registry
            .resolve_name("dec")
            .is_none(),
        "other functions from the replaced source should be retired"
    );
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
fn nested_struct_member_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "s = struct(); s.a.b = 3; y = s.a.b;";
    let prepared = session
        .compile_input(source)
        .expect("compile nested struct member assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "nested struct member assignment should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn nested_dynamic_member_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "s = struct(); name = 'a'; s.(name).b = 4; y = s.a.b;";
    let prepared = session
        .compile_input(source)
        .expect("compile nested dynamic member assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "nested dynamic member assignment should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "4"
    }));
}

#[test]
fn indexed_cell_member_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {struct()}; C{1}.a = 5; y = C{1}.a;";
    let prepared = session
        .compile_input(source)
        .expect("compile indexed cell member assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "indexed cell member assignment should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "5"
    }));
}

#[test]
fn cell_member_access_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {struct(), struct()}; C{1}.a = 5; C{2}.a = 6; D = C.a; y = D{2};";
    let prepared = session
        .compile_input(source)
        .expect("compile cell member access");
    assert!(
        prepared.bytecode.layout.is_some(),
        "cell member access should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "6"
    }));
}

#[test]
fn cell_member_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {struct(), struct()}; C.a = 9; y = C{2}.a;";
    let prepared = session
        .compile_input(source)
        .expect("compile cell member assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "cell member assignment should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "9"
    }));
}

#[test]
fn indexed_cell_end_offset_member_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {struct(), struct(), struct()}; C{end-1}.a = 7; y = C{2}.a;";
    let prepared = session
        .compile_input(source)
        .expect("compile indexed cell end-offset member assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "indexed cell end-offset member assignment should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "7"
    }));
}

#[test]
fn scalar_struct_paren_member_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "s = struct(); s(1).a = 5; y = s(1).a;";
    let prepared = session
        .compile_input(source)
        .expect("compile scalar struct paren member assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "scalar struct paren member assignment should compile through semantic HIR/MIR/VM"
    );

    let outcome = block_on(session.execute_outcome(source)).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "5"
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
