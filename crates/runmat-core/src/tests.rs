use crate::*;
use futures::executor::block_on;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

static CWD_LOCK: Mutex<()> = Mutex::new(());
const DEEP_SEMANTIC_TEST_STACK_BYTES: usize = 32 * 1024 * 1024;

struct CwdGuard {
    original: PathBuf,
}

impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.original);
    }
}

fn push_cwd(path: &Path) -> CwdGuard {
    let original = std::env::current_dir().expect("read cwd");
    std::env::set_current_dir(path).expect("set cwd");
    CwdGuard { original }
}

fn run_deep_semantic_test(f: impl FnOnce() + Send + 'static) {
    let handle = std::thread::Builder::new()
        .name("runmat-core-deep-semantic-test".to_string())
        .stack_size(DEEP_SEMANTIC_TEST_STACK_BYTES)
        .spawn(f)
        .expect("spawn deep semantic test thread");
    if let Err(panic) = handle.join() {
        std::panic::resume_unwind(panic);
    }
}

fn end_expr_contains_display_name(expr: &runmat_vm::EndExpr, name: &str) -> bool {
    use runmat_vm::EndExpr;
    match expr {
        EndExpr::ResolvedCall { identity, args, .. } => {
            identity.display_name().as_deref() == Some(name)
                || args
                    .iter()
                    .any(|arg| end_expr_contains_display_name(arg, name))
        }
        EndExpr::Add(lhs, rhs)
        | EndExpr::Sub(lhs, rhs)
        | EndExpr::Mul(lhs, rhs)
        | EndExpr::Div(lhs, rhs)
        | EndExpr::LeftDiv(lhs, rhs)
        | EndExpr::Pow(lhs, rhs) => {
            end_expr_contains_display_name(lhs, name) || end_expr_contains_display_name(rhs, name)
        }
        EndExpr::Neg(inner)
        | EndExpr::Pos(inner)
        | EndExpr::Floor(inner)
        | EndExpr::Ceil(inner)
        | EndExpr::Round(inner)
        | EndExpr::Fix(inner) => end_expr_contains_display_name(inner, name),
        EndExpr::End | EndExpr::Const(_) | EndExpr::Var(_) => false,
    }
}

fn execute_text_request(
    session: &mut RunMatSession,
    source_text: &str,
) -> Result<abi::ExecutionOutcome, RunError> {
    execute_text_request_named_source(session, "<test>", source_text)
}

fn execute_text_request_named_source(
    session: &mut RunMatSession,
    source_name: &str,
    source_text: &str,
) -> Result<abi::ExecutionOutcome, RunError> {
    let request = abi::ExecutionRequest::for_source(
        abi::SourceInput::Text {
            name: source_name.to_string(),
            text: source_text.to_string(),
        },
        session.compat_mode(),
        abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    );
    block_on(session.execute_request(request)).result
}

fn execute_path_request(
    session: &mut RunMatSession,
    source_path: &str,
) -> Result<abi::ExecutionOutcome, RunError> {
    let request = abi::ExecutionRequest::for_source(
        abi::SourceInput::Path(source_path.to_string()),
        session.compat_mode(),
        abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    );
    block_on(session.execute_request(request)).result
}

fn outcome_has_named_upsert(
    outcome: &abi::ExecutionOutcome,
    name: &str,
    expected: &runmat_builtins::Value,
) -> bool {
    outcome.workspace_delta.upserts.iter().any(|upsert| {
        let matches_name = match &upsert.key {
            abi::WorkspaceBindingKey::Interactive {
                name: binding_name, ..
            } => binding_name.0 == name,
            abi::WorkspaceBindingKey::SourceBinding { binding, .. } => binding.0 == name,
            abi::WorkspaceBindingKey::Global { .. }
            | abi::WorkspaceBindingKey::Persistent { .. } => false,
        };
        matches_name && upsert.value == *expected
    })
}

fn outcome_has_upsert_name(outcome: &abi::ExecutionOutcome, name: &str) -> bool {
    outcome
        .workspace_delta
        .upserts
        .iter()
        .any(|upsert| match &upsert.key {
            abi::WorkspaceBindingKey::Interactive {
                name: binding_name, ..
            } => binding_name.0 == name,
            abi::WorkspaceBindingKey::SourceBinding { binding, .. } => binding.0 == name,
            abi::WorkspaceBindingKey::Global { .. }
            | abi::WorkspaceBindingKey::Persistent { .. } => false,
        })
}

#[test]
fn captures_basic_workspace_assignments() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let result = execute_text_request(&mut session, "x = 42;").expect("exec succeeds");
    assert!(
        result.workspace_delta.upserts.iter().any(|upsert| {
            matches!(
                &upsert.key,
                abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "x"
            )
        }),
        "workspace snapshot should include assigned variable"
    );
}

#[test]
fn execute_outcome_exposes_runtime_flow() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = execute_text_request(&mut session, "1 + 1").expect("exec succeeds");
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
    let outcome = execute_text_request(&mut session, "x = 42;").expect("exec succeeds");
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
fn execute_text_request_accepts_function_arguments_block_syntax() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x (1,1) double
            end
            y = x * 2;
        end
        r = typed(3);
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(6.0)),
        "arguments block syntax should parse and execute function body"
    );
}

#[test]
fn execute_text_request_accepts_function_arguments_input_block_attribute() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments (Input)
                x (1,1) double
            end
            y = x * 2;
        end
        r = typed(3);
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(6.0)),
        "arguments (Input) block syntax should parse and execute function body"
    );
}

#[test]
fn execute_text_request_enforces_function_arguments_size_and_class() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x (1,1) double
            end
            y = x * 2;
        end
        a = typed(3);
        try, b = typed([1 2]); sid = 'BAD'; catch e, sid = e.identifier; end
        try, c = typed("x"); tid = 'BAD'; catch e, tid = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(6.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "sid",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationSize".to_string())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "tid",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationClass".to_string())
    ));
}

#[test]
fn execute_text_request_rejects_arguments_block_unknown_parameter_declaration() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                z (1,1) double
            end
            y = x * 2;
        end
        r = typed(3);
    "#;
    let err = execute_text_request(&mut session, source).expect_err("expected semantic failure");
    let RunError::Semantic(err) = err else {
        panic!("expected semantic error for invalid arguments declaration");
    };
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:FunctionArgumentValidationUnknown")
    );
}

#[test]
fn execute_text_request_enforces_arguments_must_be_finite_validator() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x (1,1) double mustBeFinite
            end
            y = x * 2;
        end
        a = typed(3);
        try, b = typed(0/0); fid = 'BAD'; catch e, fid = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(6.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "fid",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_treats_single_token_arguments_constraint_as_class_name() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x mustBeNope
            end
            y = 1;
        end
        try, typed(3); cid = 'BAD'; catch e, cid = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "cid",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationClass".to_string())
    ));
}

#[test]
fn execute_text_request_must_be_finite_accepts_char_and_logical() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x mustBeFinite
            end
            y = 1;
        end
        a = typed('hello');
        b = typed(true);
        try, typed(0/0); zid = 'BAD'; catch e, zid = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "zid",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_enforces_arguments_must_be_text_validator() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x mustBeText
            end
            y = 1;
        end
        a = typed("hello");
        b = typed('world');
        c = typed({'a', "b"});
        try, typed(3); tid = 'BAD'; catch e, tid = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "c",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "tid",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_enforces_arguments_must_be_nonempty_validator() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x mustBeNonempty
            end
            y = 1;
        end
        a = typed(3);
        b = typed("");
        try, typed([]); eid1 = 'BAD'; catch e, eid1 = e.identifier; end
        try, typed(''); eid2 = 'BAD'; catch e, eid2 = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "eid1",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "eid2",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_enforces_arguments_must_be_scalar_or_empty_validator() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x mustBeScalarOrEmpty
            end
            y = 1;
        end
        a = typed(3);
        b = typed([]);
        try, typed([1 2]); eid = 'BAD'; catch e, eid = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "eid",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_enforces_arguments_must_be_real_validator() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x mustBeReal
            end
            y = 1;
        end
        a = typed(3);
        b = typed(complex(1,0));
        try, typed(complex(1,2)); eid = 'BAD'; catch e, eid = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "eid",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_enforces_arguments_must_be_integer_validator() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x mustBeInteger
            end
            y = 1;
        end
        a = typed(3);
        b = typed(3.0);
        try, typed(3.5); eid = 'BAD'; catch e, eid = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "eid",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_enforces_arguments_must_be_positive_validator() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x mustBePositive
            end
            y = 1;
        end
        a = typed(1);
        try, typed(0); eid0 = 'BAD'; catch e, eid0 = e.identifier; end
        try, typed(-2); eidn = 'BAD'; catch e, eidn = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "eid0",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "eidn",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_enforces_arguments_must_be_nonnegative_validator() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x mustBeNonnegative
            end
            y = 1;
        end
        a = typed(0);
        b = typed(2);
        try, typed(-1); eidn = 'BAD'; catch e, eidn = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "eidn",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_enforces_arguments_must_be_nonzero_validator() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x mustBeNonzero
            end
            y = 1;
        end
        a = typed(2);
        try, typed(0); eid0 = 'BAD'; catch e, eid0 = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "eid0",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_enforces_arguments_must_be_nonpositive_validator() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x mustBeNonpositive
            end
            y = 1;
        end
        a = typed(0);
        b = typed(-2);
        try, typed(1); eidp = 'BAD'; catch e, eidp = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "eidp",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_enforces_arguments_must_be_negative_validator() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x mustBeNegative
            end
            y = 1;
        end
        a = typed(-1);
        try, typed(0); eid0 = 'BAD'; catch e, eid0 = e.identifier; end
        try, typed(2); eidp = 'BAD'; catch e, eidp = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "eid0",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "eidp",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_enforces_arguments_must_be_greater_than_or_equal_validator() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x mustBeGreaterThanOrEqual(x, 0)
            end
            y = 1;
        end
        a = typed(0);
        b = typed(2);
        try, typed(-1); eidn = 'BAD'; catch e, eidn = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "eidn",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_enforces_arguments_must_be_less_than_or_equal_validator() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x mustBeLessThanOrEqual(x, 3)
            end
            y = x;
        end
        a = typed(3);
        try, typed(4); eid = 'BAD'; catch e, eid = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(3.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "eid",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_arguments_validator_threshold_supports_unary_minus_literal() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x mustBeGreaterThanOrEqual(x, -2)
            end
            y = x;
        end
        a = typed(-2);
        try, typed(-3); eid = 'BAD'; catch e, eid = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(-2.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "eid",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_enforces_arguments_must_be_greater_than_and_less_than_validators() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = gtcheck(x)
            arguments
                x mustBeGreaterThan(x, 1)
            end
            y = x;
        end
        function y = ltcheck(x)
            arguments
                x mustBeLessThan(x, 5)
            end
            y = x;
        end
        a = gtcheck(2);
        b = ltcheck(4);
        try, gtcheck(1); e1 = 'BAD'; catch e, e1 = e.identifier; end
        try, ltcheck(5); e2 = 'BAD'; catch e, e2 = e.identifier; end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(2.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(4.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "e1",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "e2",
        &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
    ));
}

#[test]
fn execute_text_request_rejects_arguments_block_unknown_validator() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x (1,1) double mustBeNope
            end
            y = x * 2;
        end
        r = typed(3);
    "#;
    let err = execute_text_request(&mut session, source).expect_err("expected semantic failure");
    let RunError::Semantic(err) = err else {
        panic!("expected semantic error for unknown arguments validator");
    };
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:FunctionArgumentValidationUnknownValidator")
    );
}

#[test]
fn execute_text_request_rejects_arguments_block_unsupported_trailing_syntax() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x (1,1) double < 10
            end
            y = x;
        end
        r = typed(3);
    "#;
    let err = execute_text_request(&mut session, source).expect_err("expected semantic failure");
    let RunError::Semantic(err) = err else {
        panic!("expected semantic error for unsupported arguments syntax");
    };
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:FunctionArgumentValidationUnsupported")
    );
}

#[test]
fn execute_text_request_rejects_advanced_arguments_block_kinds() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let repeating_source = r#"
        function y = typed(x, varargin)
            arguments (Repeating)
                varargin double
            end
            y = x;
        end
        r = typed(3, 4);
    "#;
    let err = execute_text_request(&mut session, repeating_source)
        .expect_err("expected semantic failure for repeating arguments block");
    let RunError::Semantic(err) = err else {
        panic!("expected semantic error for repeating arguments block");
    };
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:FunctionArgumentValidationUnsupported")
    );

    let output_source = r#"
        function y = typed(x)
            arguments (Output)
                y double
            end
            y = x;
        end
        r = typed(3);
    "#;
    let err = execute_text_request(&mut session, output_source)
        .expect_err("expected semantic failure for output arguments block");
    let RunError::Semantic(err) = err else {
        panic!("expected semantic error for output arguments block");
    };
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:FunctionArgumentValidationUnsupported")
    );
}

#[test]
fn execute_text_request_rejects_arguments_block_name_value_declaration() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(opts)
            arguments
                opts.Name (1,1) double = 1
            end
            y = opts.Name;
        end
        r = typed(struct('Name', 3));
    "#;
    let err = execute_text_request(&mut session, source)
        .expect_err("expected semantic failure for unsupported name-value declaration");
    let RunError::Semantic(err) = err else {
        panic!("expected semantic error for unsupported name-value declaration");
    };
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:FunctionArgumentValidationUnsupported")
    );
}

#[test]
fn execute_text_request_rejects_arguments_block_duplicate_declarations() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x (1,1) double
                x (1,1) double
            end
            y = x * 2;
        end
        r = typed(3);
    "#;
    let err = execute_text_request(&mut session, source).expect_err("expected semantic failure");
    let RunError::Semantic(err) = err else {
        panic!("expected semantic error for duplicate arguments declarations");
    };
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:FunctionArgumentValidationDuplicate")
    );
}

#[test]
fn execute_text_request_supports_arguments_default_for_omitted_input() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x (1,1) double = 3
            end
            y = x * 2;
        end
        a = typed();
        b = typed(4);
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(6.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(8.0)
    ));
}

#[test]
fn execute_text_request_supports_arguments_signed_numeric_default_for_omitted_input() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x (1,1) double = -3
            end
            y = x * 2;
        end
        a = typed();
        b = typed(4);
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(-6.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(8.0)
    ));
}

#[test]
fn execute_text_request_rejects_arguments_default_non_literal_expression() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x (1,1) double = sqrt(9)
            end
            y = x * 2;
        end
        r = typed();
    "#;
    let err = execute_text_request(&mut session, source).expect_err("expected semantic failure");
    let RunError::Semantic(err) = err else {
        panic!("expected semantic error for non-literal arguments default");
    };
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:FunctionArgumentDefaultUnsupported")
    );
}

#[test]
fn execute_text_request_supports_arguments_empty_array_default_for_omitted_input() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function y = typed(x)
            arguments
                x = []
            end
            y = isempty(x);
        end
        a = typed();
        b = typed(4);
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Bool(true)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Bool(false)
    ));
}

#[test]
fn execute_text_request_supports_multi_assign_index_cell_targets() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        function varargout = pair(x)
            varargout{1} = x + 1;
            varargout{2} = x + 2;
        end
        [a, b] = pair(5);
        c = {0, 0};
        [c{1}, c{2}] = pair(10);
        d = c{1};
        e = c{2};
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(6.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(7.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "d",
        &runmat_builtins::Value::Num(11.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "e",
        &runmat_builtins::Value::Num(12.0)
    ));
}

#[test]
fn execute_text_request_supports_cell_brace_range_assignment_from_multi_output_call() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        c = {0, 0};
        c{1:2} = pair(10);
        a = c{1};
        b = c{2};
        function varargout = pair(x)
            varargout{1} = x + 1;
            varargout{2} = x + 2;
        end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(11.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(12.0)
    ));
}

#[test]
fn execute_text_request_supports_nested_varargout_forwarding_with_nargout_slice() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        [a,b] = outer(5);
        function varargout = outer(x)
            [varargout{1:nargout}] = inner(x);
            function varargout = inner(v)
                varargout{1} = v + 1;
                varargout{2} = v + 2;
            end
        end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(6.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(7.0)
    ));
}

#[test]
fn execute_text_request_supports_nested_varargout_forwarding_with_nargout_slice_via_feval() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        [a,b] = outer(5);
        function varargout = outer(x)
            [varargout{1:nargout}] = feval('pair', x);
        end
        function varargout = pair(v)
            varargout{1} = v + 10;
            varargout{2} = v + 20;
        end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(15.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(25.0)
    ));
}

#[test]
fn execute_text_request_supports_nested_varargout_forwarding_with_nargout_slice_via_nested_feval() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        [a,b] = outer(5);
        function varargout = outer(x)
            [varargout{1:nargout}] = feval('inner', x);
            function varargout = inner(v)
                varargout{1} = v + 10;
                varargout{2} = v + 20;
            end
        end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(15.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(25.0)
    ));
}

#[test]
fn execute_text_request_supports_direct_recursive_function() {
    run_deep_semantic_test(|| {
        let mut session =
            RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
        let source = r#"
            y = fact(5);
            function out = fact(n)
                if n <= 1
                    out = 1;
                else
                    out = n * fact(n - 1);
                end
            end
        "#;
        let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
        assert!(outcome_has_named_upsert(
            &outcome,
            "y",
            &runmat_builtins::Value::Num(120.0)
        ));
    });
}

#[test]
fn execute_text_request_supports_dynamic_recursive_function_routes() {
    run_deep_semantic_test(|| {
        let mut session =
            RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
        let source = r#"
            h = @fact_handle;
            a = h(5);
            b = feval('fact_feval', 5);
            function out = fact_handle(n)
                if n <= 1
                    out = 1;
                else
                    out = n * fact_handle(n - 1);
                end
            end
            function out = fact_feval(n)
                if n <= 1
                    out = 1;
                else
                    out = n * feval('fact_feval', n - 1);
                end
            end
        "#;
        let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
        assert!(outcome_has_named_upsert(
            &outcome,
            "a",
            &runmat_builtins::Value::Num(120.0)
        ));
        assert!(outcome_has_named_upsert(
            &outcome,
            "b",
            &runmat_builtins::Value::Num(120.0)
        ));
    });
}

#[test]
fn execute_text_request_supports_arity_check_helpers() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        ok = guarded(1, 2, 3);
        try
            too_many_input(1, 2);
            id_in = "NOERR";
        catch e
            id_in = e.identifier;
        end
        try
            one = needs_two_outputs();
            id_out_low = "NOERR";
        catch e
            id_out_low = e.identifier;
        end
        try
            [a, b] = one_output_only();
            id_out_high = "NOERR";
        catch e
            id_out_high = e.identifier;
        end

        function y = guarded(a, b, varargin)
            narginchk(2, Inf);
            nargoutchk(1, 1);
            y = a + b + length(varargin);
        end

        function y = too_many_input(a, varargin)
            narginchk(1, 1);
            y = a;
        end

        function [a, b] = needs_two_outputs()
            nargoutchk(2, 2);
            a = 1;
            b = 2;
        end

        function varargout = one_output_only()
            nargoutchk(0, 1);
            varargout{1} = 1;
        end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "ok",
        &runmat_builtins::Value::Num(4.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "id_in",
        &runmat_builtins::Value::String("RunMat:TooManyInputs".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "id_out_low",
        &runmat_builtins::Value::String("RunMat:NotEnoughOutputs".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "id_out_high",
        &runmat_builtins::Value::String("RunMat:TooManyOutputs".into())
    ));
}

#[test]
fn execute_text_request_supports_dynamic_arity_check_helpers() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        ok = guarded_inputs(1, 2);
        one = guarded_outputs(1);
        try
            guarded_inputs(1, 2, 3);
            in_eid = "BAD";
        catch e
            in_eid = e.identifier;
        end
        try
            [a, b] = guarded_outputs(1);
            out_eid = "BAD";
        catch e
            out_eid = e.identifier;
        end

        function y = guarded_inputs(a, varargin)
            checker = @narginchk;
            feval(checker, 1, 2);
            y = a;
        end

        function [a, b] = guarded_outputs(x)
            checker = @nargoutchk;
            feval(checker, 1, 1);
            a = x;
            b = x;
        end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "ok",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "one",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "in_eid",
        &runmat_builtins::Value::String("RunMat:TooManyInputs".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "out_eid",
        &runmat_builtins::Value::String("RunMat:TooManyOutputs".into())
    ));
}

#[test]
fn execute_path_request_supports_mfilename_name_and_fullpath() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let source_path = tmp.path().join("named_source.m");
    std::fs::write(
        &source_path,
        r#"
        name = mfilename();
        full = mfilename("fullpath");
        fallback = mfilename("not-a-mode");
    "#,
    )
    .expect("write source");
    let mut expected_full = source_path.clone();
    expected_full.set_extension("");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute path source");
    assert!(outcome_has_named_upsert(
        &outcome,
        "name",
        &runmat_builtins::Value::String("named_source".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "fallback",
        &runmat_builtins::Value::String("named_source".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "full",
        &runmat_builtins::Value::String(expected_full.to_string_lossy().to_string())
    ));
}

#[test]
fn execute_path_request_supports_mfilename_class_option_for_static_method() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("C.m"),
        r#"
classdef C
  methods(Static)
    function cls = who()
      cls = mfilename("class");
    end
  end
end
"#,
    )
    .expect("write class source");
    let source_path = tmp.path().join("main.m");
    std::fs::write(
        &source_path,
        r#"
        cls = C.who();
        outside = mfilename("class");
    "#,
    )
    .expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute path source");
    assert!(outcome_has_named_upsert(
        &outcome,
        "cls",
        &runmat_builtins::Value::String("C".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "outside",
        &runmat_builtins::Value::String(String::new())
    ));
}

#[test]
fn execute_path_request_supports_mfilename_for_private_package_and_class_folder_helpers() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("private")).expect("create private dir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::create_dir_all(tmp.path().join("@C")).expect("create class-folder dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("private/private_where.m"),
        r#"
        function [name, full] = private_where()
            name = mfilename();
            full = mfilename("fullpath");
        end
        "#,
    )
    .expect("write private helper");
    std::fs::write(
        tmp.path().join("+pkg/whereami.m"),
        r#"
        function [name, full] = whereami()
            name = mfilename();
            full = mfilename("fullpath");
        end
        "#,
    )
    .expect("write package helper");
    std::fs::write(
        tmp.path().join("@C/whereami.m"),
        r#"
        function [name, full] = whereami()
            name = mfilename();
            full = mfilename("fullpath");
        end
        "#,
    )
    .expect("write class-folder helper");
    let main_path = tmp.path().join("main.m");
    std::fs::write(
        &main_path,
        r#"
        [private_name, private_full] = private_where();
        [package_name, package_full] = pkg.whereami();
        [class_name, class_full] = C.whereami();
        "#,
    )
    .expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let source_root = std::env::current_dir().expect("read temp cwd");
    let outcome = execute_path_request(&mut session, "main.m").expect("exec");
    let assert_named = |name: &str, expected: runmat_builtins::Value| {
        assert!(
            outcome_has_named_upsert(&outcome, name, &expected),
            "expected {name}={expected:?}; upserts={:?}; diagnostics={:?}",
            outcome.workspace_delta.upserts,
            outcome.diagnostics
        );
    };

    assert_named(
        "private_name",
        runmat_builtins::Value::String("private_where".into()),
    );
    assert_named(
        "private_full",
        runmat_builtins::Value::String(
            source_root
                .join("./private/private_where")
                .to_string_lossy()
                .to_string(),
        ),
    );
    assert_named(
        "package_name",
        runmat_builtins::Value::String("whereami".into()),
    );
    assert_named(
        "package_full",
        runmat_builtins::Value::String(
            source_root
                .join("./+pkg/whereami")
                .to_string_lossy()
                .to_string(),
        ),
    );
    assert_named(
        "class_name",
        runmat_builtins::Value::String("whereami".into()),
    );
    assert_named(
        "class_full",
        runmat_builtins::Value::String(
            source_root
                .join("./@C/whereami")
                .to_string_lossy()
                .to_string(),
        ),
    );
}

#[test]
fn execute_text_request_supports_functions_metadata_builtin() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        from_text = functions(str2func("sin"));
        text_name = getfield(from_text, "function");
        text_type = from_text.type;
        local_handle = @local_id;
        local_info = functions(local_handle);
        local_name = getfield(local_info, "function");
        anon_info = functions(@(x) x + 1);
        anon_type = anon_info.type;

        function y = local_id(x)
            y = x;
        end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "text_name",
        &runmat_builtins::Value::String("sin".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "text_type",
        &runmat_builtins::Value::String("simple".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "local_name",
        &runmat_builtins::Value::String("local_id".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "anon_type",
        &runmat_builtins::Value::String("anonymous".into())
    ));
}

#[test]
fn execute_text_request_supports_inputname_builtin() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        alpha = 10;
        beta = 20;
        [n1, n2, n3, n4] = probe(alpha, alpha + 1, 7, beta);
        [f1, f2] = feval(@probe2, beta, beta + 1);

        function [a, b, c, d] = probe(x, y, z, w)
            a = inputname(1);
            b = inputname(2);
            c = inputname(3);
            d = inputname(4);
        end

        function [a, b] = probe2(x, y)
            a = inputname(1);
            b = inputname(2);
        end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "n1",
        &runmat_builtins::Value::String("alpha".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "n2",
        &runmat_builtins::Value::String(String::new())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "n3",
        &runmat_builtins::Value::String(String::new())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "n4",
        &runmat_builtins::Value::String("beta".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "f1",
        &runmat_builtins::Value::String("beta".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "f2",
        &runmat_builtins::Value::String(String::new())
    ));
}

#[test]
fn execute_path_request_supports_inputname_for_sibling_package_and_class_methods() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("probe.m"),
        r#"
        function [a, b, c, d] = probe(x, y, z, w)
            a = inputname(1);
            b = inputname(2);
            c = inputname(3);
            d = inputname(4);
        end
        "#,
    )
    .expect("write sibling probe");
    std::fs::write(
        tmp.path().join("+pkg/probe.m"),
        r#"
        function [a, b] = probe(x, y)
            a = inputname(1);
            b = inputname(2);
        end
        "#,
    )
    .expect("write package probe");
    std::fs::write(
        tmp.path().join("C.m"),
        r#"
classdef C
  methods(Static)
    function [a, b, c] = probe(cls, x, y)
      a = inputname(1);
      b = inputname(2);
      c = inputname(3);
    end
  end
end
"#,
    )
    .expect("write class source");
    let main_path = tmp.path().join("main.m");
    std::fs::write(
        &main_path,
        r#"
        alpha = 10;
        beta = 20;
        [s1, s2, s3, s4] = probe(alpha, alpha + 1, 7, beta);
        [p1, p2] = pkg.probe(beta, beta + 1);
        [c1, c2, c3] = C.probe(alpha, beta + 1);
        "#,
    )
    .expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome =
        execute_path_request(&mut session, main_path.to_string_lossy().as_ref()).expect("exec");
    let assert_named = |name: &str, expected: runmat_builtins::Value| {
        assert!(
            outcome_has_named_upsert(&outcome, name, &expected),
            "expected {name}={expected:?}; upserts={:?}; diagnostics={:?}",
            outcome.workspace_delta.upserts,
            outcome.diagnostics
        );
    };
    assert_named("s1", runmat_builtins::Value::String("alpha".into()));
    assert_named("s2", runmat_builtins::Value::String(String::new()));
    assert_named("s3", runmat_builtins::Value::String(String::new()));
    assert_named("s4", runmat_builtins::Value::String("beta".into()));
    assert_named("p1", runmat_builtins::Value::String("beta".into()));
    assert_named("p2", runmat_builtins::Value::String(String::new()));
    assert_named("c1", runmat_builtins::Value::String("C".into()));
    assert_named("c2", runmat_builtins::Value::String("alpha".into()));
    assert_named("c3", runmat_builtins::Value::String(String::new()));
}

#[test]
fn execute_text_request_inputname_handles_nested_and_expanded_arguments() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        alpha = 10;
        beta = 20;
        nested_name = outer(alpha);
        cells = {beta};
        expanded_name = probe(cells{:});
        expanded_feval_name = feval(@probe, cells{:});

        function name = outer(x)
            name = inner(x);
            function out = inner(y)
                out = inputname(1);
            end
        end

        function out = probe(x)
            out = inputname(1);
        end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "nested_name",
        &runmat_builtins::Value::String("x".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "expanded_name",
        &runmat_builtins::Value::String(String::new())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "expanded_feval_name",
        &runmat_builtins::Value::String(String::new())
    ));
}

#[test]
fn execute_text_request_supports_localfunctions_builtin() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        handles = localfunctions();
        h1 = handles{1};
        h2 = handles{2};
        total = feval(h1, 5) + feval(h2, 5);

        function y = helper_one(x)
            y = x + 1;
        end

        function y = helper_two(x)
            y = x + 2;
        end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "total",
        &runmat_builtins::Value::Num(13.0)
    ));
}

#[test]
fn execute_request_supports_command_syntax_rewrites_through_semantic_pipeline() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = block_on(session.execute_request(abi::ExecutionRequest {
        source: abi::SourceInput::Text {
            name: "command-syntax-semantic.m".to_string(),
            text: "hold on; h = hold(); axis off;".to_string(),
        },
        compatibility: CompatMode::Matlab,
        host_policy: abi::HostExecutionPolicy::default(),
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace: abi::WorkspaceHandle(uuid::Uuid::from_u128(17)),
    }))
    .result
    .expect("command syntax should execute");
    let h_is_logical_scalar = outcome.workspace_delta.upserts.iter().any(|upsert| {
        let is_h = match &upsert.key {
            abi::WorkspaceBindingKey::Interactive { name, .. } => name.0 == "h",
            abi::WorkspaceBindingKey::SourceBinding { binding, .. } => binding.0 == "h",
            abi::WorkspaceBindingKey::Global { .. }
            | abi::WorkspaceBindingKey::Persistent { .. } => false,
        };
        if !is_h {
            return false;
        }
        match &upsert.value {
            runmat_builtins::Value::Bool(_) => true,
            runmat_builtins::Value::LogicalArray(array) => array.shape == vec![1, 1],
            _ => false,
        }
    });
    assert!(
        h_is_logical_scalar,
        "hold() result should be captured as a logical scalar binding"
    );
}

#[test]
fn execute_request_rejects_command_syntax_in_strict_mode() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let err = block_on(session.execute_request(abi::ExecutionRequest {
        source: abi::SourceInput::Text {
            name: "command-syntax-strict.m".to_string(),
            text: "hold on".to_string(),
        },
        compatibility: CompatMode::Strict,
        host_policy: abi::HostExecutionPolicy::default(),
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace: abi::WorkspaceHandle(uuid::Uuid::from_u128(18)),
    }))
    .result
    .expect_err("strict compatibility should reject command syntax");
    let RunError::Syntax(syntax) = err else {
        panic!("expected syntax error for strict command syntax rejection");
    };
    assert!(
        syntax
            .message
            .contains("Command syntax is disabled in strict compatibility mode"),
        "unexpected strict-mode command syntax error: {}",
        syntax.message
    );
}

#[test]
fn execute_request_supports_warning_off_all_command_rewrite() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = block_on(session.execute_request(abi::ExecutionRequest {
        source: abi::SourceInput::Text {
            name: "command-warning-off-all.m".to_string(),
            text: "warning off all; warning('hello from test'); ok = 1;".to_string(),
        },
        compatibility: CompatMode::Matlab,
        host_policy: abi::HostExecutionPolicy::default(),
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace: abi::WorkspaceHandle(uuid::Uuid::from_u128(19)),
    }))
    .result
    .expect("warning command syntax should execute");
    assert!(outcome_has_named_upsert(
        &outcome,
        "ok",
        &runmat_builtins::Value::Num(1.0)
    ));
}

#[test]
fn execute_request_supports_clearvars_name_command_rewrite() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = block_on(
        session.execute_request(abi::ExecutionRequest {
            source: abi::SourceInput::Text {
                name: "command-clearvars-name.m".to_string(),
                text: "x = 1; y = 2; clearvars x; ex = exist('x', 'var'); ey = exist('y', 'var');"
                    .to_string(),
            },
            compatibility: CompatMode::Matlab,
            host_policy: abi::HostExecutionPolicy::default(),
            requested_outputs: runmat_hir::RequestedOutputCount::Zero,
            workspace: abi::WorkspaceHandle(uuid::Uuid::from_u128(20)),
        }),
    )
    .result
    .expect("clearvars command syntax should execute");
    assert!(outcome_has_named_upsert(
        &outcome,
        "ex",
        &runmat_builtins::Value::Num(0.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "ey",
        &runmat_builtins::Value::Num(1.0)
    ));
}

#[test]
fn execute_request_supports_close_all_command_rewrite() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = block_on(session.execute_request(abi::ExecutionRequest {
        source: abi::SourceInput::Text {
            name: "command-close-all.m".to_string(),
            text: "close all; ok = 1;".to_string(),
        },
        compatibility: CompatMode::Matlab,
        host_policy: abi::HostExecutionPolicy::default(),
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace: abi::WorkspaceHandle(uuid::Uuid::from_u128(21)),
    }))
    .result
    .expect("close all command syntax should execute");
    assert!(outcome_has_named_upsert(
        &outcome,
        "ok",
        &runmat_builtins::Value::Num(1.0)
    ));
}

#[test]
fn execute_request_supports_clearvars_except_command_rewrite() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = block_on(session.execute_request(abi::ExecutionRequest {
        source: abi::SourceInput::Text {
            name: "command-clearvars-except.m".to_string(),
            text:
                "x = 1; y = 2; z = 3; clearvars -except y; ex = exist('x', 'var'); ey = exist('y', 'var'); ez = exist('z', 'var');"
                    .to_string(),
        },
        compatibility: CompatMode::Matlab,
        host_policy: abi::HostExecutionPolicy::default(),
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace: abi::WorkspaceHandle(uuid::Uuid::from_u128(22)),
    }))
    .result
    .expect("clearvars -except command syntax should execute");
    assert!(outcome_has_named_upsert(
        &outcome,
        "ex",
        &runmat_builtins::Value::Num(0.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "ey",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "ez",
        &runmat_builtins::Value::Num(0.0)
    ));
}

#[test]
fn execute_request_rejects_clearvars_except_without_names_command_rewrite() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = block_on(session.execute_request(abi::ExecutionRequest {
        source: abi::SourceInput::Text {
            name: "command-clearvars-except-missing.m".to_string(),
            text: "clearvars -except".to_string(),
        },
        compatibility: CompatMode::Matlab,
        host_policy: abi::HostExecutionPolicy::default(),
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace: abi::WorkspaceHandle(uuid::Uuid::from_u128(23)),
    }))
    .result
    .expect("request should complete with runtime diagnostic");
    assert!(
        outcome.diagnostics.iter().any(|diag| diag
            .message
            .contains("clearvars: -except requires at least one variable name"),),
        "missing clearvars -except diagnostic: {:?}",
        outcome.diagnostics
    );
}

#[test]
fn compile_input_lowers_print_command_dash_and_dotted_args_to_semantic_builtin_call() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let prepared = session
        .compile_input("print -dpng out.v1;")
        .expect("print command syntax should compile");

    assert!(
        prepared.bytecode.layout.is_some(),
        "print command syntax should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::LoadString(s) if s == "-dpng")),
        "print command should carry '-dpng' as a semantic string argument"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::LoadString(s) if s == "out.v1")),
        "print command should carry dotted filename token as a semantic string argument"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(
            |instr| matches!(instr, runmat_vm::Instr::CallBuiltinMulti(name, 2, _) if name == "print")
        ),
        "print command should lower to builtin dispatch with two normalized arguments"
    );
}

#[test]
fn compile_input_lowers_format_command_keyword_to_semantic_builtin_call() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let prepared = session
        .compile_input("format long;")
        .expect("format command syntax should compile");

    assert!(
        prepared.bytecode.layout.is_some(),
        "format command syntax should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::LoadString(s) if s == "long")),
        "format command should carry normalized keyword as semantic string argument"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(
            |instr| matches!(instr, runmat_vm::Instr::CallBuiltinMulti(name, 1, _) if name == "format")
        ),
        "format command should lower to builtin dispatch with one normalized argument"
    );
}

#[test]
fn execute_request_supports_format_command_rewrite_through_semantic_pipeline() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = block_on(session.execute_request(abi::ExecutionRequest {
        source: abi::SourceInput::Text {
            name: "command-format-long.m".to_string(),
            text: "format long; x = 1;".to_string(),
        },
        compatibility: CompatMode::Matlab,
        host_policy: abi::HostExecutionPolicy::default(),
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace: abi::WorkspaceHandle(uuid::Uuid::from_u128(24)),
    }))
    .result
    .expect("format command syntax should execute");
    assert!(outcome_has_named_upsert(
        &outcome,
        "x",
        &runmat_builtins::Value::Num(1.0)
    ));
}

#[test]
fn compile_input_lowers_grid_command_keyword_to_semantic_builtin_call() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let prepared = session
        .compile_input("grid on;")
        .expect("grid command syntax should compile");

    assert!(
        prepared.bytecode.layout.is_some(),
        "grid command syntax should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::LoadString(s) if s == "on")),
        "grid command should carry normalized keyword as semantic string argument"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(
            |instr| matches!(instr, runmat_vm::Instr::CallBuiltinMulti(name, 1, _) if name == "grid")
        ),
        "grid command should lower to builtin dispatch with one normalized argument"
    );
}

#[test]
fn compile_input_lowers_box_command_keyword_to_semantic_builtin_call() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let prepared = session
        .compile_input("box off;")
        .expect("box command syntax should compile");

    assert!(
        prepared.bytecode.layout.is_some(),
        "box command syntax should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::LoadString(s) if s == "off")),
        "box command should carry normalized keyword as semantic string argument"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(
            |instr| matches!(instr, runmat_vm::Instr::CallBuiltinMulti(name, 1, _) if name == "box")
        ),
        "box command should lower to builtin dispatch with one normalized argument"
    );
}

#[test]
fn compile_input_lowers_axis_command_keyword_to_semantic_builtin_call() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let prepared = session
        .compile_input("axis tight;")
        .expect("axis command syntax should compile");

    assert!(
        prepared.bytecode.layout.is_some(),
        "axis command syntax should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::LoadString(s) if s == "tight")),
        "axis command should carry normalized keyword as semantic string argument"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(
            |instr| matches!(instr, runmat_vm::Instr::CallBuiltinMulti(name, 1, _) if name == "axis")
        ),
        "axis command should lower to builtin dispatch with one normalized argument"
    );
}

#[test]
fn compile_input_lowers_colormap_command_keyword_to_semantic_builtin_call() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let prepared = session
        .compile_input("colormap jet;")
        .expect("colormap command syntax should compile");

    assert!(
        prepared.bytecode.layout.is_some(),
        "colormap command syntax should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::LoadString(s) if s == "jet")),
        "colormap command should carry normalized keyword as semantic string argument"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(
            |instr| matches!(instr, runmat_vm::Instr::CallBuiltinMulti(name, 1, _) if name == "colormap")
        ),
        "colormap command should lower to builtin dispatch with one normalized argument"
    );
}

#[test]
fn compile_input_lowers_shading_command_keyword_to_semantic_builtin_call() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let prepared = session
        .compile_input("shading interp;")
        .expect("shading command syntax should compile");

    assert!(
        prepared.bytecode.layout.is_some(),
        "shading command syntax should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::LoadString(s) if s == "interp")),
        "shading command should carry normalized keyword as semantic string argument"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(
            |instr| matches!(instr, runmat_vm::Instr::CallBuiltinMulti(name, 1, _) if name == "shading")
        ),
        "shading command should lower to builtin dispatch with one normalized argument"
    );
}

#[test]
fn compile_input_lowers_colorbar_command_without_args_to_semantic_builtin_call() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let prepared = session
        .compile_input("colorbar;")
        .expect("colorbar command syntax should compile");

    assert!(
        prepared.bytecode.layout.is_some(),
        "colorbar command syntax should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(
            |instr| matches!(instr, runmat_vm::Instr::CallBuiltinMulti(name, 0, _) if name == "colorbar")
        ),
        "colorbar command without args should lower to builtin dispatch with zero arguments"
    );
}

#[test]
fn compile_input_lowers_colorbar_command_keyword_to_semantic_builtin_call() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let prepared = session
        .compile_input("colorbar off;")
        .expect("colorbar command syntax should compile");

    assert!(
        prepared.bytecode.layout.is_some(),
        "colorbar command syntax should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::LoadString(s) if s == "off")),
        "colorbar command should carry normalized keyword as semantic string argument"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(
            |instr| matches!(instr, runmat_vm::Instr::CallBuiltinMulti(name, 1, _) if name == "colorbar")
        ),
        "colorbar command with keyword should lower to builtin dispatch with one argument"
    );
}

#[test]
fn compile_input_rewrites_ident_paren_call_to_index_when_binding_shadows_callable() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "foo = [10, 20, 30]; y = foo(2);";
    let prepared = session
        .compile_input(source)
        .expect("binding-shadowed paren call should compile");

    assert!(
        prepared.bytecode.layout.is_some(),
        "binding-shadowed paren call should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::Index(1))),
        "binding-shadowed paren call should lower to index bytecode"
    );
    assert!(
        !prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallBuiltinMulti(name, _, _) if name == "foo"
        )),
        "binding-shadowed paren call must not lower as callable dispatch"
    );

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "20"
    }));
}

#[test]
fn execute_outcome_exposes_workspace_removals_and_effects() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(&mut session, "x = 1; y = 2;").expect("seed workspace");
    let outcome = execute_text_request(&mut session, "clear x;").expect("clear succeeds");

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
        compatibility: CompatMode::Matlab,
        host_policy: abi::HostExecutionPolicy::default(),
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace,
    }))
    .result
    .expect("exec succeeds");

    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(
            &upsert.key,
            abi::WorkspaceBindingKey::SourceBinding { binding, .. }
                if binding.0 == "requested"
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
        compatibility: CompatMode::Matlab,
        host_policy: abi::HostExecutionPolicy::default(),
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace: abi::WorkspaceHandle(uuid::Uuid::from_u128(9)),
    }))
    .result
    .expect("exec succeeds");

    assert!(outcome.flow.is_no_value());
    assert_eq!(outcome.display_events.len(), 1);
}

#[test]
fn execute_request_path_error_returns_resolved_source_context() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let dir = tempfile::tempdir().expect("tempdir");
    let source_path = dir.path().join("source-context-error.m");
    let source_text = "ss\nx = 1;\n";
    std::fs::write(&source_path, source_text).expect("write test source");
    let source_path = source_path.to_string_lossy().to_string();

    let response = block_on(session.execute_request(abi::ExecutionRequest {
        source: abi::SourceInput::Path(source_path.clone()),
        compatibility: CompatMode::Matlab,
        host_policy: abi::HostExecutionPolicy::default(),
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace: abi::WorkspaceHandle(uuid::Uuid::from_u128(25)),
    }));

    assert_eq!(response.source_context.source_name(), source_path);
    assert_eq!(response.source_context.source_text(), Some(source_text));
    assert!(matches!(
        response.source_context.identity,
        Some(abi::SourceIdentity::PathAndContentHash { .. })
    ));
    let err = response.result.expect_err("undefined name should fail");
    let RunError::Semantic(err) = err else {
        panic!("expected semantic error for undefined path source name");
    };
    assert_eq!(err.identifier.as_deref(), Some("RunMat:UndefinedVariable"));
    assert!(err.span.is_some(), "semantic error should carry a span");
}

#[test]
fn execute_request_runtime_diagnostic_preserves_span_and_callstack() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "x = [1];\ny = x(2);\n";
    let outcome = block_on(session.execute_request(abi::ExecutionRequest {
        source: abi::SourceInput::Text {
            name: "runtime-span.m".to_string(),
            text: source.to_string(),
        },
        compatibility: CompatMode::Matlab,
        host_policy: abi::HostExecutionPolicy::default(),
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace: abi::WorkspaceHandle(uuid::Uuid::from_u128(26)),
    }))
    .result
    .expect("runtime failures are returned as outcome diagnostics");

    let diagnostic = outcome
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.severity == abi::DiagnosticSeverity::Error)
        .expect("runtime error diagnostic");
    assert!(
        diagnostic.span.is_some(),
        "runtime diagnostic should preserve VM source span"
    );
    assert!(
        !diagnostic.callstack.is_empty(),
        "runtime diagnostic should preserve VM callstack"
    );
}

#[test]
fn execute_request_honors_top_level_await_host_policy() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let err = block_on(session.execute_request(abi::ExecutionRequest {
        source: abi::SourceInput::Text {
            name: "request-await-policy.m".to_string(),
            text: "y = await(1);".to_string(),
        },
        compatibility: CompatMode::Matlab,
        host_policy: abi::HostExecutionPolicy {
            top_level_await: false,
            dynamic_eval: true,
        },
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace: abi::WorkspaceHandle(uuid::Uuid::from_u128(11)),
    }))
    .result
    .expect_err("request should reject top-level await when host policy disables it");
    let RunError::Semantic(err) = err else {
        panic!("expected semantic top-level-await policy error");
    };
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:AwaitContextInvalid")
    );
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
fn compile_input_records_mir_analysis_facts() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let prepared = session
        .compile_input("x = [1 2; 3 4]; y = x(:, 2);")
        .expect("compile");
    assert!(
        prepared.analysis().diagnostics.is_empty(),
        "valid semantic compile should not emit MIR diagnostics"
    );
    assert!(
        !prepared.analysis().mir_locals.is_empty(),
        "semantic compile should carry MIR local analysis facts for entrypoint execution"
    );
}

#[test]
fn compile_input_resolves_wildcard_import_from_project_source_index() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+stats")).expect("create package dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("+stats/summarize.m"),
        "function y = summarize(x); y = x; end",
    )
    .expect("write dependency symbol");
    std::fs::write(tmp.path().join("main.m"), "import pkg.*; foo()").expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_name = tmp.path().join("main.m").to_string_lossy().to_string();
    let prepared = session
        .compile_input_for_source_name(&source_name, "import stats.*; y = summarize(1);")
        .expect("compile");

    let calls = &prepared.lowering().hir_index.calls;
    assert!(
        calls.iter().any(|call| {
            matches!(
                call.kind,
                runmat_hir::CallKind::PackageFunction(_) | runmat_hir::CallKind::DirectFunction(_)
            )
        }),
        "wildcard import call should resolve to package function when source index symbols are available; calls={calls:#?}"
    );
}

#[test]
fn execute_path_request_loads_sibling_classdef_sources_without_manifest() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("Vec2.m"),
        r#"
classdef Vec2
  properties
    x
    y
  end
  methods
    function obj = Vec2(x, y)
      obj.x = x;
      obj.y = y;
    end
    function m = magnitude(obj)
      m = sqrt(obj.x*obj.x + obj.y*obj.y);
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "p = Vec2(3,4); cls = class(p); m = p.magnitude();",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(
        outcome_has_named_upsert(
            &outcome,
            "cls",
            &runmat_builtins::Value::String("Vec2".into())
        ),
        "expected class() result to be Vec2"
    );
    assert!(
        outcome_has_named_upsert(&outcome, "m", &runmat_builtins::Value::Num(5.0)),
        "expected method call result to be 5"
    );
}

#[test]
fn execute_path_request_loads_package_classdef_sources_without_manifest() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+geom")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+geom").join("Point.m"),
        r#"
classdef Point
  properties
    x
  end
  methods
    function obj = Point(v)
      obj.x = v;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "p = geom.Point(42); c = class(p); x = p.x;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "c",
        &runmat_builtins::Value::String("geom.Point".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "x",
        &runmat_builtins::Value::Num(42.0)
    ));
}

#[test]
fn execute_path_request_loads_dependency_package_classdef_sources_via_project_composition() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let dep_root = tmp.path().join("deps/geombase");
    std::fs::create_dir_all(dep_root.join("+geom")).expect("create dependency package dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]

[dependencies]
geombase = { path = "deps/geombase" }
"#,
    )
    .expect("write root manifest");
    std::fs::write(
        dep_root.join("runmat.toml"),
        r#"
[package]
name = "geombase"

[sources]
roots = ["."]
"#,
    )
    .expect("write dependency manifest");
    std::fs::write(
        dep_root.join("+geom").join("Point.m"),
        r#"
classdef Point
  properties
    x
  end
  methods
    function obj = Point(v)
      obj.x = v;
    end
  end
end
"#,
    )
    .expect("write dependency class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "p = geom.Point(42); c = class(p); x = p.x;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "c",
        &runmat_builtins::Value::String("geom.Point".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "x",
        &runmat_builtins::Value::Num(42.0)
    ));
}

#[test]
fn execute_path_request_source_authoring_oop_smoke() {
    let handle = std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || {
            let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
            let tmp = tempfile::TempDir::new().expect("tempdir");
            std::fs::write(
                tmp.path().join("Vec2.m"),
                r#"
classdef Vec2
  properties
    x
    y
  end
  methods
    function obj = Vec2(x, y)
      obj.x = x;
      obj.y = y;
    end
    function m = magnitude(obj)
      m = sqrt(obj.x*obj.x + obj.y*obj.y);
    end
  end
  methods(Static)
    function u = unitX()
      u = Vec2(1, 0);
    end
  end
end
"#,
            )
            .expect("write Vec2 source");
            std::fs::write(
                tmp.path().join("Money.m"),
                r#"
classdef Money
  properties
    amount
  end
  methods
    function obj = Money(v)
      obj.amount = v;
    end
    function out = plus(a, b)
      out = Money(a.amount + b.amount);
    end
  end
end
"#,
            )
            .expect("write Money source");
            std::fs::write(
                tmp.path().join("main.m"),
                r#"
v = Vec2(3, 4);
cls = class(v);
isVec = isa(v, 'Vec2');
mag = v.magnitude();
ux = Vec2.unitX();
uxx = ux.x;
a = Money(10);
b = Money(5);
c = a + b;
isMoney = isa(c, 'Money');
amt = c.amount;
"#,
            )
            .expect("write main source");

            let mut session =
                RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
            let source_path = tmp.path().join("main.m");
            let outcome =
                execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
                    .expect("execute script");

            assert!(outcome_has_named_upsert(
                &outcome,
                "cls",
                &runmat_builtins::Value::String("Vec2".into())
            ));
            assert!(outcome_has_named_upsert(
                &outcome,
                "isVec",
                &runmat_builtins::Value::Bool(true)
            ));
            assert!(outcome_has_named_upsert(
                &outcome,
                "mag",
                &runmat_builtins::Value::Num(5.0)
            ));
            assert!(outcome_has_named_upsert(
                &outcome,
                "uxx",
                &runmat_builtins::Value::Num(1.0)
            ));
            assert!(outcome_has_named_upsert(
                &outcome,
                "isMoney",
                &runmat_builtins::Value::Bool(true)
            ));
            assert!(outcome_has_named_upsert(
                &outcome,
                "amt",
                &runmat_builtins::Value::Num(15.0)
            ));
        })
        .expect("spawn source authoring oop smoke thread");
    handle
        .join()
        .expect("source authoring oop smoke thread failed");
}

#[test]
fn execute_path_request_source_authoring_one_file_oop_smoke() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let source = tmp.path().join("main.m");
    std::fs::write(
        &source,
        r#"
classdef Vec2
  properties
    x
    y
  end
  methods
    function obj = Vec2(x, y)
      obj.x = x;
      obj.y = y;
    end
    function m = magnitude(obj)
      m = sqrt(obj.x*obj.x + obj.y*obj.y);
    end
  end
  methods(Static)
    function u = unitX()
      u = Vec2(1, 0);
    end
  end
end

p = Vec2(3,4);
cls = class(p);
isv = isa(p,'Vec2');
m = p.magnitude();
u = Vec2.unitX();
ux = u.x;
"#,
    )
    .expect("write single-file source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = execute_path_request(&mut session, source.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "cls",
        &runmat_builtins::Value::String("Vec2".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "isv",
        &runmat_builtins::Value::Bool(true)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "m",
        &runmat_builtins::Value::Num(5.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "ux",
        &runmat_builtins::Value::Num(1.0)
    ));
}

#[test]
fn execute_path_request_source_authoring_one_file_operator_overload_plus() {
    let handle = std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || {
            let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
            let tmp = tempfile::TempDir::new().expect("tempdir");
            let source = tmp.path().join("main.m");
            std::fs::write(
                &source,
                r#"
classdef Money
  properties
    amount
  end
  methods
    function obj = Money(v)
      obj.amount = v;
    end
    function out = plus(a, b)
      out = Money(a.amount + b.amount);
    end
  end
end

a = Money(10);
b = Money(5);
c = a + b;
cls = class(c);
ism = isa(c, 'Money');
v = c.amount;
"#,
            )
            .expect("write single-file source");

            let mut session =
                RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
            let outcome = execute_path_request(&mut session, source.to_string_lossy().as_ref())
                .expect("execute script");

            assert!(outcome_has_named_upsert(
                &outcome,
                "cls",
                &runmat_builtins::Value::String("Money".into())
            ));
            assert!(outcome_has_named_upsert(
                &outcome,
                "ism",
                &runmat_builtins::Value::Bool(true)
            ));
            assert!(outcome_has_named_upsert(
                &outcome,
                "v",
                &runmat_builtins::Value::Num(15.0)
            ));
        })
        .expect("spawn one-file operator overload plus thread");
    handle
        .join()
        .expect("one-file operator overload plus thread failed");
}

#[test]
fn execute_path_request_preserves_handle_alias_semantics_for_sibling_classdef() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("H.m"),
        r#"
classdef H < handle
  properties
    x
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "a = H(); a.x = 1; b = a; a.x = 9; y = b.x;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(
        outcome_has_named_upsert(&outcome, "y", &runmat_builtins::Value::Num(9.0)),
        "expected handle alias update to be visible through b.x"
    );
}

#[test]
fn execute_path_request_delete_invalidates_all_handle_aliases() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("H.m"),
        r#"
classdef H < handle
  properties
    x
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "a = H(); b = a; a.x = 5; delete(a); va = isvalid(a); vb = isvalid(b); try, y = b.x; id = 'BAD'; catch e, id = e.identifier; end;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "va",
        &runmat_builtins::Value::Bool(false)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "vb",
        &runmat_builtins::Value::Bool(false)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "id",
        &runmat_builtins::Value::String("RunMat:getfield:InvalidHandle".into())
    ));
}

#[test]
fn execute_path_request_calls_handle_delete_method_before_invalidation() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("H.m"),
        r#"
classdef H < handle
  properties(Static)
    DeletedCount = 0;
  end
  methods
    function delete(obj)
      H.DeletedCount = H.DeletedCount + 1;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "h = H(); delete(h); g = H.DeletedCount; v = isvalid(h);",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "g",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "v",
        &runmat_builtins::Value::Bool(false)
    ));
}

#[test]
fn execute_path_request_delete_listener_disables_future_notifications() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("H.m"),
        r#"
classdef H < handle
  events
    Tick
  end
  methods
    function fire(obj)
      notify(obj, 'Tick');
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("CB.m"),
        r#"
classdef CB
  properties(Static)
    Count = 0;
  end
  methods(Static)
    function on_tick(src)
      CB.Count = CB.Count + 1;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "h = H(); l = addlistener(h, 'Tick', @CB.on_tick); delete(l); h.fire(); c = CB.Count;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "c",
        &runmat_builtins::Value::Num(0.0)
    ));
}

#[test]
fn execute_path_request_supports_listener_member_access_via_dot_syntax() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("H.m"),
        r#"
classdef H < handle
  events
    Tick
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("CB.m"),
        r#"
classdef CB
  methods(Static)
    function on_tick(src)
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "h = H(); l = addlistener(h, 'Tick', @CB.on_tick); enabled = l.Enabled; valid = l.Valid;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "enabled",
        &runmat_builtins::Value::Bool(true)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "valid",
        &runmat_builtins::Value::Bool(true)
    ));
}

#[test]
fn execute_path_request_supports_addlistener_char_callback_name_without_at_prefix() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("H.m"),
        r#"
classdef H < handle
  events
    Tick
  end
  methods
    function fire(obj)
      notify(obj, 'Tick');
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("CB.m"),
        r#"
classdef CB
  properties(Static)
    Count = 0;
  end
  methods(Static)
    function on_tick(src)
      CB.Count = CB.Count + 1;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "h = H(); addlistener(h, 'Tick', 'CB.on_tick'); h.fire(); g = CB.Count;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "g",
        &runmat_builtins::Value::Num(1.0)
    ));
}

#[test]
fn compile_local_function_global_decl_emits_named_global_workspace_effect() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "global G; G = 0; y = on_tick(1,2);\nfunction out = on_tick(a,b)\n  global G; G = G + 1; out = 77;\nend";
    let prepared = session.compile_input(source).expect("compile");
    let fid = prepared
        .bytecode
        .function_registry
        .resolve_name("on_tick")
        .expect("local function should resolve");
    let function = prepared
        .bytecode
        .function_registry
        .get(fid)
        .expect("local function bytecode should exist");
    assert!(
        function
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::DeclareGlobalNamed(_, _))),
        "local function global declaration should lower to DeclareGlobalNamed"
    );
    let declared_slots: std::collections::HashSet<usize> = function
        .instructions
        .iter()
        .filter_map(|instr| match instr {
            runmat_vm::Instr::DeclareGlobalNamed(indices, names)
                if names.iter().any(|name| name == "G") =>
            {
                Some(indices.to_vec())
            }
            _ => None,
        })
        .flatten()
        .collect();
    let stores_global_slot = function.instructions.iter().any(|instr| match instr {
        runmat_vm::Instr::StoreVar(index) | runmat_vm::Instr::StoreLocal(index) => {
            declared_slots.contains(index)
        }
        _ => false,
    });
    assert!(
        stores_global_slot,
        "global binding slot should be the same slot being stored for assignment"
    );
}

#[test]
fn execute_path_request_reports_undefined_function_for_missing_event_callback() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("H.m"),
        r#"
classdef H < handle
  events
    Tick
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "h = H(); addlistener(h, 'Tick', @definitely_missing_callback); try, notify(h, 'Tick'); id = 'BAD'; catch e, id = e.identifier; end;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "id",
        &runmat_builtins::Value::String("RunMat:UndefinedFunction".into())
    ));
}

#[test]
fn execute_path_request_reports_undefined_function_for_uncaught_missing_event_callback() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("H.m"),
        r#"
classdef H < handle
  events
    Tick
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "h = H(); addlistener(h, 'Tick', @definitely_missing_callback); notify(h, 'Tick');",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execution should complete with diagnostic");

    assert!(
        outcome
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RunMat:UndefinedFunction"),
        "uncaught missing event callback should surface RunMat:UndefinedFunction diagnostic"
    );
}

#[test]
fn execute_path_request_supports_enumeration_member_static_access() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("Color.m"),
        r#"
classdef Color
  enumeration
    Red
    Blue
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(tmp.path().join("main.m"), "c = Color.Red; cls = class(c);")
        .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "cls",
        &runmat_builtins::Value::String("Color".into())
    ));
}

#[test]
fn execute_path_request_supports_call_method_subsasgn_dot_without_custom_subsasgn() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("H.m"),
        r#"
classdef H < handle
  properties
    x
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "a = H(); a.x = 1; b = a; a = call_method(a, 'subsasgn', '.', 'x', 9); y = b.x;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(9.0)
    ));
}

#[test]
fn execute_path_request_supports_call_method_subsref_dot_without_custom_subsref() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("H.m"),
        r#"
classdef H < handle
  properties
    x
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "a = H(); a.x = 11; y = call_method(a, 'subsref', '.', 'x');",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(11.0)
    ));
}

#[test]
fn execute_path_request_rejects_subclassing_sealed_class() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef (Sealed) A
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B < A
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "try, b = B(); id = 'BAD'; catch e, id = e.identifier; msg = e.message; end;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(
        !outcome_has_named_upsert(
            &outcome,
            "id",
            &runmat_builtins::Value::String("BAD".into())
        ),
        "sealed class inheritance should fail"
    );
    assert!(
        outcome_has_named_upsert(
            &outcome,
            "id",
            &runmat_builtins::Value::String("RunMat:ClassSealed".into())
        ) || outcome
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RunMat:ClassSealed"),
        "sealed class failure should surface RunMat:ClassSealed"
    );
}

#[test]
fn execute_path_request_reports_class_sealed_for_uncaught_subclassing() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef (Sealed) A
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B < A
end
"#,
    )
    .expect("write class source");
    std::fs::write(tmp.path().join("main.m"), "b = B();").expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execution should complete with diagnostic");

    assert!(
        outcome
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RunMat:ClassSealed"),
        "uncaught sealed-class subclassing should surface RunMat:ClassSealed diagnostic"
    );
}

#[test]
fn execute_path_request_rejects_instantiating_abstract_class() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef (Abstract) A
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "try, a = A(); id = 'BAD'; catch e, id = e.identifier; msg = e.message; end;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(
        !outcome_has_named_upsert(
            &outcome,
            "id",
            &runmat_builtins::Value::String("BAD".into())
        ),
        "abstract class instantiation should fail"
    );
}

#[test]
fn execute_path_request_reports_abstract_method_missing_for_uncaught_abstract_instantiation() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef (Abstract) A
  methods (Abstract)
    y = f(obj);
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(tmp.path().join("main.m"), "a = A();").expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(
        outcome.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "RunMat:AbstractMethodMissing"
                && diagnostic
                    .message
                    .contains("Cannot instantiate abstract class")
        }),
        "uncaught abstract instantiation should surface RunMat:AbstractMethodMissing diagnostics; got {:?}",
        outcome.diagnostics
    );
}

#[test]
fn execute_path_request_rejects_concrete_subclass_missing_abstract_method() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef (Abstract) A
  methods (Abstract)
    y = f(obj)
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B < A
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "try, b = B(); id = 'BAD'; catch e, id = e.identifier; msg = e.message; end;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(
        !outcome_has_named_upsert(
            &outcome,
            "id",
            &runmat_builtins::Value::String("BAD".into())
        ),
        "concrete subclass missing abstract method should fail"
    );
    assert!(
        outcome_has_named_upsert(
            &outcome,
            "id",
            &runmat_builtins::Value::String("RunMat:AbstractMethodMissing".into())
        ) || outcome
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RunMat:AbstractMethodMissing"),
        "abstract contract failure should surface RunMat:AbstractMethodMissing"
    );
}

#[test]
fn execute_path_request_reports_abstract_method_missing_for_uncaught_partial_implementation() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef (Abstract) A
  methods (Abstract)
    y = f(obj);
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B < A
end
"#,
    )
    .expect("write class source");
    std::fs::write(tmp.path().join("main.m"), "b = B();").expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(
        outcome
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RunMat:AbstractMethodMissing"),
        "uncaught concrete subclass missing abstract method should surface RunMat:AbstractMethodMissing diagnostics"
    );
}

#[test]
fn execute_path_request_rejects_overriding_sealed_method() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef A
  methods(Sealed)
    function y = f(obj)
      y = 1;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B < A
  methods
    function y = f(obj)
      y = 2;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "try, b = B(); y = b.f(); id = 'BAD'; catch e, id = e.identifier; msg = e.message; end;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(
        !outcome_has_named_upsert(
            &outcome,
            "id",
            &runmat_builtins::Value::String("BAD".into())
        ),
        "overriding sealed method should fail"
    );
    assert!(
        outcome_has_named_upsert(
            &outcome,
            "id",
            &runmat_builtins::Value::String("RunMat:MethodSealed".into())
        ) || outcome
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RunMat:MethodSealed"),
        "sealed override failure should surface RunMat:MethodSealed"
    );
}

#[test]
fn execute_path_request_reports_method_sealed_for_uncaught_override() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef A
  methods(Sealed)
    function y = f(obj)
      y = 1;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B < A
  methods
    function y = f(obj)
      y = 2;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(tmp.path().join("main.m"), "b = B(); y = b.f();").expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(
        outcome
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RunMat:MethodSealed"),
        "uncaught sealed override should surface RunMat:MethodSealed diagnostics"
    );
}

#[test]
fn execute_path_request_enforces_private_constructor_access() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef A
  methods(Access=private)
    function obj = A()
    end
  end
  methods(Static)
    function obj = make()
      obj = A();
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "try, a = A(); id = 'BAD'; catch e, id = e.identifier; end; b = A.make(); cb = class(b);",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "id",
        &runmat_builtins::Value::String("RunMat:MethodPrivate".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "cb",
        &runmat_builtins::Value::String("A".into())
    ));
}

#[test]
fn execute_path_request_enforces_protected_constructor_access() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef A
  methods(Access=protected)
    function obj = A(v)
      if nargin < 1, v = 1; end
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B < A
  methods
    function obj = B()
      obj@A(2);
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "try, a = A(); id = 'BAD'; catch e, id = e.identifier; end; b = B(); cb = class(b);",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "id",
        &runmat_builtins::Value::String("RunMat:MethodPrivate".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "cb",
        &runmat_builtins::Value::String("B".into())
    ));
}

#[test]
fn execute_path_request_treats_constant_property_as_static_readonly() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("C.m"),
        r#"
classdef C
  properties(Constant)
    K
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "a = C.K; try, C.K = 9; id = 'BAD'; catch e, id = e.identifier; end; b = C.K;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    let empty = runmat_builtins::Value::Tensor(
        runmat_builtins::Tensor::new(vec![], vec![0, 0]).expect("empty tensor"),
    );
    assert!(outcome_has_named_upsert(&outcome, "a", &empty));
    assert!(outcome_has_named_upsert(
        &outcome,
        "id",
        &runmat_builtins::Value::String("RunMat:PropertyReadOnly".into())
    ));
    assert!(outcome_has_named_upsert(&outcome, "b", &empty));
}

#[test]
fn execute_path_request_supports_dependent_getter_setter_methods() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("D.m"),
        r#"
classdef D
  properties(Dependent)
    p
  end
  properties(Access=private)
    p_backing
  end
  methods
    function obj = D()
      obj.p_backing = 2;
    end
    function val = get.p(obj)
      val = obj.p_backing;
    end
    function obj = set.p(obj, val)
      obj.p_backing = val;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "d = D(); a = d.p; d.p = 9; b = d.p;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(2.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(9.0)
    ));
}

#[test]
fn execute_path_request_applies_property_default_initializers() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("S.m"),
        r#"
classdef S
  properties
    x = 4
  end
  properties(Static)
    y = 6
  end
  properties(Constant)
    k = 8
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "s = S(); a = s.x; b = S.y; c = S.k;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(4.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(6.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "c",
        &runmat_builtins::Value::Num(8.0)
    ));
}

#[test]
fn execute_path_request_applies_constant_expression_property_defaults_and_empty_fallback() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("E.m"),
        r#"
classdef E
  properties
    x = 2 + 3
    z
  end
  properties(Static)
    s
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "e = E(); a = e.x; b = e.z; c = E.s;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(5.0)
    ));
    let empty = runmat_builtins::Value::Tensor(
        runmat_builtins::Tensor::new(vec![], vec![0, 0]).expect("empty tensor"),
    );
    assert!(outcome_has_named_upsert(&outcome, "b", &empty));
    assert!(outcome_has_named_upsert(&outcome, "c", &empty));
}

#[test]
fn execute_path_request_allows_private_property_access_within_class_method() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("C.m"),
        r#"
classdef C
  properties(Access=private)
    p
  end
  methods
    function obj = C()
      obj.p = 7;
    end
    function y = getP(obj)
      y = obj.p;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "c = C(); try, x = c.p; id = 'BAD'; catch e, id = e.identifier; end; y = c.getP();",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "id",
        &runmat_builtins::Value::String("RunMat:PropertyPrivateAccess".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(7.0)
    ));
}

#[test]
fn execute_path_request_supports_superclass_constructor_syntax() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef A < handle
  properties
    x
  end
  methods
    function obj = A(v)
      obj.x = v;
    end
    function obj = setX(obj, v)
      obj.x = v;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B < A
  methods
    function obj = B(v)
      obj@A(v);
    end
  end
end
"#,
    )
    .expect("write subclass source");
    std::fs::write(
        tmp.path().join("main.m"),
        "b = B(1); cls = class(b); isaA = isa(b,'A'); b2 = b; b = b.setX(9); y = b2.x;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "cls",
        &runmat_builtins::Value::String("B".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "isaA",
        &runmat_builtins::Value::Bool(true)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(9.0)
    ));
}

#[test]
fn execute_path_request_supports_qualified_superclass_constructor_syntax() {
    let _cwd_lock = CWD_LOCK.lock().unwrap();
    let project = tempfile::tempdir().expect("tempdir");
    let root = project.path();
    std::fs::create_dir_all(root.join("+pkg1")).expect("create package");

    std::fs::write(
        root.join("+pkg1").join("A.m"),
        r#"
classdef A < handle
  properties
    x
  end
  methods
    function obj = A(v)
      obj.x = v;
    end
  end
end
"#,
    )
    .expect("write A.m");

    std::fs::write(
        root.join("B.m"),
        r#"
classdef B < pkg1.A
  methods
    function obj = B(v)
      obj@pkg1.A(v + 1);
    end
  end
end
"#,
    )
    .expect("write B.m");

    std::fs::write(
        root.join("main.m"),
        r#"
b = B(8);
y = b.x;
"#,
    )
    .expect("write main.m");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd_guard = push_cwd(root);
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");
    assert!(
        outcome_has_named_upsert(&outcome, "y", &runmat_builtins::Value::Num(9.0)),
        "qualified super constructor should initialize inherited state"
    );
}

#[test]
fn compile_input_lowers_super_constructor_to_semantic_super_instruction() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
classdef A < handle
  methods
    function obj = A(v)
      obj.v = v;
    end
  end
end
classdef B < A
  methods
    function obj = B(v)
      obj@A(v);
    end
  end
end
b = B(3);
"#;
    let prepared = session
        .compile_input(source)
        .expect("compile super constructor syntax");
    let function_registry = prepared.bytecode.function_registry();
    assert!(
        function_registry
            .functions
            .values()
            .flat_map(|f| f.instructions.iter())
            .any(|instr| matches!(
                instr,
                runmat_vm::Instr::CallSuperConstructorMulti { .. }
                    | runmat_vm::Instr::CallSuperConstructorExpandMultiOutput { .. }
            )),
        "super constructor syntax should lower to dedicated semantic super-constructor bytecode"
    );
    assert!(
        !function_registry
            .functions
            .values()
            .flat_map(|f| f.instructions.iter())
            .any(|instr| matches!(
                instr,
                runmat_vm::Instr::CallBuiltinMulti(name, _, _)
                    if name == "__runmat_super_ctor__"
            )),
        "super constructor syntax must not lower through builtin-name dispatch"
    );
}

#[test]
fn execute_path_request_supports_superclass_method_syntax() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef A
  methods
    function v = f(obj)
      v = 3;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B < A
  methods
    function v = f(obj)
      v = f@A(obj) + 4;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(tmp.path().join("main.m"), "b = B(); v = b.f();").expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "v",
        &runmat_builtins::Value::Num(7.0)
    ));
}

#[test]
fn compile_input_lowers_super_method_to_semantic_super_instruction() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
classdef A
  methods
    function y = f(obj, x)
      y = x + 1;
    end
  end
end
classdef B < A
  methods
    function y = f(obj, x)
      y = f@A(obj, x) + 2;
    end
  end
end
y = B().f(4);
"#;
    let prepared = session
        .compile_input(source)
        .expect("compile super method syntax");
    let function_registry = prepared.bytecode.function_registry();
    assert!(
        function_registry
            .functions
            .values()
            .flat_map(|f| f.instructions.iter())
            .any(|instr| matches!(
                instr,
                runmat_vm::Instr::CallSuperMethodMulti { .. }
                    | runmat_vm::Instr::CallSuperMethodExpandMultiOutput { .. }
            )),
        "super method syntax should lower to dedicated semantic super-method bytecode"
    );
    assert!(
        !function_registry
            .functions
            .values()
            .flat_map(|f| f.instructions.iter())
            .any(|instr| matches!(
                instr,
                runmat_vm::Instr::CallBuiltinMulti(name, _, _)
                    if name == "__runmat_super_method__"
            )),
        "super method syntax must not lower through builtin-name dispatch"
    );
}

#[test]
fn execute_path_request_supports_qualified_superclass_method_syntax() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg1")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg1").join("A.m"),
        r#"
classdef A
  methods
    function v = f(obj)
      v = 8;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B < pkg1.A
  methods
    function v = f(obj)
      v = f@pkg1.A(obj) + 1;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(tmp.path().join("main.m"), "b = B(); v = f(b);").expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "v",
        &runmat_builtins::Value::Num(9.0)
    ));
}

#[test]
fn execute_path_request_supports_nested_package_qualified_superclass_syntax() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg").join("+sub")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg").join("+sub").join("A.m"),
        r#"
classdef A
  properties
    x
  end
  methods
    function obj = A(v)
      obj.x = v;
    end
    function y = f(obj, n)
      y = obj.x + n;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B < pkg.sub.A
  methods
    function obj = B(v)
      obj@pkg.sub.A(v + 1);
    end
    function y = f(obj, n)
      y = f@pkg.sub.A(obj, n + 2);
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "b = B(4); x = b.x; y = b.f(3); cls = class(b); isaA = isa(b,'pkg.sub.A');",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "x",
        &runmat_builtins::Value::Num(5.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(10.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "cls",
        &runmat_builtins::Value::String("B".to_string())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "isaA",
        &runmat_builtins::Value::Bool(true)
    ));
}

#[test]
fn execute_path_request_supports_function_style_instance_method_dispatch() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("P.m"),
        r#"
classdef P
  properties
    x
  end
  methods
    function obj = P(v)
      obj.x = v;
    end
    function y = twice(obj)
      y = obj.x * 2;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "p = P(6); a = p.twice(); b = twice(p);",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(12.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(12.0)
    ));
}

#[test]
fn execute_path_request_supports_function_style_protected_method_inside_subclass() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef A
  methods(Access=protected)
    function v = secret(obj)
      v = 9;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B < A
  methods
    function v = callsecret(obj)
      v = secret(obj);
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "b = B(); a = callsecret(b); try, c = secret(b); id = 'BAD'; catch e, id = e.identifier; end;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(9.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "id",
        &runmat_builtins::Value::String("RunMat:MethodPrivate".into())
    ));
}

#[test]
fn execute_path_request_supports_function_style_private_method_inside_class_only() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("C.m"),
        r#"
classdef C
  methods
    function v = callpvt(obj)
      v = pvt(obj);
    end
  end
  methods(Access=private)
    function v = pvt(obj)
      v = 1;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "c = C(); a = callpvt(c); try, b = pvt(c); id = 'BAD'; catch e, id = e.identifier; end;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(1.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "id",
        &runmat_builtins::Value::String("RunMat:MethodPrivate".into())
    ));
}

#[test]
fn execute_path_request_reports_method_private_for_uncaught_function_style_private_call() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("C.m"),
        r#"
classdef C
  methods(Access=private)
    function v = pvt(obj)
      v = 1;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(tmp.path().join("main.m"), "c = C(); b = pvt(c);").expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execution should complete with diagnostic");

    assert!(
        outcome
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RunMat:MethodPrivate"),
        "uncaught function-style private method call should surface RunMat:MethodPrivate diagnostic"
    );
}

#[test]
fn execute_path_request_allows_private_method_calls_within_class() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("C.m"),
        r#"
classdef C
  methods(Access=private)
    function y = hidden(obj)
      y = 42;
    end
  end
  methods
    function y = callHidden(obj)
      y = obj.hidden();
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "c = C(); try, x = c.hidden(); id = 'BAD'; catch e, id = e.identifier; end; y = c.callHidden();",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "id",
        &runmat_builtins::Value::String("RunMat:MethodPrivate".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(42.0)
    ));
}

#[test]
fn execute_path_request_supports_static_method_calls() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("C.m"),
        r#"
classdef C
  methods(Static)
    function y = secret()
      y = 7;
    end
    function y = reveal()
      y = C.secret();
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(tmp.path().join("main.m"), "y = C.reveal();").expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(7.0)
    ));
}

#[test]
fn execute_path_request_supports_source_class_operator_overload_plus() {
    let handle = std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || {
            let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
            let tmp = tempfile::TempDir::new().expect("tempdir");
            std::fs::write(
                tmp.path().join("Money.m"),
                r#"
classdef Money
  properties
    amount
  end
  methods
    function obj = Money(v)
      obj.amount = v;
    end
    function out = plus(a, b)
      out = Money(a.amount + b.amount);
    end
  end
end
"#,
            )
            .expect("write class source");
            std::fs::write(
                tmp.path().join("main.m"),
                "a = Money(10); b = Money(5); c = a + b; cls = class(c); isaMoney = isa(c, 'Money'); v = c.amount;",
            )
            .expect("write script source");

            let mut session =
                RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
            let source_path = tmp.path().join("main.m");
            let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
                .expect("execute script");

            assert!(outcome_has_named_upsert(
                &outcome,
                "cls",
                &runmat_builtins::Value::String("Money".into())
            ));
            assert!(outcome_has_named_upsert(
                &outcome,
                "isaMoney",
                &runmat_builtins::Value::Bool(true)
            ));
            assert!(outcome_has_named_upsert(
                &outcome,
                "v",
                &runmat_builtins::Value::Num(15.0)
            ));
        })
        .expect("spawn operator overload plus e2e thread");
    handle
        .join()
        .expect("operator overload plus e2e thread failed");
}

#[test]
fn execute_path_request_rejects_unqualified_static_method_name_in_script_scope() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("S.m"),
        r#"
classdef S
  methods(Static)
    function v = f(x)
      v = x + 2;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "a = S.f(3); try, b = f(3); id='BAD'; catch e, id = e.identifier; end;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(5.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "id",
        &runmat_builtins::Value::String("RunMat:UndefinedFunction".into())
    ));
}

#[test]
fn execute_path_request_reports_undefined_function_for_uncaught_unqualified_static_method_name() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("S.m"),
        r#"
classdef S
  methods(Static)
    function v = f(x)
      v = x + 2;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(tmp.path().join("main.m"), "a = S.f(3); b = f(3);")
        .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execution should complete with diagnostic");

    assert!(
        outcome
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RunMat:UndefinedFunction"),
        "uncaught unqualified static method should surface RunMat:UndefinedFunction diagnostic"
    );
}

#[test]
fn execute_path_request_supports_unqualified_static_method_name_via_wildcard_import() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("S.m"),
        r#"
classdef S
  methods(Static)
    function v = f(x)
      v = x + 2;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(tmp.path().join("main.m"), "import S.*; a = f(3);")
        .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(5.0)
    ));
}

#[test]
fn execute_path_request_supports_unqualified_static_property_via_wildcard_import() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("S.m"),
        r#"
classdef S
  properties(Static)
    K = 9;
  end
  methods(Static)
    function setk(v)
      S.K = v;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "import S.*; a = K; setk(13); b = K;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(9.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(13.0)
    ));
}

#[test]
fn execute_path_request_local_variable_shadows_imported_static_property() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("S.m"),
        r#"
classdef S
  properties(Static)
    K = 9;
  end
  methods(Static)
    function y = readk()
      y = S.K;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "import S.*; K = 42; a = K; b = readk();",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(42.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(9.0)
    ));
}

#[test]
fn execute_path_request_supports_unqualified_static_method_handle_via_wildcard_import() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("S.m"),
        r#"
classdef S
  methods(Static)
    function v = f(x)
      v = x + 2;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(tmp.path().join("main.m"), "import S.*; h = @f; a = h(3);")
        .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(5.0)
    ));
}

#[test]
fn execute_path_request_supports_unqualified_static_method_handle_via_specific_import() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("S.m"),
        r#"
classdef S
  methods(Static)
    function v = f(x)
      v = x + 2;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(tmp.path().join("main.m"), "import S.f; h = @f; a = h(3);")
        .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(5.0)
    ));
}

#[test]
fn execute_path_request_rejects_ambiguous_unqualified_static_method_handle_via_wildcard_imports() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef A
  methods(Static)
    function y = f(x)
      y = x + 1;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B
  methods(Static)
    function y = f(x)
      y = x + 2;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(tmp.path().join("main.m"), "import A.*; import B.*; h = @f;")
        .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let err = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect_err("ambiguous static handle import should fail");
    let RunError::Semantic(err) = err else {
        panic!("expected semantic error for ambiguous static handle import");
    };
    assert_eq!(err.identifier.as_deref(), Some("RunMat:ImportAmbiguous"));
}

#[test]
fn execute_path_request_rejects_ambiguous_unqualified_static_method_via_wildcard_imports() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef A
  methods(Static)
    function y = f(x)
      y = x + 1;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B
  methods(Static)
    function y = f(x)
      y = x + 2;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "import A.*; import B.*; try, z = f(3); id='BAD'; catch e, id = e.identifier; end;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let err = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect_err("ambiguous wildcard static method should fail semantic resolution");
    let RunError::Semantic(err) = err else {
        panic!("expected semantic error, got {err:?}");
    };
    assert_eq!(err.identifier.as_deref(), Some("RunMat:ImportAmbiguous"));
}

#[test]
fn execute_path_request_enforces_private_static_property_access_identifier() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("P.m"),
        r#"
classdef P
  properties(Static, Access=private)
    v
  end
  methods(Static)
    function y = readv()
      y = P.v;
    end
    function setv(x)
      P.v = x;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "try, a = P.v; id='BAD'; catch e, id=e.identifier; end; P.setv(8); b = P.readv();",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "id",
        &runmat_builtins::Value::String("RunMat:PropertyPrivateAccess".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(8.0)
    ));
}

#[test]
fn execute_path_request_allows_private_static_calls_within_class_only() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("C.m"),
        r#"
classdef C
  methods(Static, Access=private)
    function y = secret()
      y = 7;
    end
  end
  methods(Static)
    function y = reveal()
      y = C.secret();
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "try, x = C.secret(); id = 'BAD'; catch e, id = e.identifier; end; y = C.reveal();",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "id",
        &runmat_builtins::Value::String("RunMat:MethodPrivate".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(7.0)
    ));
}

#[test]
fn execute_path_request_enforces_protected_property_and_method_access() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef A
  properties(Access=protected)
    x
  end
  methods
    function obj = A(v)
      obj.x = v;
    end
  end
  methods(Access=protected)
    function y = hidden(obj)
      y = obj.x + 1;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B < A
  methods
    function obj = B(v)
      obj@A(v);
    end
    function y = getX(obj)
      y = obj.x;
    end
    function y = callHidden(obj)
      y = obj.hidden();
    end
  end
end
"#,
    )
    .expect("write subclass source");
    std::fs::write(
        tmp.path().join("main.m"),
        "b = B(5); try, a = b.x; p = 'BAD'; catch e, p = e.identifier; end; try, c = b.hidden(); m = 'BAD'; catch e2, m = e2.identifier; end; y = b.getX(); z = b.callHidden();",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "p",
        &runmat_builtins::Value::String("RunMat:PropertyPrivateAccess".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "m",
        &runmat_builtins::Value::String("RunMat:MethodPrivate".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(5.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "z",
        &runmat_builtins::Value::Num(6.0)
    ));
}

#[test]
fn execute_path_request_supports_multilevel_super_constructor_chain_for_handle_classes() {
    let handle = std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || {
            let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
            let tmp = tempfile::TempDir::new().expect("tempdir");
            std::fs::write(
                tmp.path().join("A.m"),
                r#"
classdef A < handle
  properties
    x
  end
  methods
    function obj = A(v)
      obj.x = v;
    end
  end
end
"#,
            )
            .expect("write class source");
            std::fs::write(
                tmp.path().join("B.m"),
                r#"
classdef B < A
  methods
    function obj = B(v)
      obj@A(v + 1);
    end
  end
end
"#,
            )
            .expect("write class source");
            std::fs::write(
                tmp.path().join("C.m"),
                r#"
classdef C < B
  methods
    function obj = C(v)
      obj@B(v + 2);
    end
  end
end
"#,
            )
            .expect("write class source");
            std::fs::write(
                tmp.path().join("main.m"),
                "c = C(1); cls = class(c); isaA = isa(c,'A'); y = c.x;",
            )
            .expect("write script source");

            let mut session =
                RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
            let source_path = tmp.path().join("main.m");
            let outcome =
                execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
                    .expect("execute script");

            assert!(outcome_has_named_upsert(
                &outcome,
                "cls",
                &runmat_builtins::Value::String("C".into())
            ));
            assert!(outcome_has_named_upsert(
                &outcome,
                "isaA",
                &runmat_builtins::Value::Bool(true)
            ));
            assert!(outcome_has_named_upsert(
                &outcome,
                "y",
                &runmat_builtins::Value::Num(4.0)
            ));
        })
        .expect("spawn multilevel super ctor e2e thread");
    handle
        .join()
        .expect("multilevel super ctor e2e thread failed");
}

#[test]
fn execute_path_request_supports_static_property_read_write_via_class_name() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("S.m"),
        r#"
classdef S
  properties(Static)
    v
  end
  methods(Static)
    function y = setv(x)
      S.v = x;
      y = S.v;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "a = S.v; b = S.setv(9); c = S.v; S.v = 12; d = S.v;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    let empty = runmat_builtins::Value::Tensor(
        runmat_builtins::Tensor::new(vec![], vec![0, 0]).expect("empty tensor"),
    );
    assert!(outcome_has_named_upsert(&outcome, "a", &empty));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(9.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "c",
        &runmat_builtins::Value::Num(9.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "d",
        &runmat_builtins::Value::Num(12.0)
    ));
}

#[test]
fn execute_path_request_supports_source_class_subsref_signature() {
    let _cwd_lock = CWD_LOCK.lock().unwrap();
    let project = tempfile::tempdir().expect("tempdir");
    let root = project.path();

    std::fs::write(
        root.join("Ov.m"),
        r#"
classdef Ov
  properties
    data
  end
  methods
    function obj = Ov(v)
      obj.data = v;
    end
    function out = subsref(obj, S)
      if strcmp(S(1).type, '()')
        idx = S(1).subs{1};
        out = obj.data(idx) * 10;
      else
        out = builtin('subsref', obj, S);
      end
    end
  end
end
"#,
    )
    .expect("write Ov.m");

    std::fs::write(
        root.join("main.m"),
        r#"
o = Ov([2 3 4]);
y = o(2);
"#,
    )
    .expect("write main.m");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd_guard = push_cwd(root);
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");
    assert!(
        outcome_has_named_upsert(&outcome, "y", &runmat_builtins::Value::Num(30.0)),
        "source-authored subsref(obj, S) should drive () overload dispatch"
    );
}

#[test]
fn execute_path_request_supports_source_class_subsasgn_signature() {
    let _cwd_lock = CWD_LOCK.lock().unwrap();
    let project = tempfile::tempdir().expect("tempdir");
    let root = project.path();

    std::fs::write(
        root.join("Ov.m"),
        r#"
classdef Ov
  properties
    data
  end
  methods
    function obj = Ov(v)
      obj.data = v;
    end
    function out = subsref(obj, S)
      if strcmp(S(1).type, '()')
        idx = S(1).subs{1};
        out = obj.data(idx);
      else
        out = builtin('subsref', obj, S);
      end
    end
    function obj = subsasgn(obj, S, rhs)
      if strcmp(S(1).type, '()')
        idx = S(1).subs{1};
        obj.data(idx) = rhs;
      else
        obj = builtin('subsasgn', obj, S, rhs);
      end
    end
  end
end
"#,
    )
    .expect("write Ov.m");

    std::fs::write(
        root.join("main.m"),
        r#"
o = Ov([1 2 3]);
o(2) = 9;
y = o(2);
"#,
    )
    .expect("write main.m");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd_guard = push_cwd(root);
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");
    assert!(
        outcome_has_named_upsert(&outcome, "y", &runmat_builtins::Value::Num(9.0)),
        "source-authored subsasgn(obj, S, rhs) should drive () assignment overload dispatch"
    );
}

#[test]
fn execute_path_request_supports_source_class_subsref_and_member_index_chains() {
    let _cwd_lock = CWD_LOCK.lock().unwrap();
    let project = tempfile::tempdir().expect("tempdir");
    let root = project.path();

    std::fs::write(
        root.join("C.m"),
        r#"
classdef C
  properties
    data
  end
  methods
    function obj = C(v)
      obj.data = v;
    end
    function out = getdata(obj)
      out = obj.data;
    end
    function out = subsref(obj, S)
      if strcmp(S(1).type, '()')
        idx = S(1).subs{1};
        out = obj.data(idx);
      else
        out = builtin('subsref', obj, S);
      end
    end
  end
end
"#,
    )
    .expect("write C.m");

    std::fs::write(
        root.join("main.m"),
        r#"
c = C([10 20 30]);
a = c(2);
b = c.getdata();
d = c.data(3);
"#,
    )
    .expect("write main.m");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd_guard = push_cwd(root);
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(20.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "d",
        &runmat_builtins::Value::Num(30.0)
    ));
}

#[test]
fn execute_path_request_supports_idiomatic_classdef_source_layout_end_to_end() {
    let handle = std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || {
            let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
            let tmp = tempfile::TempDir::new().expect("tempdir");
            std::fs::write(
                tmp.path().join("Vec2.m"),
                r#"
classdef Vec2
  properties
    x
    y
  end
  methods
    function obj = Vec2(x, y)
      obj.x = x;
      obj.y = y;
    end
    function out = magnitude(obj)
      out = sqrt(obj.x^2 + obj.y^2);
    end
  end
  methods (Static)
    function u = unit()
      u = Vec2(1, 0);
    end
  end
end
"#,
            )
            .expect("write Vec2 class source");
            std::fs::write(
                tmp.path().join("Money.m"),
                r#"
classdef Money
  properties
    amount
  end
  methods
    function obj = Money(a)
      obj.amount = a;
    end
    function out = plus(a, b)
      out = Money(a.amount + b.amount);
    end
  end
end
"#,
            )
            .expect("write Money class source");
            std::fs::write(
                tmp.path().join("main.m"),
                r#"
p = Vec2(3,4);
m = p.magnitude();
u = Vec2.unit();
a = Money(10); b = Money(5); c = a + b;
t = class(p);
ok = isa(p,'Vec2');
ux = u.x;
total = c.amount;
"#,
            )
            .expect("write main source");

            let mut session =
                RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
            let source_path = tmp.path().join("main.m");
            let outcome =
                execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
                    .expect("execute script");

            assert!(
                outcome_has_named_upsert(
                    &outcome,
                    "t",
                    &runmat_builtins::Value::String("Vec2".to_string())
                ),
                "class() should preserve source class identity"
            );
            assert!(
                outcome_has_named_upsert(&outcome, "ok", &runmat_builtins::Value::Bool(true)),
                "isa() should report source class membership"
            );
            assert!(
                outcome_has_named_upsert(&outcome, "m", &runmat_builtins::Value::Num(5.0)),
                "dot-method dispatch should resolve source-authored instance methods"
            );
            assert!(
                outcome_has_named_upsert(&outcome, "ux", &runmat_builtins::Value::Num(1.0)),
                "static method dispatch should resolve Class.method() calls"
            );
            assert!(
                outcome_has_named_upsert(&outcome, "total", &runmat_builtins::Value::Num(15.0)),
                "operator overload dispatch should resolve source-authored plus"
            );
        })
        .expect("spawn idiomatic classdef e2e thread");
    handle.join().expect("idiomatic classdef e2e thread failed");
}

#[test]
fn execute_path_request_supports_package_qualified_classdef_source_layout_end_to_end() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg").join("Vec2.m"),
        r#"
classdef Vec2
  properties
    x
    y
  end
  methods
    function obj = Vec2(x, y)
      obj.x = x;
      obj.y = y;
    end
    function out = magnitude(obj)
      out = sqrt(obj.x^2 + obj.y^2);
    end
  end
  methods (Static)
    function u = unit()
      u = pkg.Vec2(1, 0);
    end
  end
end
"#,
    )
    .expect("write pkg.Vec2 class source");
    std::fs::write(
        tmp.path().join("main.m"),
        r#"
p = pkg.Vec2(3,4);
m = p.magnitude();
u = pkg.Vec2.unit();
t = class(p);
ok = isa(p,'pkg.Vec2');
ux = u.x;
"#,
    )
    .expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(
        outcome_has_named_upsert(
            &outcome,
            "t",
            &runmat_builtins::Value::String("pkg.Vec2".to_string())
        ),
        "class() should preserve qualified source class identity"
    );
    assert!(
        outcome_has_named_upsert(&outcome, "ok", &runmat_builtins::Value::Bool(true)),
        "isa() should report qualified source class membership"
    );
    assert!(
        outcome_has_named_upsert(&outcome, "m", &runmat_builtins::Value::Num(5.0)),
        "dot-method dispatch should resolve source-authored package instance methods"
    );
    assert!(
        outcome_has_named_upsert(&outcome, "ux", &runmat_builtins::Value::Num(1.0)),
        "static method dispatch should resolve package-qualified Class.method() calls"
    );
}

#[test]
fn execute_path_request_supports_nested_package_classdef_source_layout_end_to_end() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg").join("+sub")).expect("create nested package");
    std::fs::write(
        tmp.path().join("+pkg").join("+sub").join("C.m"),
        r#"
classdef C
  properties
    x
  end
  methods
    function obj = C(v)
      obj.x = v;
    end
    function y = twice(obj)
      y = obj.x * 2;
    end
  end
  methods(Static)
    function z = three()
      z = 3;
    end
  end
end
"#,
    )
    .expect("write pkg.sub.C class source");
    std::fs::write(
        tmp.path().join("main.m"),
        r#"
o = pkg.sub.C(7);
a = o.twice();
b = pkg.sub.C.three();
cls = class(o);
ok = isa(o,'pkg.sub.C');
"#,
    )
    .expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(14.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(3.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "cls",
        &runmat_builtins::Value::String("pkg.sub.C".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "ok",
        &runmat_builtins::Value::Bool(true)
    ));
}

#[test]
fn execute_path_request_supports_package_source_class_subsref_subsasgn_dispatch() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg").join("OverIdx.m"),
        r#"
classdef OverIdx
  properties
    x
  end
  methods
    function obj = OverIdx(v)
      obj.x = v;
    end
    function out = subsref(obj, S)
      if strcmp(S(1).type, '.')
        if strcmp(S(1).subs, 'x')
          out = obj.x + 1;
          return;
        end
      end
      out = builtin('subsref', obj, S);
    end
    function obj = subsasgn(obj, S, rhs)
      if strcmp(S(1).type, '.')
        if strcmp(S(1).subs, 'x')
          obj.x = rhs + 2;
          return;
        end
      end
      obj = builtin('subsasgn', obj, S, rhs);
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "o = pkg.OverIdx(5); a = o.x; o.x = 10; b = o.x;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(6.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(13.0)
    ));
}

#[test]
fn execute_path_request_supports_package_static_property_reference_inside_methods() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg").join("C.m"),
        r#"
classdef C
  properties(Constant)
    K = 9;
  end
  properties
    x
  end
  methods
    function obj = C(v)
      obj.x = v;
    end
    function y = f(obj)
      y = obj.x + pkg.C.K;
    end
  end
  methods(Static)
    function y = g(v)
      y = v + pkg.C.K;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "o = pkg.C(1); a = o.f(); b = pkg.C.g(2); c = pkg.C.K;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(10.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(11.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "c",
        &runmat_builtins::Value::Num(9.0)
    ));
}

#[test]
fn execute_path_request_supports_import_wildcard_unqualified_package_class_constructor_calls() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg").join("C.m"),
        r#"
classdef C
  properties(Constant)
    K = 9;
  end
  properties
    x
  end
  methods
    function obj = C(v)
      obj.x = v;
    end
    function y = f(obj)
      y = obj.x + pkg.C.K;
    end
  end
  methods(Static)
    function y = g(v)
      y = v + pkg.C.K;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "import pkg.*; o = C(1); a = o.f(); b = C.g(2); c = C.K;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(10.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(11.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "c",
        &runmat_builtins::Value::Num(9.0)
    ));
}

#[test]
fn execute_path_request_supports_import_specific_unqualified_package_class_constructor_calls() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg").join("C.m"),
        r#"
classdef C
  properties(Constant)
    K = 9;
  end
  properties
    x
  end
  methods
    function obj = C(v)
      obj.x = v;
    end
    function y = f(obj)
      y = obj.x + pkg.C.K;
    end
  end
  methods(Static)
    function y = g(v)
      y = v + pkg.C.K;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "import pkg.C; o = C(1); a = o.f(); b = C.g(2); c = C.K;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(10.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(11.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "c",
        &runmat_builtins::Value::Num(9.0)
    ));
}

#[test]
fn execute_path_request_reports_import_ambiguous_for_unqualified_class_constructor_wildcards() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::create_dir_all(tmp.path().join("+pkg2")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg").join("C.m"),
        "classdef C; methods; function obj = C(), end; end; end",
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("+pkg2").join("C.m"),
        "classdef C; methods; function obj = C(), end; end; end",
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "import pkg.*; import pkg2.*; o = C();",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let err = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect_err("ambiguous wildcard-import class constructor should fail compilation");
    let RunError::Semantic(err) = err else {
        panic!("expected semantic import ambiguity error");
    };
    assert_eq!(err.identifier.as_deref(), Some("RunMat:ImportAmbiguous"));
}

#[test]
fn execute_path_request_specific_import_precedes_wildcard_for_unqualified_class_name() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::create_dir_all(tmp.path().join("+pkg2")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg").join("C.m"),
        r#"
classdef C
  properties(Constant)
    K = 9;
  end
  methods
    function obj = C(), end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("+pkg2").join("C.m"),
        r#"
classdef C
  properties(Constant)
    K = 17;
  end
  methods
    function obj = C(), end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "import pkg.C; import pkg2.*; a = C.K;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(9.0)
    ));
}

#[test]
fn execute_path_request_supports_imported_class_static_function_handle_resolution() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg").join("C.m"),
        r#"
classdef C
  methods(Static)
    function y = g(v)
      y = v + 9;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "import pkg.C; fh = @C.g; y = fh(2);",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(11.0)
    ));
}

#[test]
fn execute_path_request_supports_import_wildcard_class_constructor_function_handle() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg").join("C.m"),
        r#"
classdef C
  properties
    x
  end
  methods
    function obj = C(v)
      obj.x = v;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "import pkg.*; fh = @C; o = fh(7); y = o.x;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(7.0)
    ));
}

#[test]
fn execute_path_request_reports_import_ambiguous_for_constructor_function_handle_wildcards() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::create_dir_all(tmp.path().join("+pkg2")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg").join("C.m"),
        "classdef C; methods; function obj = C(), end; end; end",
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("+pkg2").join("C.m"),
        "classdef C; methods; function obj = C(), end; end; end",
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "import pkg.*; import pkg2.*; fh = @C; o = fh();",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let err = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect_err("ambiguous wildcard-import constructor handle should fail compilation");
    let RunError::Semantic(err) = err else {
        panic!("expected semantic import ambiguity error");
    };
    assert_eq!(err.identifier.as_deref(), Some("RunMat:ImportAmbiguous"));
}

#[test]
fn execute_path_request_specific_import_precedes_wildcard_for_constructor_function_handle() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::create_dir_all(tmp.path().join("+pkg2")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg").join("C.m"),
        r#"
classdef C
  properties
    x
  end
  methods
    function obj = C(v)
      obj.x = v;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("+pkg2").join("C.m"),
        "classdef C; methods; function obj = C(), end; end; end",
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "import pkg.C; import pkg2.*; fh = @C; o = fh(7); y = o.x;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(7.0)
    ));
}

#[test]
fn execute_path_request_specific_import_precedes_wildcard_for_static_method_handle() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::create_dir_all(tmp.path().join("+pkg2")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg").join("C.m"),
        r#"
classdef C
  methods(Static)
    function y = g(v)
      y = v + 1;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("+pkg2").join("C.m"),
        r#"
classdef C
  methods(Static)
    function y = g(v)
      y = v + 99;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "import pkg.C; import pkg2.*; fh = @C.g; y = fh(3);",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(4.0)
    ));
}

#[test]
fn execute_path_request_supports_imported_static_listener_callback_dispatch() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg").join("CB.m"),
        r#"
classdef CB
  methods(Static)
    function k(src, v)
      src.x = v + 1;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "import pkg.CB; __register_test_classes(); h = new_handle_object('Point'); addlistener(h, 'changed', @CB.k); notify(h, 'changed', 42); y = h.x;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(43.0)
    ));
}

#[test]
fn execute_path_request_rejects_getmethod_for_private_method_from_outside_class() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("C.m"),
        r#"
classdef C
  methods(Access=private)
    function y = pvt(obj, x)
      y = x + 5;
    end
  end
  methods
    function y = pub(obj, x)
      y = pvt(obj, x);
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "c = C(); a = c.pub(2); try, fh = getmethod(c, 'pvt'); b = fh(3); id = 'NOERR'; catch e, id = e.identifier; end;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(7.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "id",
        &runmat_builtins::Value::String("RunMat:MethodPrivate".to_string())
    ));
}

#[test]
fn execute_path_request_allows_getmethod_for_protected_method_inside_subclass_only() {
    let result = std::thread::Builder::new()
        .name("protected-getmethod-subclass".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
            let tmp = tempfile::TempDir::new().expect("tempdir");
            std::fs::write(
                tmp.path().join("A.m"),
                r#"
classdef A
  methods(Access=protected)
    function y = prot(obj, x)
      y = x + 10;
    end
  end
end
"#,
            )
            .expect("write class source");
            std::fs::write(
                tmp.path().join("B.m"),
                r#"
classdef B < A
  methods
    function y = via_handle(obj, x)
      fh = getmethod(obj, 'prot');
      y = fh(x);
    end
  end
end
"#,
            )
            .expect("write class source");
            std::fs::write(
                tmp.path().join("main.m"),
                "a = A(); b = B(); y1 = b.via_handle(3); try, fh = getmethod(a, 'prot'); y2 = fh(4); id = 'NOERR'; catch e, id = e.identifier; end;",
            )
            .expect("write script source");

            let mut session =
                RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
            let source_path = tmp.path().join("main.m");
            let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
                .expect("execute script");

            assert!(outcome_has_named_upsert(
                &outcome,
                "y1",
                &runmat_builtins::Value::Num(13.0)
            ));
            assert!(outcome_has_named_upsert(
                &outcome,
                "id",
                &runmat_builtins::Value::String("RunMat:MethodPrivate".to_string())
            ));
        })
        .expect("spawn protected-getmethod-subclass test thread")
        .join();
    assert!(
        result.is_ok(),
        "protected getmethod subclass thread panicked"
    );
}

#[test]
fn execute_path_request_supports_getmethod_on_classref_for_public_static_method() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("S.m"),
        r#"
classdef S
  methods(Static)
    function y = pub(x)
      y = x + 11;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "cls = classref('S'); fh = getmethod(cls, 'pub'); y = fh(4);",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(15.0)
    ));
}

#[test]
fn execute_path_request_supports_getmethod_on_classref_for_inherited_static_method() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("A.m"),
        r#"
classdef A
  methods(Static)
    function y = util(x)
      y = x + 100;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("B.m"),
        r#"
classdef B < A
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "cls = classref('B'); fh = getmethod(cls, 'util'); y = fh(2);",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(102.0)
    ));
}

#[test]
fn execute_path_request_rejects_getmethod_on_classref_for_private_static_method() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("S.m"),
        r#"
classdef S
  methods(Static, Access=private)
    function y = pvt(x)
      y = x + 11;
    end
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "cls = classref('S'); try, fh = getmethod(cls, 'pvt'); y = fh(4); id = 'NOERR'; catch e, id = e.identifier; end;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "id",
        &runmat_builtins::Value::String("RunMat:MethodPrivate".to_string())
    ));
}

#[test]
fn execute_path_request_supports_single_file_classdef_with_trailing_script() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("main.m"),
        r#"
classdef Vec2
  properties
    x
    y
  end
  methods
    function obj = Vec2(x, y)
      obj.x = x;
      obj.y = y;
    end
    function m = magnitude(obj)
      m = sqrt(obj.x * obj.x + obj.y * obj.y);
    end
  end
end
p = Vec2(3, 4); c = class(p); m = p.magnitude(); ok = isa(p, 'Vec2');
"#,
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "c",
        &runmat_builtins::Value::String("Vec2".to_string())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "m",
        &runmat_builtins::Value::Num(5.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "ok",
        &runmat_builtins::Value::Bool(true)
    ));
}

#[test]
fn execute_path_request_supports_new_object_builtin_for_registered_classes() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("main.m"),
        "__register_test_classes(); o = new_object('Point'); c = class(o); x = o.x;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "c",
        &runmat_builtins::Value::String("Point".to_string())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "x",
        &runmat_builtins::Value::Num(0.0)
    ));
}

#[test]
fn execute_path_request_custom_subsref_intercepts_dot_member_access() {
    let _cwd_lock = CWD_LOCK.lock().unwrap();
    let project = tempfile::tempdir().expect("tempdir");
    let root = project.path();

    std::fs::write(
        root.join("DotSpy.m"),
        r#"
classdef DotSpy
  properties
    x
  end
  methods
    function obj = DotSpy(v)
      obj.x = v;
    end
    function out = getx(obj)
      out = obj.x;
    end
    function out = subsref(obj, S)
      if strcmp(S(1).type, '.')
        out = 999;
      else
        out = builtin('subsref', obj, S);
      end
    end
  end
end
"#,
    )
    .expect("write DotSpy.m");

    std::fs::write(
        root.join("main.m"),
        r#"
d = DotSpy(3);
a = d.x;
b = d.getx();
"#,
    )
    .expect("write main.m");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd_guard = push_cwd(root);
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "a",
        &runmat_builtins::Value::Num(999.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "b",
        &runmat_builtins::Value::Num(999.0)
    ));
}

#[test]
fn execute_path_request_custom_subsasgn_intercepts_dot_member_assignment() {
    let _cwd_lock = CWD_LOCK.lock().unwrap();
    let project = tempfile::tempdir().expect("tempdir");
    let root = project.path();

    std::fs::write(
        root.join("DotSetSpy.m"),
        r#"
classdef DotSetSpy
  properties
    x
  end
  methods
    function obj = DotSetSpy(v)
      obj.x = v;
    end
    function obj = subsasgn(obj, S, rhs)
      if strcmp(S(1).type, '.')
        return;
      else
        obj = builtin('subsasgn', obj, S, rhs);
      end
    end
  end
end
"#,
    )
    .expect("write DotSetSpy.m");

    std::fs::write(
        root.join("main.m"),
        r#"
d = DotSetSpy(3);
d.x = 5;
y = d.x;
"#,
    )
    .expect("write main.m");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd_guard = push_cwd(root);
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(3.0)
    ));
}

#[test]
fn execute_path_request_builtin_subsref_dot_falls_back_to_method_dispatch() {
    let _cwd_lock = CWD_LOCK.lock().unwrap();
    let project = tempfile::tempdir().expect("tempdir");
    let root = project.path();

    std::fs::write(
        root.join("C.m"),
        r#"
classdef C
  methods
    function y = getv(obj)
      y = 42;
    end
    function out = subsref(obj, S)
      out = builtin('subsref', obj, S);
    end
  end
end
"#,
    )
    .expect("write C.m");

    std::fs::write(root.join("main.m"), "c = C(); y = c.getv();").expect("write main.m");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd_guard = push_cwd(root);
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(42.0)
    ));
}

#[test]
fn execute_path_request_builtin_subsasgn_dot_falls_back_to_setfield() {
    let _cwd_lock = CWD_LOCK.lock().unwrap();
    let project = tempfile::tempdir().expect("tempdir");
    let root = project.path();

    std::fs::write(
        root.join("C.m"),
        r#"
classdef C
  properties
    x
  end
  methods
    function obj = C(v)
      obj.x = v;
    end
    function obj = subsasgn(obj, S, rhs)
      obj = builtin('subsasgn', obj, S, rhs);
    end
  end
end
"#,
    )
    .expect("write C.m");

    std::fs::write(root.join("main.m"), "c = C(1); c.x = 9; y = c.x;").expect("write main.m");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd_guard = push_cwd(root);
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(9.0)
    ));
}

#[test]
fn execute_path_request_rejects_static_property_access_via_instance_with_identifier() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("S.m"),
        r#"
classdef S
  properties(Static)
    v
  end
end
"#,
    )
    .expect("write class source");
    std::fs::write(
        tmp.path().join("main.m"),
        "s = S(); try, a = s.v; id='OK'; catch e, id=e.identifier; a=-1; end; try, s.v = 3; id2='OK'; catch e2, id2=e2.identifier; end; b = S.v;",
    )
    .expect("write script source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_path = tmp.path().join("main.m");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute script");

    assert!(outcome_has_named_upsert(
        &outcome,
        "id",
        &runmat_builtins::Value::String("RunMat:PropertyStaticAccess".into())
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "id2",
        &runmat_builtins::Value::String("RunMat:PropertyStaticAccess".into())
    ));
    let empty = runmat_builtins::Value::Tensor(
        runmat_builtins::Tensor::new(vec![], vec![0, 0]).expect("empty tensor"),
    );
    assert!(outcome_has_named_upsert(&outcome, "b", &empty));
}

#[test]
fn compile_input_reports_import_ambiguity_identifier() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let err = match session.compile_input("import PkgF.foo; import PkgG.foo; y = foo();") {
        Ok(_) => panic!("ambiguous specific imports should fail compilation"),
        Err(err) => err,
    };
    let RunError::Semantic(err) = err else {
        panic!("expected semantic import ambiguity error");
    };
    assert_eq!(err.identifier.as_deref(), Some("RunMat:ImportAmbiguous"));
}

#[test]
fn compile_input_reports_duplicate_import_identifier() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let err = match session.compile_input("import Point.*; import Point.*; y = 1;") {
        Ok(_) => panic!("duplicate imports should fail compilation"),
        Err(err) => err,
    };
    let RunError::Semantic(err) = err else {
        panic!("expected semantic duplicate import error");
    };
    assert_eq!(err.identifier.as_deref(), Some("RunMat:ImportDuplicate"));
}

#[test]
fn execute_outcome_resolves_wildcard_import_from_project_package_function() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("+pkg/foo.m"),
        "function y = foo(); y = 42; end",
    )
    .expect("write package function");
    std::fs::write(tmp.path().join("main.m"), "x = 1;").expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let source_name = tmp.path().join("main.m").to_string_lossy().to_string();
    let outcome = execute_path_request(&mut session, &source_name).expect("exec succeeds");

    assert!(
        outcome.diagnostics.is_empty(),
        "package wildcard import execution should resolve without diagnostics: {:?}",
        outcome.diagnostics
    );
}

#[test]
fn execute_path_request_resolves_wildcard_import_package_function_call_with_manifest() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("+pkg/foo.m"),
        "function y = foo(x); y = x + 2; end",
    )
    .expect("write package function");
    std::fs::write(tmp.path().join("main.m"), "import pkg.*; r = foo(40);")
        .expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let source_name = tmp.path().join("main.m").to_string_lossy().to_string();
    let outcome = execute_path_request(&mut session, &source_name).expect("exec succeeds");

    assert!(
        outcome.diagnostics.is_empty(),
        "wildcard-imported package function call should resolve; diagnostics={:?}",
        outcome.diagnostics
    );
    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(42.0)),
        "expected wildcard-imported package function result binding; upserts={:?}",
        outcome.workspace_delta.upserts
    );
}

#[test]
fn execute_path_request_resolves_wildcard_import_package_function_call_with_manifest_relative_path()
{
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("+pkg/foo.m"),
        "function y = foo(x); y = x + 2; end",
    )
    .expect("write package function");
    std::fs::write(tmp.path().join("main.m"), "import pkg.*; r = foo(40);")
        .expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");

    assert!(
        outcome.diagnostics.is_empty(),
        "wildcard-imported package function call should resolve with relative source path; diagnostics={:?}",
        outcome.diagnostics
    );
    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(42.0)),
        "expected wildcard-imported package function result binding; upserts={:?}",
        outcome.workspace_delta.upserts
    );
}

#[test]
fn execute_path_request_resolves_subdir_helper_function_with_manifest_relative_path() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("helpers")).expect("create helper dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("helpers/add1.m"),
        "function y = add1(x); y = x + 1; end",
    )
    .expect("write helper function");
    std::fs::write(tmp.path().join("main.m"), "r = add1(41);").expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");

    assert!(
        outcome.diagnostics.is_empty(),
        "subdir helper function call should resolve with relative source path; diagnostics={:?}",
        outcome.diagnostics
    );
    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(42.0)),
        "expected helper function result binding; upserts={:?}",
        outcome.workspace_delta.upserts
    );
}

#[test]
fn execute_path_request_resolves_private_function_for_parent_source() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("private")).expect("create private dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("private/helper.m"),
        "function y = helper(x); y = x + 1; end",
    )
    .expect("write private helper");
    std::fs::write(
        tmp.path().join("main.m"),
        "h = @helper; s = str2func('helper'); a = helper(40); b = h(41); c = feval('helper', 42); d = feval(s, 43); r = a + b + c + d;",
    )
    .expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");

    assert!(
        outcome.diagnostics.is_empty(),
        "parent source should resolve private helper through direct, handle, and feval routes; diagnostics={:?}",
        outcome.diagnostics
    );
    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(170.0)),
        "expected private helper result binding; upserts={:?}",
        outcome.workspace_delta.upserts
    );
}

#[test]
fn execute_path_request_resolves_private_function_without_manifest_for_parent_source() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("private")).expect("create private dir");
    std::fs::write(
        tmp.path().join("private/helper.m"),
        "function y = helper(x); y = x + 1; end",
    )
    .expect("write private helper");
    std::fs::write(tmp.path().join("main.m"), "r = helper(41);").expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");

    assert!(
        outcome.diagnostics.is_empty(),
        "parent source should resolve private helper without a manifest; diagnostics={:?}",
        outcome.diagnostics
    );
    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(42.0)),
        "expected private helper result binding without manifest; upserts={:?}",
        outcome.workspace_delta.upserts
    );
}

#[test]
fn execute_path_request_does_not_resolve_private_function_for_sibling_source() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("private")).expect("create private dir");
    std::fs::create_dir_all(tmp.path().join("sub")).expect("create sub dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("private/helper.m"),
        "function y = helper(x); y = x + 1; end",
    )
    .expect("write private helper");
    std::fs::write(
        tmp.path().join("sub/main.m"),
        "try; direct = helper(1); direct_eid = 'NOERR'; catch e; direct_eid = e.identifier; end; try; s = str2func('helper'); dyn = feval(s, 1); dyn_eid = 'NOERR'; catch e; dyn_eid = e.identifier; end;",
    )
    .expect("write sub source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "sub/main.m").expect("exec succeeds");

    assert!(
        outcome_has_named_upsert(
            &outcome,
            "direct_eid",
            &runmat_builtins::Value::String("RunMat:UndefinedFunction".to_string())
        ),
        "sibling source should not resolve parent private helper directly; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
    assert!(
        outcome_has_named_upsert(
            &outcome,
            "dyn_eid",
            &runmat_builtins::Value::String("RunMat:UndefinedFunction".to_string())
        ),
        "sibling source should not resolve parent private helper through str2func/feval; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
}

#[test]
fn execute_path_request_private_function_does_not_persist_into_session_registry() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("private")).expect("create private dir");
    std::fs::create_dir_all(tmp.path().join("sub")).expect("create sub dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("private/helper.m"),
        "function y = helper(x); y = x + 1; end",
    )
    .expect("write private helper");
    std::fs::write(tmp.path().join("main.m"), "r = helper(41);").expect("write main source");
    std::fs::write(
        tmp.path().join("sub/main.m"),
        "try; r = helper(1); eid = 'NOERR'; catch e; eid = e.identifier; end;",
    )
    .expect("write sub source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let parent_outcome = execute_path_request(&mut session, "main.m").expect("parent succeeds");
    assert!(
        outcome_has_named_upsert(&parent_outcome, "r", &runmat_builtins::Value::Num(42.0)),
        "parent source should resolve private helper; upserts={:?}",
        parent_outcome.workspace_delta.upserts
    );

    let sibling_outcome =
        execute_path_request(&mut session, "sub/main.m").expect("sibling succeeds");
    assert!(
        outcome_has_named_upsert(
            &sibling_outcome,
            "eid",
            &runmat_builtins::Value::String("RunMat:UndefinedFunction".to_string())
        ),
        "sibling source should not resolve private helper through session registry; upserts={:?}, diagnostics={:?}",
        sibling_outcome.workspace_delta.upserts,
        sibling_outcome.diagnostics
    );
}

#[test]
fn execute_path_request_local_function_shadows_private_function() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("private")).expect("create private dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("private/helper.m"),
        "function y = helper(x); y = x + 100; end",
    )
    .expect("write private helper");
    std::fs::write(
        tmp.path().join("main.m"),
        "r = helper(1);\nfunction y = helper(x)\ny = x + 1;\nend\n",
    )
    .expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");

    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(2.0)),
        "local function should shadow same-named private helper; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
}

#[test]
fn execute_path_request_private_function_shadows_public_sibling_function() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("private")).expect("create private dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("helper.m"),
        "function y = helper(x); y = x + 100; end",
    )
    .expect("write public helper");
    std::fs::write(
        tmp.path().join("private/helper.m"),
        "function y = helper(x); y = x + 1; end",
    )
    .expect("write private helper");
    std::fs::write(tmp.path().join("main.m"), "r = helper(41);").expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");

    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(42.0)),
        "visible private helper should shadow same-named public sibling; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
}

#[test]
fn execute_path_request_private_function_shadows_imported_and_builtin_names() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("private")).expect("create private dir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("+pkg/helper.m"),
        "function y = helper(x); y = x + 100; end",
    )
    .expect("write package helper");
    std::fs::write(
        tmp.path().join("private/helper.m"),
        "function y = helper(x); y = x + 1; end",
    )
    .expect("write private helper");
    std::fs::write(
        tmp.path().join("private/sum.m"),
        "function y = sum(x); y = x + 10; end",
    )
    .expect("write private builtin-name helper");
    std::fs::write(
        tmp.path().join("main.m"),
        "import pkg.*; a = helper(41); b = sum(32); r = a + b;",
    )
    .expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");

    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(84.0)),
        "private helpers should shadow wildcard-imported and builtin names; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
}

#[test]
fn execute_path_request_private_function_shadows_dependency_alias_import() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let dep_root = tmp.path().join("deps/statslib");
    std::fs::create_dir_all(tmp.path().join("private")).expect("create private dir");
    std::fs::create_dir_all(&dep_root).expect("create dependency dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]

[dependencies]
statsdep = { path = "deps/statslib" }
"#,
    )
    .expect("write root manifest");
    std::fs::write(
        dep_root.join("runmat.toml"),
        r#"
[package]
name = "statslib"

[sources]
roots = ["."]
"#,
    )
    .expect("write dependency manifest");
    std::fs::write(
        dep_root.join("helper.m"),
        "function y = helper(x); y = x + 100; end",
    )
    .expect("write dependency helper");
    std::fs::write(
        tmp.path().join("private/helper.m"),
        "function y = helper(x); y = x + 1; end",
    )
    .expect("write private helper");
    std::fs::write(
        tmp.path().join("main.m"),
        "import statsdep.*; r = helper(41);",
    )
    .expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");

    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(42.0)),
        "private helper should shadow same-named dependency-alias wildcard import; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
}

#[test]
fn execute_path_request_resolves_package_private_function_for_active_package_source() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg/private")).expect("create package private dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(tmp.path().join("+pkg/main.m"), "r = helper(41);")
        .expect("write package source");
    std::fs::write(
        tmp.path().join("+pkg/private/helper.m"),
        "function y = helper(x); y = x + 1; end",
    )
    .expect("write package private helper");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "+pkg/main.m").expect("exec succeeds");

    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(42.0)),
        "active package source should resolve its package private helper; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
}

#[test]
fn execute_path_request_resolves_class_folder_private_function_for_active_class_folder_source() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("@C/private")).expect("create class private dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(tmp.path().join("@C/main.m"), "r = helper(41);")
        .expect("write class-folder source");
    std::fs::write(
        tmp.path().join("@C/private/helper.m"),
        "function y = helper(x); y = x + 1; end",
    )
    .expect("write class private helper");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "@C/main.m").expect("exec succeeds");

    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(42.0)),
        "active class-folder source should resolve its class-folder private helper; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
}

#[test]
fn execute_path_request_resolves_package_private_function_for_package_callee_only() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg/private")).expect("create package private dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("+pkg/entry.m"),
        "function y = entry(); a = helper(40); h = @helper; b = h(41); c = feval(@helper, 42); y = a + b + c; end",
    )
    .expect("write package entry");
    std::fs::write(
        tmp.path().join("+pkg/private/helper.m"),
        "function y = helper(x); y = x + 1; end",
    )
    .expect("write package private helper");
    std::fs::write(
        tmp.path().join("main.m"),
        "r = pkg.entry(); try; leak = helper(1); leak_eid = 'NOERR'; catch e; leak_eid = e.identifier; end;",
    )
    .expect("write root source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");

    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(126.0)),
        "package callee should resolve its private helper through direct, handle, and feval(@handle) routes; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
    assert!(
        outcome_has_named_upsert(
            &outcome,
            "leak_eid",
            &runmat_builtins::Value::String("RunMat:UndefinedFunction".to_string())
        ),
        "root caller should not resolve package private helper directly; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
}

#[test]
fn execute_path_request_resolves_package_private_string_routes_for_package_callee() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("private")).expect("create root private dir");
    std::fs::create_dir_all(tmp.path().join("+pkg/private")).expect("create package private dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("private/helper.m"),
        "function y = helper(x); y = x + 100; end",
    )
    .expect("write root private helper");
    std::fs::write(
        tmp.path().join("+pkg/entry.m"),
        "function y = entry(); a = helper(40); s = str2func('helper'); b = feval(s, 41); c = feval('helper', 42); y = a + b + c; end",
    )
    .expect("write package entry");
    std::fs::write(
        tmp.path().join("+pkg/private/helper.m"),
        "function y = helper(x); y = x + 1; end",
    )
    .expect("write package private helper");
    std::fs::write(
        tmp.path().join("main.m"),
        "root_value = helper(1); pkg_value = pkg.entry();",
    )
    .expect("write root source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");

    assert!(
        outcome_has_named_upsert(&outcome, "root_value", &runmat_builtins::Value::Num(101.0)),
        "root source should resolve root private helper; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
    assert!(
        outcome_has_named_upsert(&outcome, "pkg_value", &runmat_builtins::Value::Num(126.0)),
        "package callee string routes should prefer package private helper; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
}

#[test]
fn execute_path_request_package_private_function_precedes_root_private_for_package_callee() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("private")).expect("create root private dir");
    std::fs::create_dir_all(tmp.path().join("+pkg/private")).expect("create package private dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("private/helper.m"),
        "function y = helper(x); y = x + 100; end",
    )
    .expect("write root private helper");
    std::fs::write(
        tmp.path().join("+pkg/private/helper.m"),
        "function y = helper(x); y = x + 1; end",
    )
    .expect("write package private helper");
    std::fs::write(
        tmp.path().join("+pkg/entry.m"),
        "function y = entry(); y = helper(41); end",
    )
    .expect("write package entry");
    std::fs::write(
        tmp.path().join("main.m"),
        "root_value = helper(1); pkg_value = pkg.entry();",
    )
    .expect("write root source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");

    assert!(
        outcome_has_named_upsert(&outcome, "root_value", &runmat_builtins::Value::Num(101.0)),
        "root source should resolve root private helper; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
    assert!(
        outcome_has_named_upsert(&outcome, "pkg_value", &runmat_builtins::Value::Num(42.0)),
        "package callee should prefer its package private helper over root private helper; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
}

#[test]
fn execute_path_request_resolves_class_folder_private_function_for_class_folder_callee() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("private")).expect("create root private dir");
    std::fs::create_dir_all(tmp.path().join("@C/private")).expect("create class private dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("private/helper.m"),
        "function y = helper(x); y = x + 100; end",
    )
    .expect("write root private helper");
    std::fs::write(
        tmp.path().join("@C/entry.m"),
        "function y = entry(); a = helper(40); h = @helper; b = h(41); s = str2func('helper'); c = feval(s, 42); d = feval('helper', 43); y = a + b + c + d; end",
    )
    .expect("write class-folder entry");
    std::fs::write(
        tmp.path().join("@C/private/helper.m"),
        "function y = helper(x); y = x + 1; end",
    )
    .expect("write class private helper");
    std::fs::write(
        tmp.path().join("main.m"),
        "root_value = helper(1); class_value = C.entry();",
    )
    .expect("write root source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");

    assert!(
        outcome_has_named_upsert(&outcome, "root_value", &runmat_builtins::Value::Num(101.0)),
        "root source should resolve root private helper; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
    assert!(
        outcome_has_named_upsert(&outcome, "class_value", &runmat_builtins::Value::Num(170.0)),
        "class-folder callee should resolve its private helper through direct, handle, str2func, and feval string routes; upserts={:?}, diagnostics={:?}",
        outcome.workspace_delta.upserts,
        outcome.diagnostics
    );
}

#[test]
fn execute_outcome_qualified_package_function_call_resolves_from_source_roots() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("+pkg/foo.m"),
        "function y = foo(); y = 42; end",
    )
    .expect("write package function");
    std::fs::write(tmp.path().join("main.m"), "r = pkg.foo();").expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let source_name = tmp.path().join("main.m").to_string_lossy().to_string();
    let outcome = execute_path_request(&mut session, &source_name).expect("exec succeeds");

    assert!(
        outcome.diagnostics.is_empty(),
        "qualified package function in configured sources root should resolve; diagnostics={:?}",
        outcome.diagnostics
    );
    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(42.0)),
        "qualified package function should execute and bind result; upserts={:?}",
        outcome.workspace_delta.upserts
    );
}

#[test]
fn execute_outcome_wildcard_import_without_manifest_reports_unresolved_function() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("+pkg/foo.m"),
        "function y = foo(); y = 42; end",
    )
    .expect("write package function");
    std::fs::write(tmp.path().join("main.m"), "import pkg.*; foo()").expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let source_name = tmp.path().join("main.m").to_string_lossy().to_string();
    let outcome = execute_path_request(&mut session, &source_name).expect("exec succeeds");

    assert!(
        outcome
            .diagnostics
            .iter()
            .any(|d| d.code == "RunMat:UndefinedFunction" && d.message.contains("foo")),
        "wildcard import without manifest should not resolve package symbols; diagnostics={:?}",
        outcome.diagnostics
    );
}

#[test]
fn execute_outcome_unqualified_helper_from_source_tree_requires_registered_symbol() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("helpers")).expect("create helper dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("helpers/add1.m"),
        "function y = add1(x); y = x + 1; end",
    )
    .expect("write helper function");
    std::fs::write(tmp.path().join("main.m"), "add1(41)").expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let source_name = tmp.path().join("main.m").to_string_lossy().to_string();
    let outcome = execute_path_request(&mut session, &source_name).expect("exec succeeds");

    assert!(
        outcome.diagnostics.is_empty(),
        "unqualified helper function in configured sources root should resolve; diagnostics={:?}",
        outcome.diagnostics
    );
}

#[test]
fn execute_path_request_resolves_sibling_function_file() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("outer.m"),
        "function out = outer(a)\nbase = 100;\nfunction y = add(x)\nbase = base + x;\ny = base;\nend\nout = add(a);\nend\n",
    )
    .expect("write outer function");
    std::fs::write(tmp.path().join("main.m"), "r = outer(5);").expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let source_name = tmp.path().join("main.m").to_string_lossy().to_string();
    let outcome = execute_path_request(&mut session, &source_name).expect("exec succeeds");

    assert!(
        outcome_has_named_upsert(&outcome, "r", &runmat_builtins::Value::Num(105.0)),
        "sibling function file should resolve and execute; upserts={:?}",
        outcome.workspace_delta.upserts
    );
}

#[test]
fn execute_path_request_resolves_sibling_function_with_arguments_block_validation() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(
        tmp.path().join("typed.m"),
        "function y = typed(x)\narguments\nx (1,1) double\nend\ny = x * 2;\nend\n",
    )
    .expect("write typed function");
    std::fs::write(
        tmp.path().join("main.m"),
        "ok = typed(3); try; typed('x'); catch e; eid = e.identifier; end;",
    )
    .expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let source_name = tmp.path().join("main.m").to_string_lossy().to_string();
    let outcome = execute_path_request(&mut session, &source_name).expect("exec succeeds");

    assert!(
        outcome_has_named_upsert(&outcome, "ok", &runmat_builtins::Value::Num(6.0)),
        "typed sibling function should execute; upserts={:?}",
        outcome.workspace_delta.upserts
    );
    assert!(
        outcome_has_named_upsert(
            &outcome,
            "eid",
            &runmat_builtins::Value::String("RunMat:ArgumentValidationClass".to_string())
        ),
        "typed sibling function should enforce arguments class validation; upserts={:?}",
        outcome.workspace_delta.upserts
    );
}

#[test]
fn execute_path_request_resolves_package_function_with_arguments_block_validation() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("+pkg/typed.m"),
        "function y = typed(x)\narguments\nx (1,1) double mustBeFinite\nend\ny = x * 2;\nend\n",
    )
    .expect("write package typed function");
    std::fs::write(
        tmp.path().join("main.m"),
        "ok = pkg.typed(3); try; pkg.typed([1 2]); catch e; sid = e.identifier; end; try; pkg.typed(0/0); catch e; fid = e.identifier; end;",
    )
    .expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let outcome = execute_path_request(&mut session, "main.m").expect("exec succeeds");

    assert!(
        outcome_has_named_upsert(&outcome, "ok", &runmat_builtins::Value::Num(6.0)),
        "package typed function should execute; upserts={:?}",
        outcome.workspace_delta.upserts
    );
    assert!(
        outcome_has_named_upsert(
            &outcome,
            "sid",
            &runmat_builtins::Value::String("RunMat:ArgumentValidationSize".to_string())
        ),
        "package typed function should enforce size validation; upserts={:?}",
        outcome.workspace_delta.upserts
    );
    assert!(
        outcome_has_named_upsert(
            &outcome,
            "fid",
            &runmat_builtins::Value::String("RunMat:ArgumentValidationFunction".to_string())
        ),
        "package typed function should enforce mustBeFinite validation; upserts={:?}",
        outcome.workspace_delta.upserts
    );
}

#[test]
fn execute_path_request_rejects_package_function_with_advanced_arguments_block() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+pkg")).expect("create package dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("+pkg/typed.m"),
        "function y = typed(x, varargin)\narguments (Repeating)\nvarargin double\nend\ny = x;\nend\n",
    )
    .expect("write package typed function");
    std::fs::write(tmp.path().join("main.m"), "r = pkg.typed(3, 4);").expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let err = execute_path_request(&mut session, "main.m")
        .expect_err("expected semantic failure for package advanced arguments block");
    let RunError::Semantic(err) = err else {
        panic!("expected semantic error for package advanced arguments block");
    };
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:FunctionArgumentValidationUnsupported")
    );
}

#[test]
fn execute_outcome_ignores_invalid_project_source_without_warning() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("helpers")).expect("create helper dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(tmp.path().join("helpers/bad.m"), "function y = bad(; end")
        .expect("write invalid helper function");
    std::fs::write(tmp.path().join("main.m"), "x = 1;").expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_name = tmp.path().join("main.m").to_string_lossy().to_string();
    let outcome = execute_text_request_named_source(&mut session, &source_name, "v = 1;")
        .expect("exec succeeds");

    assert!(
        outcome.diagnostics.is_empty(),
        "invalid unrelated project sources should not emit preload warnings; diagnostics={:?}",
        outcome.diagnostics
    );
}

#[test]
fn execute_outcome_load_statement_assigns_workspace_bindings_with_semicolon() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let mat_path = tmp.path().join("data.mat");
    let source_path = tmp.path().join("main.m");
    let mat_path_literal = mat_path.to_string_lossy().replace('\\', "\\\\");
    let source = format!(
        "x = 42; save('{mat_path_literal}', 'x'); clear x; load('{mat_path_literal}'); y = x;"
    );
    std::fs::write(&source_path, &source).expect("write source file");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_name = source_path.to_string_lossy().to_string();
    let outcome = execute_text_request_named_source(&mut session, &source_name, &source)
        .expect("exec succeeds");

    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(42.0)
    ));
}

#[test]
fn execute_outcome_load_statement_assigns_workspace_bindings_without_semicolon() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let mat_path = tmp.path().join("data.mat");
    let source_path = tmp.path().join("main.m");
    let mat_path_literal = mat_path.to_string_lossy().replace('\\', "\\\\");
    let source = format!(
        "x = 42; save('{mat_path_literal}', 'x'); clear x; load('{mat_path_literal}')\ny = x;"
    );
    std::fs::write(&source_path, &source).expect("write source file");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_name = source_path.to_string_lossy().to_string();
    let outcome = execute_text_request_named_source(&mut session, &source_name, &source)
        .expect("exec succeeds");

    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(42.0)
    ));
}

#[test]
fn execute_load_statement_assigns_workspace_bindings_with_semicolon() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let mat_path = tmp.path().join("data.mat");
    let mat_path_literal = mat_path.to_string_lossy().replace('\\', "\\\\");
    let source = format!(
        "x = 42; save('{mat_path_literal}', 'x'); clear x; load('{mat_path_literal}'); y = x;"
    );

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(&mut session, &source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "z = y;").expect("follow-up succeeds");

    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(
            &upsert.key,
            abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "z"
        ) && upsert.value.to_string() == "42"
    }));
}

#[test]
fn compile_input_reports_isolated_capture_identifier() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let err = match session.compile_input(
        "function y = outer(x); isolated function z = inner(); z = x; end; y = 1; end",
    ) {
        Ok(_) => panic!("isolated nested functions should reject lexical capture"),
        Err(err) => err,
    };
    let RunError::Semantic(err) = err else {
        panic!("expected semantic isolated-capture error");
    };
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:IsolatedLexicalCaptureUnsupported")
    );
}

#[test]
fn compile_input_reports_class_self_inheritance_identifier() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let err = match session.compile_input("classdef A < A; end") {
        Ok(_) => panic!("class self-inheritance should fail semantic compilation"),
        Err(err) => err,
    };
    let RunError::Semantic(err) = err else {
        panic!("expected semantic class self-inheritance error");
    };
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:ClassSelfInheritanceInvalid")
    );
}

#[test]
fn compile_input_reports_duplicate_class_member_identifier() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
classdef A
    properties
        x
        x
    end
end
"#;
    let err = match session.compile_input(source) {
        Ok(_) => panic!("duplicate class member should fail semantic compilation"),
        Err(err) => err,
    };
    let RunError::Semantic(err) = err else {
        panic!("expected semantic duplicate class member error");
    };
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:ClassMemberDuplicate")
    );
}

#[test]
fn compile_input_reports_class_member_name_conflict_identifier() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
classdef A
    properties
        x
    end
    methods
        function y = x(obj)
            y = 1;
        end
    end
end
"#;
    let err = match session.compile_input(source) {
        Ok(_) => panic!("class property/method name conflict should fail semantic compilation"),
        Err(err) => err,
    };
    let RunError::Semantic(err) = err else {
        panic!("expected semantic class member name conflict error");
    };
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:ClassMemberNameConflict")
    );
}

#[test]
fn compile_input_reports_cell_assignment_colon_selector_identifier() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let err = match session.compile_input("c = {1,2;3,4}; c{:,2} = 9;") {
        Ok(_) => panic!("brace assignment with colon selector should fail compilation"),
        Err(err) => err,
    };
    let RunError::Compile(err) = err else {
        panic!("expected compile-time cell selector plan error");
    };
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:MirCellIndexPlanInvalid")
    );
}

#[test]
fn compile_input_resolves_wildcard_import_from_dependency_alias() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let dep_root = tmp.path().join("deps/statslib");
    std::fs::create_dir_all(&dep_root).expect("create dependency dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]

[dependencies]
statsdep = { path = "deps/statslib" }
"#,
    )
    .expect("write root manifest");
    std::fs::write(
        dep_root.join("runmat.toml"),
        r#"
[package]
name = "statslib"

[sources]
roots = ["."]
"#,
    )
    .expect("write dependency manifest");
    std::fs::write(
        dep_root.join("summarize.m"),
        "function y = summarize(x); y = x; end",
    )
    .expect("write dependency function");
    std::fs::write(tmp.path().join("main.m"), "x = 1;").expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_name = tmp.path().join("main.m").to_string_lossy().to_string();
    let prepared = session
        .compile_input_for_source_name(&source_name, "import statsdep.*; y = summarize(1);")
        .expect("compile");

    let calls = &prepared.lowering().hir_index.calls;
    assert!(
        calls.iter().any(|call| {
            matches!(
                call.kind,
                runmat_hir::CallKind::PackageFunction(_) | runmat_hir::CallKind::DirectFunction(_)
            )
        }),
        "wildcard import call should resolve through dependency alias symbols from project composition; calls={calls:#?}"
    );
}

#[test]
fn compile_input_resolves_function_handle_from_dependency_alias_wildcard_import() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let dep_root = tmp.path().join("deps/statslib");
    std::fs::create_dir_all(&dep_root).expect("create dependency dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]

[dependencies]
statsdep = { path = "deps/statslib" }
"#,
    )
    .expect("write root manifest");
    std::fs::write(
        dep_root.join("runmat.toml"),
        r#"
[package]
name = "statslib"

[sources]
roots = ["."]
"#,
    )
    .expect("write dependency manifest");
    std::fs::write(
        dep_root.join("summarize.m"),
        "function y = summarize(x); y = x; end",
    )
    .expect("write dependency function");
    std::fs::write(tmp.path().join("main.m"), "x = 1;").expect("write main source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source_name = tmp.path().join("main.m").to_string_lossy().to_string();
    let prepared = session
        .compile_input_for_source_name(&source_name, "import statsdep.*; f = @summarize;")
        .expect("compile");

    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateExternalFunctionHandle(name)
                if name == "statsdep.summarize"
                    || name == "summarize"
                    || name.ends_with(".summarize")
        ) || matches!(
            instr,
            runmat_vm::Instr::CreateFunctionHandle(name)
                if name == "summarize" || name.ends_with(".summarize")
        ) || matches!(
            instr,
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name)
                if name == "summarize" || name.ends_with(".summarize")
        )),
        "wildcard dependency-alias function handle should lower to exact alias-qualified external function-handle bytecode"
    );
}

#[test]
fn compile_input_does_not_leak_local_project_symbols_for_remote_source_names() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+stats")).expect("create package dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("+stats/summarize.m"),
        "function y = summarize(x); y = x; end",
    )
    .expect("write package function");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let prepared = session
        .compile_input_for_source_name("remote:scripts/main.m", "import stats.*; y = summarize(1);")
        .expect("compile");

    assert!(
        !prepared
            .lowering()
            .hir_index
            .calls
            .iter()
            .any(|call| matches!(call.kind, runmat_hir::CallKind::PackageFunction(_))),
        "remote-style source names should not pull local project symbols into wildcard import resolution"
    );
}

#[test]
fn compile_input_does_not_leak_local_project_symbols_for_colon_remote_name() {
    let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("+stats")).expect("create package dir");
    std::fs::write(
        tmp.path().join("runmat.toml"),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        tmp.path().join("+stats/summarize.m"),
        "function y = summarize(x); y = x; end",
    )
    .expect("write package function");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _cwd = push_cwd(tmp.path());
    let prepared = session
        .compile_input_for_source_name("remote:main.m", "import stats.*; y = summarize(1);")
        .expect("compile");

    assert!(
        !prepared
            .lowering()
            .hir_index
            .calls
            .iter()
            .any(|call| matches!(call.kind, runmat_hir::CallKind::PackageFunction(_))),
        "colon-style remote source names should not pull local project symbols into wildcard import resolution"
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::StoreIndexDelete(1))),
        "indexed deletion should lower to explicit delete store bytecode"
    );

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "3");
}

#[test]
fn cell_brace_empty_assignment_is_not_deletion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2}; C{1} = []; y = C{2};";
    let prepared = session
        .compile_input(source)
        .expect("compile cell brace empty assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "cell brace empty assignment should compile through semantic HIR/MIR/VM"
    );
    assert!(
        !prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::StoreIndexCellDelete { .. })),
        "cell brace assignment should not lower to deletion bytecode"
    );

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "2");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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
        prepared.bytecode.instructions.iter().any(|instr| {
            matches!(
                instr,
                runmat_vm::Instr::IndexSlice(..) | runmat_vm::Instr::IndexSliceExpr { .. }
            )
        }),
        "range indexing should lower to slice bytecode"
    );

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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
    let saw_end_numeric_expr = prepared.bytecode.instructions.iter().any(|instr| {
        matches!(
            instr,
            runmat_vm::Instr::IndexSliceExpr {
                end_numeric_exprs, ..
            } if !end_numeric_exprs.is_empty()
        )
    });
    let saw_semantic_call = prepared.bytecode.instructions.iter().any(|instr| {
        matches!(
            instr,
            runmat_vm::Instr::CallSemanticFunctionMulti(_, _, _)
                | runmat_vm::Instr::CallSemanticFunctionExpandMultiOutput(_, _, _)
        )
    });
    assert!(
        (prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::IndexSliceExpr { end_numeric_exprs, .. }
                if end_numeric_exprs
                    .iter()
                    .any(|(_, expr)| end_expr_contains_display_name(expr, "pick"))
        ))) || (saw_end_numeric_expr && saw_semantic_call),
        "end-expression user calls should carry semantic function identity"
    );
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "a")
            && upsert.value.to_string() == "50"
    }));
}

#[test]
fn end_expression_session_function_call_uses_semantic_identity() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction y = pick(n)\n  y = n;\nend",
    )
    .expect("define session function");
    let source = "x = [10 20 30 40 50 60 70 80]; a = x(pick(end-3));";
    let prepared = session
        .compile_input(source)
        .expect("compile session end-expression function call");
    let saw_end_numeric_expr = prepared.bytecode.instructions.iter().any(|instr| {
        matches!(
            instr,
            runmat_vm::Instr::IndexSliceExpr {
                end_numeric_exprs, ..
            } if !end_numeric_exprs.is_empty()
        )
    });
    let saw_semantic_call = prepared.bytecode.instructions.iter().any(|instr| {
        matches!(
            instr,
            runmat_vm::Instr::CallSemanticFunctionMulti(_, _, _)
                | runmat_vm::Instr::CallSemanticFunctionExpandMultiOutput(_, _, _)
        )
    });
    assert!(
        (prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::IndexSliceExpr { end_numeric_exprs, .. }
                if end_numeric_exprs
                    .iter()
                    .any(|(_, expr)| end_expr_contains_display_name(expr, "pick"))
        ))) || (saw_end_numeric_expr && saw_semantic_call),
        "session end-expression user calls should carry semantic function identity"
    );
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "a")
            && upsert.value.to_string() == "50"
    }));
}

#[test]
fn for_range_loop_uses_semantic_vm_without_rerunning_prefix() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(&mut session, "prefix = 0;").expect("seed prefix");
    let source = "prefix = prefix + 1; s = 0; for i = 1:3; s = s + i; end; y = s + prefix;";
    let prepared = session.compile_input(source).expect("compile for loop");
    assert!(
        prepared.bytecode.layout.is_some(),
        "for range loops should compile through semantic HIR/MIR/VM"
    );

    execute_text_request(&mut session, source).expect("exec succeeds");
    let y_outcome = execute_text_request(&mut session, "y").expect("read y");
    let y = y_outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(y.to_string(), "7");
    let s_outcome = execute_text_request(&mut session, "s").expect("read s");
    let s = s_outcome
        .flow
        .durable_workspace_value()
        .expect("s should be readable from workspace");
    assert_eq!(s.to_string(), "6");
    let prefix_outcome = execute_text_request(&mut session, "prefix").expect("read prefix");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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
        prepared.bytecode.instructions.iter().any(|instr| {
            matches!(
                instr,
                runmat_vm::Instr::StoreSlice(..) | runmat_vm::Instr::StoreSliceExpr { .. }
            )
        }),
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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
        prepared.bytecode.instructions.iter().any(|instr| {
            matches!(
                instr,
                runmat_vm::Instr::StoreSlice(..) | runmat_vm::Instr::StoreSliceExpr { .. }
            )
        }),
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "9");
}

#[test]
fn range_deletion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [1, 2, 3, 4]; A(2:3) = []; y = A(2);";
    let prepared = session
        .compile_input(source)
        .expect("compile range deletion");
    assert!(
        prepared.bytecode.layout.is_some(),
        "range deletion should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| {
            matches!(
                instr,
                runmat_vm::Instr::StoreSliceDelete(..)
                    | runmat_vm::Instr::StoreSliceExprDelete { .. }
            )
        }),
        "range deletion should lower to explicit slice deletion bytecode"
    );

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "4");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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
        prepared.bytecode.instructions.iter().any(|instr| {
            matches!(
                instr,
                runmat_vm::Instr::StoreSlice(..) | runmat_vm::Instr::StoreSliceExpr { .. }
            )
        }),
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| {
            matches!(
                instr,
                runmat_vm::Instr::StoreSliceDelete(..)
                    | runmat_vm::Instr::StoreSliceExprDelete { .. }
            )
        }),
        "cell range deletion should lower to explicit slice deletion bytecode"
    );

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "9");
}

#[test]
fn workspace_read_across_submissions_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(&mut session, "x = 42;").expect("seed workspace");

    let prepared = session.compile_input("x").expect("compile workspace read");
    assert!(
        prepared.bytecode.layout.is_some(),
        "workspace read should compile through semantic HIR/MIR/VM"
    );

    let outcome = execute_text_request(&mut session, "x").expect("read workspace value");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("workspace read should return a durable value");
    assert_eq!(value.to_string(), "42");
}

#[test]
fn dynamic_workspace_eval_mutates_and_reads_current_workspace() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        eval('dyn_x = 41;');
        dyn_y = eval('dyn_x + 1');
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "dyn_x",
        &runmat_builtins::Value::Num(41.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "dyn_y",
        &runmat_builtins::Value::Num(42.0)
    ));

    let outcome = execute_text_request(&mut session, "dyn_x + dyn_y").expect("read workspace");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("dynamic workspace variables should persist");
    assert_eq!(value.to_string(), "83");
}

#[test]
fn dynamic_workspace_evalin_base_and_assignin_update_base_workspace() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        assignin('base', 'base_x', 7);
        base_y = evalin('base', 'base_x + 1');
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "base_x",
        &runmat_builtins::Value::Num(7.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "base_y",
        &runmat_builtins::Value::Num(8.0)
    ));

    let outcome = execute_text_request(&mut session, "base_x + base_y").expect("read workspace");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("base workspace variables should persist");
    assert_eq!(value.to_string(), "15");
}

#[test]
fn dynamic_workspace_evalin_caller_from_function_targets_script_workspace() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        outer_x = 5;
        y = helper();

        function out = helper()
            out = evalin('caller', 'outer_x + 2');
            assignin('caller', 'caller_z', 11);
        end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "y",
        &runmat_builtins::Value::Num(7.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "caller_z",
        &runmat_builtins::Value::Num(11.0)
    ));

    let outcome = execute_text_request(&mut session, "caller_z + y").expect("read workspace");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("caller-assigned workspace variables should persist");
    assert_eq!(value.to_string(), "18");
}

#[test]
fn dynamic_workspace_evalin_caller_from_nested_function_targets_parent_function_workspace() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        outer_y = outer();

        function y = outer()
            local_x = 6;
            y = nested();
            y = y + eval('nested_assigned');

            function out = nested()
                out = evalin('caller', 'local_x + 3');
                assignin('caller', 'nested_assigned', 14);
            end
        end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "outer_y",
        &runmat_builtins::Value::Num(23.0)
    ));
}

#[test]
fn dynamic_workspace_evalin_base_and_assignin_base_work_from_path_source_function() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let source_path = tmp.path().join("dynamic_workspace_path.m");
    std::fs::write(
        &source_path,
        r#"
        path_seed = 2;
        path_result = path_helper();

        function out = path_helper()
            assignin('base', 'path_base', 19);
            out = evalin('base', 'path_seed + path_base');
        end
    "#,
    )
    .expect("write dynamic workspace path source");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute path source");
    assert!(outcome_has_named_upsert(
        &outcome,
        "path_base",
        &runmat_builtins::Value::Num(19.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "path_result",
        &runmat_builtins::Value::Num(21.0)
    ));
}

#[test]
fn dynamic_workspace_eval_resolves_active_registry_functions() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let source_path = tmp.path().join("dynamic_workspace_registry.m");
    let private_dir = tmp.path().join("private");
    let package_dir = tmp.path().join("+pkg");
    std::fs::create_dir_all(&private_dir).expect("create private dir");
    std::fs::create_dir_all(&package_dir).expect("create package dir");
    std::fs::write(
        &source_path,
        r#"
        seed = 3;
        local_value = eval('local_helper(seed)');
        sibling_value = eval('sibling_helper(5)');
        private_value = eval('private_helper(7)');
        package_value = eval('pkg.package_helper(9)');

        function out = local_helper(x)
            out = x + 10;
        end
    "#,
    )
    .expect("write dynamic workspace registry source");
    std::fs::write(
        tmp.path().join("sibling_helper.m"),
        "function out = sibling_helper(x)\n  out = x + 20;\nend\n",
    )
    .expect("write sibling helper");
    std::fs::write(
        private_dir.join("private_helper.m"),
        "function out = private_helper(x)\n  out = x + 30;\nend\n",
    )
    .expect("write private helper");
    std::fs::write(
        package_dir.join("package_helper.m"),
        "function out = package_helper(x)\n  out = x + 40;\nend\n",
    )
    .expect("write package helper");

    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("execute path source");
    assert!(outcome_has_named_upsert(
        &outcome,
        "local_value",
        &runmat_builtins::Value::Num(13.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "sibling_value",
        &runmat_builtins::Value::Num(25.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "private_value",
        &runmat_builtins::Value::Num(37.0)
    ));
    assert!(outcome_has_named_upsert(
        &outcome,
        "package_value",
        &runmat_builtins::Value::Num(49.0)
    ));
}

#[test]
fn dynamic_workspace_eval_does_not_discover_files_outside_active_registry() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let source_dir = tmp.path().join("src");
    let hidden_dir = tmp.path().join("hidden");
    std::fs::create_dir_all(&source_dir).expect("create source dir");
    std::fs::create_dir_all(&hidden_dir).expect("create hidden dir");
    let source_path = source_dir.join("dynamic_workspace_boundary.m");
    std::fs::write(
        &source_path,
        r#"
        try
            hidden_value = eval('hidden_helper(1)');
            err = "NOERR";
        catch e
            err = e.identifier;
        end
    "#,
    )
    .expect("write source");
    std::fs::write(
        hidden_dir.join("hidden_helper.m"),
        "function out = hidden_helper(x)\n  out = x + 1;\nend\n",
    )
    .expect("write hidden helper");
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = execute_path_request(&mut session, source_path.to_string_lossy().as_ref())
        .expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "err",
        &runmat_builtins::Value::String("RunMat:UndefinedFunction".into())
    ));
    assert!(
        !outcome_has_upsert_name(&outcome, "hidden_value"),
        "unregistered helper should not be discovered from eval text"
    );
}

#[test]
fn dynamic_workspace_execute_request_can_disable_dynamic_eval_host_policy() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = block_on(session.execute_request(abi::ExecutionRequest {
        source: abi::SourceInput::Text {
            name: "dynamic-eval-policy.m".to_string(),
            text: "eval('policy_x = 1');".to_string(),
        },
        compatibility: CompatMode::Matlab,
        host_policy: abi::HostExecutionPolicy {
            top_level_await: true,
            dynamic_eval: false,
        },
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace: session.workspace_handle(),
    }))
    .result
    .expect("request should return an outcome with a policy diagnostic");
    assert_eq!(outcome.diagnostics.len(), 1);
    assert_eq!(outcome.diagnostics[0].code, "RunMat:DynamicEvalDisabled");
    assert!(
        !outcome_has_upsert_name(&outcome, "policy_x"),
        "disabled eval must not mutate the workspace"
    );
}

#[test]
fn dynamic_workspace_execute_request_dynamic_eval_policy_does_not_block_assignin() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = block_on(session.execute_request(abi::ExecutionRequest {
        source: abi::SourceInput::Text {
            name: "dynamic-eval-policy-assignin.m".to_string(),
            text: "assignin('base', 'policy_assign', 12);".to_string(),
        },
        compatibility: CompatMode::Matlab,
        host_policy: abi::HostExecutionPolicy {
            top_level_await: true,
            dynamic_eval: false,
        },
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        workspace: session.workspace_handle(),
    }))
    .result
    .expect("assignin should remain available");
    assert!(outcome.diagnostics.is_empty());
    assert!(outcome_has_named_upsert(
        &outcome,
        "policy_assign",
        &runmat_builtins::Value::Num(12.0)
    ));
}

#[test]
fn dynamic_workspace_evalin_invalid_selector_is_catchable() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = r#"
        try
            evalin('workspace', '1');
            err = "BAD";
        catch e
            err = e.identifier;
        end
    "#;
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome_has_named_upsert(
        &outcome,
        "err",
        &runmat_builtins::Value::String("RunMat:DynamicWorkspaceSelector".into())
    ));
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

    let outcome = execute_text_request(&mut session, "w = 'hello';").expect("exec succeeds");
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
    let outcome = execute_text_request(&mut session, "disp('alpha')").expect("exec disp");
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
    let outcome = execute_text_request(&mut session, "fprintf('foo')").expect("exec fprintf");
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

    let outcome = execute_text_request(&mut session, "sin(0)").expect("exec sin");
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

    let outcome = execute_text_request(&mut session, "[a, b] = deal(1, 2)").expect("exec succeeds");
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
            runmat_vm::Instr::CallBuiltinMulti(name, _, _)
                if matches!(name.as_str(), "and" | "or" | "not")
        )),
        "semantic logical ops should not lower through string-keyed builtin calls"
    );
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn builtin_call_with_paren_index_argument_does_not_infer_expansion() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "T = [1, 2; 3, 4]; y = max(T(:));";
    let prepared = session
        .compile_input(source)
        .expect("compile builtin call with paren-index argument");
    assert!(
        !prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalMulti(_, _))),
        "paren indexing on a bound value must not be rewritten to dynamic feval"
    );

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "4");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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
    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "a").expect("read a");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "a").expect("read a");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "9"
    }));
}

#[test]
fn cell_end_offset_range_paren_assignment_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2, 3, 4}; C(1:end-1) = {9, 8, 7}; y = C{3};";
    let prepared = session
        .compile_input(source)
        .expect("compile cell end-offset range paren assignment");
    assert!(
        prepared.bytecode.layout.is_some(),
        "cell end-offset range paren assignment should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::StoreSliceExpr { .. })),
        "cell end-offset range paren assignment should lower to expression slice store bytecode"
    );

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "7");
}

#[test]
fn cell_end_offset_range_paren_deletion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "C = {1, 2, 3, 4}; C(1:end-1) = []; y = C{1};";
    let prepared = session
        .compile_input(source)
        .expect("compile cell end-offset range paren deletion");
    assert!(
        prepared.bytecode.layout.is_some(),
        "cell end-offset range paren deletion should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::StoreSliceExprDelete { .. })),
        "cell end-offset range paren deletion should lower to expression slice deletion bytecode"
    );

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "4");
}

#[test]
fn end_offset_range_deletion_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "A = [1, 2, 3, 4]; A(1:end-1) = []; y = A(1);";
    let prepared = session
        .compile_input(source)
        .expect("compile end-offset range deletion");
    assert!(
        prepared.bytecode.layout.is_some(),
        "end-offset range deletion should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::StoreSliceExprDelete { .. })),
        "end-offset range deletion should lower to expression slice deletion bytecode"
    );

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "4");
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
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalMulti(1, 1))),
        "feval should lower from dynamic MIR callee to the VM feval ABI instruction"
    );
    assert!(
        !prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallBuiltinMulti(name, _, _) if name == "feval"
        )),
        "feval should not lower as a string-keyed builtin call"
    );

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name) if name == "inc"
        )),
        "local feval string callee should lower to a semantic function handle"
    );
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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
            runmat_vm::Instr::CallBuiltinMulti(name, 1, _) if name == "make_handle"
        )),
        "function handle literals should not lower through the internal make_handle builtin"
    );

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "0");
}

#[test]
fn dynamic_function_handle_tensor_arg_uses_semantic_vm() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "f = @sum; y = f([1, 2, 3]);";
    let prepared = session
        .compile_input(source)
        .expect("compile dynamic function handle tensor-arg call");
    assert!(
        prepared.bytecode.layout.is_some(),
        "dynamic function handle tensor-arg call should compile through semantic HIR/MIR/VM"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateFunctionHandle(name) if name == "sum"
        )),
        "builtin function handle literal should lower to typed function-handle bytecode"
    );

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
    let value = outcome
        .flow
        .durable_workspace_value()
        .expect("y should be readable from workspace");
    assert_eq!(value.to_string(), "6");
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
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name) if name == "inc"
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
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    execute_text_request(&mut session, source).expect("exec succeeds");
    let outcome = execute_text_request(&mut session, "y").expect("read y");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalMulti(_, 2))),
        "dynamic multi-output function handle call should lower to typed feval multi-output bytecode"
    );

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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
    let source = "f = @pair; C = {2}; [a, b] = feval(f, C{:});\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend";
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    let y_message = outcome
        .workspace_delta
        .upserts
        .iter()
        .find_map(|upsert| match &upsert.key {
            abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y" => {
                Some(upsert.value.to_string())
            }
            _ => None,
        })
        .expect("try/catch binding should materialize y");
    assert!(
        y_message == "'boom'",
        "unexpected catch message payload: {y_message:?}"
    );
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
        prepared.bytecode.instructions.iter().any(|instr| {
            matches!(
                instr,
                runmat_vm::Instr::StoreSlice(..) | runmat_vm::Instr::StoreSliceExpr { .. }
            )
        }),
        "indexed member slice assignment should lower to typed slice store bytecode"
    );

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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
            runmat_vm::Instr::CallMethodOrMemberIndexMulti {
                identity,
                arg_count: 1,
                out_count: 1,
                ..
            } if identity.display_name().as_deref() == Some("a")
        )),
        "dotted member index call should lower to typed method/member-index bytecode"
    );

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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
            runmat_vm::Instr::CallMethodOrMemberIndexExpandMultiOutput { identity, out_count: 1, .. }
                if identity.display_name().as_deref() == Some("a")
        )),
        "expanded dotted member index call should lower to typed expansion bytecode"
    );

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalExpandMultiOutput(_, 1))),
        "expanded feval should use the VM feval ABI instruction"
    );
    assert!(
        !prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallBuiltinMulti(name, _, _) if name == "feval"
        )),
        "expanded feval should not lower as a string-keyed builtin call"
    );

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalExpandMultiOutput(_, 1))),
        "2d expanded feval should use the VM feval ABI instruction"
    );

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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
            .function_registry
            .resolve_name("inc")
            .is_some(),
        "local callback target should be available as semantic function bytecode"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name) if name == "inc"
        )),
        "local string callback should be bound to a semantic function handle"
    );
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn cellfun_runtime_string_callback_uses_semantic_resolver() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "name = 'inc'; C = {2}; B = cellfun(name, C); y = B(1);\nfunction z = inc(x)\n  z = x + 1;\nend";
    let prepared = session
        .compile_input(source)
        .expect("compile runtime string callback");
    assert!(
        !prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name) if name == "inc"
        )),
        "runtime string callback variables should not be compile-time literal rewrites"
    );
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn cellfun_session_function_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction z = inc(x)\n  z = x + 1;\nend",
    )
    .expect("define session function");

    let source = "C = {2}; B = cellfun('inc', C); y = B(1);";
    let prepared = session
        .compile_input(source)
        .expect("compile callback using session function");
    let inc_id = prepared
        .bytecode
        .function_registry
        .resolve_name("inc")
        .expect("session callback target should resolve through semantic function registry");
    assert!(
        prepared
            .bytecode
            .function_registry
            .get(inc_id)
            .and_then(|function| function.source_id)
            .is_some(),
        "session semantic callback target should retain source metadata"
    );
    let inc_source = prepared
        .bytecode
        .function_registry
        .get(inc_id)
        .and_then(|function| function.source_id)
        .expect("inc source metadata");
    assert!(
        prepared
            .bytecode
            .function_registry
            .functions_for_source(inc_source)
            .contains(&inc_id),
        "session semantic callback target should be indexed by source ownership"
    );
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name) if name == "inc"
        )),
        "session string callback should be bound to a semantic function handle"
    );
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name) if name == "inc"
        )),
        "local arrayfun string callback should be bound to a semantic function handle"
    );
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "4"
    }));
}

#[test]
fn arrayfun_session_function_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction z = inc(x)\n  z = x + 1;\nend",
    )
    .expect("define session function");

    let source = "A = [2, 3]; B = arrayfun('inc', A); y = B(2);";
    let prepared = session
        .compile_input(source)
        .expect("compile session arrayfun callback");
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name) if name == "inc"
        )),
        "session arrayfun string callback should be bound to a semantic function handle"
    );
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "4"
    }));
}

#[test]
fn arrayfun_runtime_string_callback_uses_semantic_resolver() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let source = "name = 'inc'; A = [2, 3]; B = arrayfun(name, A); y = B(2);\nfunction z = inc(x)\n  z = x + 1;\nend";
    let prepared = session
        .compile_input(source)
        .expect("compile runtime arrayfun string callback");
    assert!(
        !prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name) if name == "inc"
        )),
        "runtime arrayfun string callback variables should not be compile-time literal rewrites"
    );
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "4"
    }));
}

#[test]
fn cellfun_unresolved_external_callback_reports_undefined_function_identifier() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome = execute_text_request(&mut session, "C = {2}; y = cellfun('pkg.callback', C);")
        .expect("unresolved external cellfun callback should surface a runtime diagnostic");
    assert!(outcome.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RunMat:UndefinedFunction"
            && matches!(diagnostic.severity, abi::DiagnosticSeverity::Error)
    }));
}

#[test]
fn arrayfun_unresolved_external_callback_reports_undefined_function_identifier() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let outcome =
        execute_text_request(&mut session, "A = [2, 3]; y = arrayfun('pkg.callback', A);")
            .expect("unresolved external arrayfun callback should surface a runtime diagnostic");
    assert!(outcome.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RunMat:UndefinedFunction"
            && matches!(diagnostic.severity, abi::DiagnosticSeverity::Error)
    }));
}

#[test]
fn direct_session_function_call_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction z = inc(x)\n  z = x + 1;\nend",
    )
    .expect("define session function");

    let prepared = session
        .compile_input("y = inc(2);")
        .expect("compile direct session function call");
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallSemanticFunctionMulti(_, 1, 1))),
        "direct call should lower to semantic function bytecode"
    );
    let outcome = execute_text_request(&mut session, "y = inc(2);").expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn direct_session_function_multi_output_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend",
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
    let outcome = execute_text_request(&mut session, "[a, b] = pair(2);").expect("exec succeeds");
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
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction z = inc(x)\n  z = x + 1;\nend",
    )
    .expect("define session function");

    let prepared = session
        .compile_input("C = {2}; y = inc(C{:});")
        .expect("compile direct session expansion function call");
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallSemanticFunctionExpandMultiOutput(_, _, 1)
        )),
        "direct expansion call should lower to semantic function bytecode"
    );
    let outcome =
        execute_text_request(&mut session, "C = {2}; y = inc(C{:});").expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn direct_session_function_expansion_multi_output_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend",
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
    let outcome =
        execute_text_request(&mut session, "C = {2}; [a, b] = pair(C{:});").expect("exec succeeds");
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
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction z = inc(x)\n  z = x + 1;\nend",
    )
    .expect("define session function");

    let prepared = session
        .compile_input("f = @inc; y = f(2);")
        .expect("compile function handle session call");
    assert!(
        prepared
            .bytecode
            .function_registry
            .resolve_name("inc")
            .is_some(),
        "function handle target should be present in semantic registry"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CreateBoundFunctionHandle(_, _))),
        "session function handles should carry semantic identity"
    );
    assert!(
        !prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateFunctionHandle(name) if name == "inc"
        )),
        "session function handles should not remain name-only handles"
    );
    let outcome = execute_text_request(&mut session, "f = @inc; y = f(2);")
        .expect("function handle call succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "3"
    }));
}

#[test]
fn session_function_handle_feval_multi_output_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend",
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
    let outcome = execute_text_request(&mut session, "f = @pair; [a, b] = feval(f, 2);")
        .expect("exec succeeds");
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
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend",
    )
    .expect("define session function");

    let prepared = session
        .compile_input("[a, b] = feval('pair', 2);")
        .expect("compile feval string multi-output session call");
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name) if name == "pair"
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
    let outcome =
        execute_text_request(&mut session, "[a, b] = feval('pair', 2);").expect("exec succeeds");
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
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction z = add2(a, b)\n  z = a + b;\nend",
    )
    .expect("define session function");

    let prepared = session
        .compile_input("f = @add2; C = {2, 3}; y = feval(f, C{:});")
        .expect("compile feval expansion session handle call");
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalExpandMultiOutput(_, 1))),
        "function handle feval expansion call should use typed feval expansion bytecode"
    );
    let outcome = execute_text_request(&mut session, "f = @add2; C = {2, 3}; y = feval(f, C{:});")
        .expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "5"
    }));
}

#[test]
fn session_feval_string_expansion_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction z = add2(a, b)\n  z = a + b;\nend",
    )
    .expect("define session function");

    let prepared = session
        .compile_input("C = {2, 3}; y = feval('add2', C{:});")
        .expect("compile feval string expansion session call");
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name) if name == "add2"
        )),
        "session feval string expansion callee should carry semantic identity"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalExpandMultiOutput(_, 1))),
        "session feval string expansion call should use typed feval expansion bytecode"
    );
    let outcome = execute_text_request(&mut session, "C = {2, 3}; y = feval('add2', C{:});")
        .expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "5"
    }));
}

#[test]
fn session_function_handle_feval_expansion_multi_output_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend",
    )
    .expect("define session function");

    let prepared = session
        .compile_input("f = @pair; C = {2}; [a, b] = feval(f, C{:});")
        .expect("compile feval expansion multi-output session handle call");
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalExpandMultiOutput(_, 2))),
        "function handle feval expansion multi-output call should use typed feval expansion bytecode"
    );
    let outcome =
        execute_text_request(&mut session, "f = @pair; C = {2}; [a, b] = feval(f, C{:});")
            .expect("exec succeeds");
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
fn session_feval_string_expansion_multi_output_uses_semantic_registry() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction [x, y] = pair(n)\n  x = n;\n  y = n + 1;\nend",
    )
    .expect("define session function");

    let prepared = session
        .compile_input("C = {2}; [a, b] = feval('pair', C{:});")
        .expect("compile feval string expansion multi-output session call");
    assert!(
        prepared.bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name) if name == "pair"
        )),
        "session feval string expansion multi-output callee should carry semantic identity"
    );
    assert!(
        prepared
            .bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalExpandMultiOutput(_, 2))),
        "session feval string expansion multi-output call should use typed feval expansion bytecode"
    );
    let outcome = execute_text_request(&mut session, "C = {2}; [a, b] = feval('pair', C{:});")
        .expect("exec succeeds");
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
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction z = inc(x)\n  z = x + 1;\nend",
    )
    .expect("define first function");
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction z = inc(x)\n  z = x + 10;\nend",
    )
    .expect("redefine function");

    let source = "C = {2}; B = cellfun('inc', C); y = B(1);";
    let prepared = session
        .compile_input(source)
        .expect("compile callback using redefined session function");
    assert_eq!(
        prepared
            .bytecode
            .function_registry
            .functions
            .values()
            .filter(|function| function.display_name == "inc")
            .count(),
        1,
        "semantic registry should retire the old definition"
    );
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "12"
    }));
}

#[test]
fn session_semantic_registry_retires_replaced_source_group() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction z = inc(x)\n  z = x + 1;\nend\nfunction z = dec(x)\n  z = x - 1;\nend",
    )
    .expect("define source function group");
    execute_text_request(
        &mut session,
        "seed = 0;\nfunction z = inc(x)\n  z = x + 10;\nend",
    )
    .expect("replace one function from group");

    let prepared = session
        .compile_input("y = inc(2);")
        .expect("compile using replacement function");
    assert!(
        prepared
            .bytecode
            .function_registry
            .resolve_name("inc")
            .is_some(),
        "replacement function should remain in semantic registry"
    );
    assert!(
        prepared
            .bytecode
            .function_registry
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
    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
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

    let outcome = execute_text_request(&mut session, source).expect("exec succeeds");
    assert!(outcome.workspace_delta.upserts.iter().any(|upsert| {
        matches!(&upsert.key, abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "y")
            && upsert.value.to_string() == "5"
    }));
}

#[test]
fn workspace_reports_datetime_array_shape() {
    let mut session = RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    execute_text_request(
        &mut session,
        "d = datetime([739351; 739352], 'ConvertFrom', 'datenum');",
    )
    .expect("exec succeeds");
    let entry = block_on(session.materialize_variable(
        WorkspaceMaterializeTarget::Name("d".to_string()),
        WorkspaceMaterializeOptions::default(),
    ))
    .expect("workspace entry for d");
    assert_eq!(entry.class_name, "datetime");
    assert_eq!(entry.shape, vec![2, 1]);
}

#[test]
fn workspace_state_roundtrip_replace_only() {
    let mut source_session =
        RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _ =
        execute_text_request(&mut source_session, "x = 42; y = [1, 2, 3];").expect("exec succeeds");

    let bytes = block_on(source_session.export_workspace_state(WorkspaceExportMode::Force))
        .expect("workspace export")
        .expect("workspace bytes");

    let mut restore_session =
        RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
    let _ = execute_text_request(&mut restore_session, "z = 99;").expect("seed workspace");
    restore_session
        .import_workspace_state(&bytes)
        .expect("workspace import");

    let _restored =
        execute_text_request(&mut restore_session, "r = x + y(2);").expect("post-import exec");

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
    match z {
        Ok(value) => panic!(
            "replace-only import should drop stale z variable, but materialized: {:?}",
            value.name
        ),
        Err(err) => assert_eq!(err.to_string(), "Variable 'z' not found in workspace"),
    }
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
