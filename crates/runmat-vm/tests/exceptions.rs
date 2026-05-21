#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_semantic_source;

#[test]
fn error_identifier_and_catch() {
    // Emit the message and ensure identifier/message are preserved exactly.
    let vars = execute_semantic_source(
        "try; error(\"RunMat:domainError\", \"bad\"); catch e; id = getfield(e, 'identifier'); msg = getfield(e, 'message'); out_exc = e; end",
    )
    .unwrap();
    let has_id = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::String(s) if s == "RunMat:domainError"));
    let has_msg = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::String(s) if s == "bad"));
    let has_exc = vars.iter().any(|v| {
        matches!(
            v,
            runmat_builtins::Value::MException(me)
                if me.identifier == "RunMat:domainError" && me.message == "bad"
        )
    });
    assert!(has_exc || (has_id && has_msg));
}

#[test]
fn nested_try_catch_rethrow() {
    let vars = execute_semantic_source(
        "try; try; error('RunMat:oops','x'); catch e; rethrow(e); end; catch f; g=1; end",
    )
    .unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs()<1e-9)));
}

#[test]
fn catch_and_multi_assign_propagation() {
    // Ensure catch-bound exception can be read and subsequent assignments proceed
    let vars = execute_semantic_source(
        "A=[1 2]; try; x=A(10); catch e; [m,id,ok] = deal(getfield(e,'message'), getfield(e,'identifier'), 1); end",
    )
    .unwrap();
    let has_ok = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs()<1e-9));
    let has_id = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::String(s) if s == "RunMat:IndexOutOfBounds"));
    assert!(has_ok && has_id);
}

#[test]
fn dot_access_identifier_and_message() {
    // err.identifier and err.message via dot syntax (LoadMember, not getfield);
    // use an index-out-of-bounds error so the try body always fires from the VM directly.
    let vars = execute_semantic_source(
        "A=[1 2]; try; x=A(10); catch e; id=e.identifier; msg=e.message; ok=1; end",
    )
    .unwrap();
    let catch_ran = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs() < 1e-9));
    assert!(catch_ran, "catch block did not run");
    let has_id = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::String(s) if s == "RunMat:IndexOutOfBounds"));
    let has_msg = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::String(s) if !s.is_empty() && s != "RunMat:IndexOutOfBounds"));
    assert!(
        has_id,
        "err.identifier did not produce a String via dot access"
    );
    assert!(
        has_msg,
        "err.message did not produce a String via dot access"
    );
}

#[test]
fn catch_index_error_and_continue() {
    // Index out of bounds on tensor, caught by try/catch
    let vars = execute_semantic_source(
        "A = [1 2;3 4]; try; x = A(10); catch e; id = getfield(e,'identifier'); msg = getfield(e,'message'); ok=1; end",
    )
    .unwrap();
    let has_ok = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs()<1e-9));
    let has_id = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::String(s) if s == "RunMat:IndexOutOfBounds"));
    assert!(has_ok && has_id);
}
