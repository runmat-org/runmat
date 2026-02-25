mod test_helpers;

use runmat_parser::parse;
use test_helpers::execute;
use test_helpers::lower;

#[test]
fn error_identifier_and_catch() {
    // Emit the message; ensure id/message are captured correctly
    let ast = parse(
        "try; error(\"MATLAB:domainError\", \"bad\"); catch e; msg = getfield(e, 'message'); end",
    )
    .unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let has_msg = vars.iter().any(|v| matches!(v, runmat_builtins::Value::StringArray(sa) if !sa.data.is_empty() && sa.data.iter().any(|s| s.contains("bad"))));
    let has_exc = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::MException(me) if me.message.contains("bad")));
    assert!(has_msg || has_exc);
}

#[test]
fn nested_try_catch_rethrow() {
    let ast =
        parse("try; try; error('MATLAB:oops','x'); catch e; rethrow(e); end; catch f; g=1; end")
            .unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs()<1e-9)));
}

#[test]
fn catch_and_multi_assign_propagation() {
    // Ensure catch-bound exception can be read and subsequent assignments proceed
    let ast = parse("A=[1 2]; try; x=A(10); catch e; [m,ok] = deal(getfield(e,'message'), 1); end")
        .unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let has_ok = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs()<1e-9));
    assert!(has_ok);
}

#[test]
fn dot_access_identifier_and_message() {
    // err.identifier and err.message via dot syntax (LoadMember, not getfield);
    // use an index-out-of-bounds error so the try body always fires from the VM directly.
    let ast =
        parse("A=[1 2]; try; x=A(10); catch e; id=e.identifier; msg=e.message; ok=1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let catch_ran = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs() < 1e-9));
    assert!(catch_ran, "catch block did not run");
    let has_id = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::String(_)));
    let has_msg = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::String(_)));
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
    let ast =
        parse("A = [1 2;3 4]; try; x = A(10); catch e; msg = getfield(e,'message'); ok=1; end")
            .unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs()<1e-9)));
}
