use runmat_hir::lower;
use runmat_ignition::execute;
use runmat_parser::parse;

#[test]
fn error_identifier_and_catch() {
    // Produce the message as the final expression deterministically; use double-quoted strings for identifier/message
    let ast = parse("try; error(\"MATLAB:domainError\", \"bad\"); catch e; string(getfield(e, 'message')); end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // The last result may be either the message or the MException depending on path; handle both
    match vars.last().cloned().unwrap() {
        runmat_builtins::Value::StringArray(sa) => {
            let got = &sa.data[0];
            assert!(got.contains("bad"));
        }
        runmat_builtins::Value::MException(me) => {
            assert!(me.message.contains("bad"));
        }
        other => panic!("unexpected last value: {:?}", other),
    }
}

#[test]
fn nested_try_catch_rethrow() {
    let ast = parse("try; try; error('MATLAB:oops','x'); catch e; rethrow(e); end; catch f; g=1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs()<1e-9)));
}

#[test]
fn catch_index_error_and_continue() {
    // Index out of bounds on tensor, caught by try/catch
    let ast = parse("A = [1 2;3 4]; try; x = A(10); catch e; msg = getfield(e,'message'); ok=1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs()<1e-9)));
}


