use runmat_hir::lower;
use runmat_ignition::execute;
use runmat_parser::parse;

#[test]
fn error_identifier_and_catch() {
    let ast = parse("try; error('MATLAB:domainError', 'bad'); catch e; msg = getfield(e, 'message'); id = getfield(e, 'identifier'); end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::String(s) if s.contains("bad"))));
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


