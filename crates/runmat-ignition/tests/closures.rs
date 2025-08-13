use runmat_hir::lower;
use runmat_ignition::execute;
use runmat_parser::parse;

#[test]
fn closure_simple_no_capture() {
    let ast = parse("f = @(x) x + 1; y = feval(f, 2);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 3.0).abs() < 1e-9)));
}

#[test]
fn closure_captures_free_variables() {
    let ast = parse("a=1; b=2; f=@(x) x + a + b; y = feval(f, 3);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 6.0).abs() < 1e-9)));
}

#[test]
fn nested_closures_capture_outer() {
    let ast = parse("a=10; f=@(x) @(y) (x + y + a); g = feval(f, 2); r = feval(g, 3);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // Expect 2 + 3 + 10 = 15
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 15.0).abs() < 1e-9)));
}

#[test]
fn feval_with_string_handle() {
    let ast = parse("r = feval('@max', 2, 5);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}


