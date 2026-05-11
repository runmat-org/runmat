#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_semantic_source;

#[test]
fn global_across_functions() {
    let program = r#"
        function y = setg(x)
            global g;
            g = x;
            y = 0; % ensure one output to satisfy current CallFunction semantics
        end
        function y = getg()
            global g;
            y = g;
        end
        setg(42);
        r = getg();
    "#;
    let vars = execute_semantic_source(program).unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 42.0).abs() < 1e-9)));
}

#[test]
fn persistent_across_calls() {
    let program = r#"
        function y = counter()
            persistent p;
            p = p + 1;
            y = p;
        end
        a = counter();
        b = counter();
    "#;
    let vars = execute_semantic_source(program).unwrap();
    // Expect to see both 1 and 2 somewhere in the variable array
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs() < 1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 2.0).abs() < 1e-9)));
}
