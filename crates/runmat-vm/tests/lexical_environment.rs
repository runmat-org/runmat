#[path = "support/mod.rs"]
mod support;

fn execute(source: &str) -> runmat_runtime::execution::RootedValues {
    let program = support::compile_source(source).expect("compile nested storage source");
    support::interpret(&program).expect("execute nested storage source")
}

fn has_number(values: &[runmat_value::Value], expected: f64) -> bool {
    values.iter().any(|value| {
        matches!(value, runmat_value::Value::Num(number) if (*number - expected).abs() < 1e-9)
    })
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn nested_child_uses_canonical_global_and_persistent_storage() {
    let values = execute(
        r#"
        result = outer();
        function y = outer()
            global G;
            persistent P;
            G = 1;
            P = 1;
            h = @child;
            first = h();
            G = G + 1;
            P = P + 1;
            second = h();
            y = first * 10 + second;
            function z = child()
                G = G + 1;
                P = P + 1;
                z = G + P;
            end
        end
        "#,
    );
    // Both the child and its parent must observe the same canonical stores:
    // the first call returns 4, then the parent mutates both stores, and the
    // second child call returns 8. A captured snapshot cannot produce 48.
    assert!(has_number(&values, 48.0), "unexpected values: {values:?}");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn anonymous_capture_remains_a_snapshot_when_nested_child_is_live() {
    let values = execute(
        r#"
        result = outer();
        function y = outer()
            shared = 1;
            f = @() shared + reader();
            shared = 2;
            y = f();
            function z = reader()
                z = shared;
            end
        end
        "#,
    );
    assert!(has_number(&values, 3.0), "unexpected values: {values:?}");
}
