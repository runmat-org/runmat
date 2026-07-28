use runmat_builtins::{IntValue, LogicalArray, Tensor, Value};

#[path = "support/mod.rs"]
mod test_helpers;
use test_helpers::execute_source;

#[test]
fn tensor_scalar_gt_mask_feeds_elementwise_multiply() {
    let vars = execute_source(
        r#"
        Z1 = [-1 2; 3 -4];
        dA1 = [10 20; 30 40];
        mask = Z1 > 0;
        dZ1 = dA1 .* mask;
        "#,
    )
    .expect("tensor-scalar greater-than mask should execute");

    assert!(vars.iter().any(|value| matches!(
        value,
        Value::LogicalArray(LogicalArray { data, shape })
            if shape == &vec![2, 2] && data == &vec![0, 1, 1, 0]
    )));
    assert!(vars.iter().any(|value| matches!(
        value,
        Value::Tensor(Tensor { data, shape, .. })
            if shape == &vec![2, 2] && data == &vec![0.0, 30.0, 20.0, 0.0]
    )));
}

#[test]
fn uint64_comparisons_do_not_round_through_double() {
    let vars = execute_source(
        r#"
        a = uint64(9007199254740992) + uint64(1);
        equal_rounded = a == 9007199254740992;
        not_equal_rounded = a ~= 9007199254740992;
        greater_than_rounded = a > 9007199254740992;
        "#,
    )
    .expect("exact uint64 comparisons should execute");

    assert!(vars.contains(&Value::Num(0.0)), "equality must be false");
    assert!(
        vars.contains(&Value::Num(1.0)),
        "inequality and ordering must be true"
    );
}

#[test]
fn direct_relational_builtins_keep_uint64_comparisons_exact() {
    let vars = execute_source(
        r#"
        a = uint64(9007199254740992) + uint64(1);
        equal_rounded = eq(a, 9007199254740992);
        greater_than_rounded = gt(a, 9007199254740992);
        "#,
    )
    .expect("direct relational builtins should execute");

    assert!(
        vars.contains(&Value::Bool(false)),
        "eq must be false: {vars:?}"
    );
    assert!(
        vars.contains(&Value::Bool(true)),
        "gt must be true: {vars:?}"
    );
}

#[test]
fn vm_power_paths_keep_uint64_integer_results_exact() {
    let vars = execute_source(
        r#"
        a = uint64(9007199254740992) + uint64(1);
        b = a ^ uint64(1);
        c = uint64(2) .^ uint64(64);
        "#,
    )
    .expect("exact uint64 power paths should execute");

    assert!(
        vars.contains(&Value::Int(IntValue::U64(9_007_199_254_740_993))),
        "matrix-power scalar fallback must preserve exact uint64 input: {vars:?}"
    );
    assert!(
        vars.contains(&Value::Int(IntValue::U64(u64::MAX))),
        "elementwise uint64 power must saturate in the uint64 class: {vars:?}"
    );
}
