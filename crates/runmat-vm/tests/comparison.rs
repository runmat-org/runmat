use runmat_builtins::{IntValue, LogicalArray, Value};

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
    assert!(vars.iter().any(|value| {
        matches!(
            value,
            Value::Tensor(tensor)
                if tensor.shape == vec![2, 2]
                    && tensor.materialize_f64() == vec![0.0, 30.0, 20.0, 0.0]
        )
    }));
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
        base = uint64(9007199254740992);
        threshold = base + uint64(1);
        values = base + uint64([0 1 2]);
        greater_equal_exact = ge(values, threshold);
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
    assert!(
        vars.iter().any(|value| matches!(
            value,
            Value::LogicalArray(array) if array.data == vec![0, 1, 1]
        )),
        "ge must preserve exact wide integer ordering: {vars:?}"
    );
}

#[test]
fn direct_ne_keeps_wide_integer_comparisons_exact() {
    let vars = execute_source(
        r#"
        base = uint64(9007199254740992);
        values = base + uint64([0 1 2]);
        scalar = ne(values, 9007199254740992);
        mixed = ne(uint64([0 18446744073709551615]), int64([0 9223372036854775807]));
        "#,
    )
    .expect("direct exact inequality");
    assert!(vars.iter().any(|value| matches!(
        value,
        Value::LogicalArray(array) if array.data == vec![0, 1, 1]
    )));
    assert!(vars.iter().any(|value| matches!(
        value,
        Value::LogicalArray(array) if array.data == vec![0, 1]
    )));
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

#[test]
fn vm_integer_power_rejects_invalid_exponents_across_lowerings() {
    let cases = [
        "out = int8(2) .^ (0 - 1);",
        "out = int16(2) .^ 0.5;",
        "out = power(int32(2), 0 - 1);",
        "out = power(int64([2; 3]), int64([1, 0 - 1]));",
        "out = 2.0 .^ int64(0 - 1);",
    ];
    for source in cases {
        let error = execute_source(source).expect_err("invalid compiled integer exponent");
        assert!(
            error
                .to_string()
                .contains("integer exponents must be finite, nonnegative integer values"),
            "{source}: unexpected error: {error}"
        );
    }

    let vars =
        execute_source("out = int16([-2 0]) .^ int16([3 0]);").expect("valid integer power edges");
    assert!(vars.iter().any(|value| matches!(
        value,
        Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&runmat_builtins::IntegerStorage::I16(vec![-8, 1]))
    )));
}

#[test]
fn vm_complex_ordering_compares_real_components_and_preserves_wide_integers() {
    let vars = execute_source(
        r#"
        exact = uint64(9007199254740992) + uint64([1 0]);
        z = complex(exact, uint64([7 0]));
        threshold = 9007199254740992;
        a = z > threshold;
        b = threshold < z;
        c = le(z, threshold);
        d = ge(z, threshold);
        e = complex([2 2], [99 -99]) <= complex([2 1], [-99 99]);
        "#,
    )
    .expect("compiled complex ordering");

    let masks: Vec<&LogicalArray> = vars
        .iter()
        .filter_map(|value| match value {
            Value::LogicalArray(array) => Some(array),
            _ => None,
        })
        .collect();
    assert!(masks.iter().any(|mask| mask.data == vec![1, 0]));
    assert!(masks.iter().any(|mask| mask.data == vec![0, 1]));
    assert!(masks.iter().any(|mask| mask.data == vec![1, 1]));
    assert!(masks.iter().filter(|mask| mask.data == vec![1, 0]).count() >= 3);
}

#[test]
fn compiled_complex_integer_sorting_remains_exact_and_class_preserving() {
    execute_source(
        r#"
        maximum = intmax('uint64');
        z = complex([maximum maximum-uint64(1) uint64(0)], [uint64(0) uint64(1) maximum]);
        [sorted, indices] = sort(z);
        if ~isa(sorted,'uint64') || real(sorted(1)) ~= maximum-uint64(1) || imag(sorted(1)) ~= uint64(1) || real(sorted(2)) ~= maximum || imag(sorted(3)) ~= maximum || ~isequal(indices,[2 1 3]); error('exact complex integer sort'); end;
        if ~issorted(sorted); error('exact complex integer issorted'); end;
        rows = reshape(z(1:2),[2 1]);
        [sortedRows,rowIndices] = sortrows(rows);
        if real(sortedRows(1)) ~= maximum-uint64(1) || ~isequal(rowIndices,[2;1]) || ~issortedrows(sortedRows); error('exact complex integer row order'); end;
        "#,
    )
    .expect("compiled exact complex integer ordering");
}

#[test]
fn vm_integer_scalar_mtimes_is_exact_and_rejects_matrix_forms() {
    let vars = execute_source(
        r#"
        wide = uint64(9007199254740992) + uint64([1 0]);
        scale = uint64(2);
        a = wide * scale;
        b = mtimes(scale, wide);
        signed = int8([127 -128 2]) * int8(2);
        "#,
    )
    .expect("compiled integer scalar mtimes");
    assert!(
        vars.iter()
            .filter(|value| matches!(
                value,
                Value::Tensor(tensor)
                    if tensor.integer_storage()
                        == Some(&runmat_builtins::IntegerStorage::U64(vec![
                            18_014_398_509_481_986,
                            18_014_398_509_481_984,
                        ]))
            ))
            .count()
            >= 2
    );
    assert!(vars.iter().any(|value| matches!(
        value,
        Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&runmat_builtins::IntegerStorage::I8(vec![127, -128, 4]))
    )));

    for source in [
        "out = int16([1 2]) * int16([3; 4]);",
        "out = int16([1 2]) * uint16(2);",
        "out = int16(2) * [1 2];",
    ] {
        let error = execute_source(source).expect_err("invalid compiled integer mtimes");
        assert_eq!(error.identifier(), Some("RunMat:mtimes:InvalidInput"));
    }
}
