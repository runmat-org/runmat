#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::{IntValue, IntegerStorage, Value};
use test_helpers::execute_source;

fn numeric_values(values: &[Value]) -> Vec<f64> {
    values
        .iter()
        .flat_map(|value| match value {
            Value::Num(value) => vec![*value],
            Value::Tensor(tensor) => tensor.materialize_f64(),
            _ => Vec::new(),
        })
        .collect()
}

#[test]
fn compiled_datetime_date_vectors_and_components_cover_all_integer_classes() {
    for (constructor, year) in [
        ("int8", 24),
        ("int16", 2024),
        ("int32", 2024),
        ("int64", 2024),
        ("uint8", 24),
        ("uint16", 2024),
        ("uint32", 2024),
        ("uint64", 2024),
    ] {
        let source = format!(
            "a = datetime({constructor}([{year} 1 2])); av = year(a)*10000 + month(a)*100 + day(a); b = datetime({constructor}({year}), {constructor}(1), {constructor}(2)); bv = year(b)*10000 + month(b)*100 + day(b);"
        );
        let values = execute_source(&source).expect("compiled typed datetime constructors");
        let stamp = (year * 10_000 + 102) as f64;
        assert_eq!(
            numeric_values(&values)
                .into_iter()
                .filter(|value| *value == stamp)
                .count(),
            2,
            "{constructor} date-vector and component forms"
        );
    }
}

#[test]
fn compiled_dateshift_and_day_use_exact_integer_controls() {
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "t = datetime(2024,3,14); shifted = dateshift(t, 'dayofweek', {constructor}(2)); stamp = year(shifted)*10000 + month(shifted)*100 + day(shifted); ordinal = day(t, 'dayofyear');"
        );
        let values = execute_source(&source).expect("compiled typed dateshift control");
        let numbers = numeric_values(&values);
        assert!(
            numbers.contains(&20_240_318.0),
            "{constructor} shifted date"
        );
        assert!(numbers.contains(&74.0), "{constructor} day-of-year");
    }
}

#[test]
fn compiled_datetime_and_day_reject_wide_integer_serials_exactly() {
    for source in [
        "u = uint64(9007199254740992) + uint64(1); t = datetime(u, 'ConvertFrom', 'datenum');",
        "u = uint64(9007199254740992) + uint64(1); d = day(u);",
    ] {
        let error = execute_source(source).expect_err("wide serial must reject before conversion");
        assert_eq!(error.identifier(), Some("RunMat:datetime:InvalidInput"));
        assert!(
            error.message().contains("supported serial-date range"),
            "{}",
            error.message()
        );
    }
}

#[test]
fn compiled_integer_decomposition_dispatches_transpose_and_both_solves() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let values = execute_source(
        "d = decomposition(uint64([2 0; 1 4])); x = d \\ [4;10]; y = [6 8] / d; dt = d'; transposed = dt.IsConjugateTransposed; xt = dt \\ [6;8];",
    )
    .expect("compiled integer decomposition operations");

    assert!(values.iter().any(|value| {
        matches!(
            value,
            Value::Object(object)
                if object.class_name == "decomposition"
                    && matches!(
                        object.properties.get("__matrix"),
                        Some(Value::Tensor(tensor))
                            if tensor.integer_storage()
                                == Some(&IntegerStorage::U64(vec![2, 1, 0, 4]))
                    )
        )
    }));
    assert!(values
        .iter()
        .any(|value| matches!(value, Value::Bool(true))));
    assert_eq!(
        values
            .iter()
            .filter(|value| {
                matches!(value, Value::Tensor(tensor)
                    if matches!(tensor.shape.as_slice(), [2, 1] | [1, 2])
                        && tensor.materialize_f64().iter().all(|entry| (*entry - 2.0).abs() < 1.0e-12))
            })
            .count(),
        3
    );
}

#[test]
fn compiled_integer_decomposition_obeys_strict_mode() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let error = execute_source("d = decomposition(int16([2 0; 0 4]));")
        .expect_err("strict mode integer construction");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:DecompositionNonfloatingInputExtension")
    );
}

#[test]
fn compiled_scalar_left_integer_division_preserves_all_classes_exactly() {
    for (constructor, values, expected) in [
        ("int8", "[6 -8 10]", IntegerStorage::I8(vec![3, -4, 5])),
        ("int16", "[6 -8 10]", IntegerStorage::I16(vec![3, -4, 5])),
        ("int32", "[6 -8 10]", IntegerStorage::I32(vec![3, -4, 5])),
        ("int64", "[6 -8 10]", IntegerStorage::I64(vec![3, -4, 5])),
        ("uint8", "[6 8 10]", IntegerStorage::U8(vec![3, 4, 5])),
        ("uint16", "[6 8 10]", IntegerStorage::U16(vec![3, 4, 5])),
        ("uint32", "[6 8 10]", IntegerStorage::U32(vec![3, 4, 5])),
        ("uint64", "[6 8 10]", IntegerStorage::U64(vec![3, 4, 5])),
    ] {
        let source = format!(
            "a = {constructor}(2) \\ {constructor}({values}); b = mldivide({constructor}(2), {constructor}({values}));"
        );
        let results = execute_source(&source).expect("compiled integer scalar-left division");
        assert_eq!(
            results
                .iter()
                .filter(|value| {
                    matches!(value, Value::Tensor(tensor) if tensor.integer_storage() == Some(&expected))
                })
                .count(),
            2,
            "{constructor} operator and builtin results"
        );
    }
}

#[test]
fn compiled_deal_distributes_and_replicates_exact_integer_values() {
    let values = execute_source(
        "function varargout = replicate(x); [varargout{1:nargout}] = deal(x); end; u = uint64(9007199254740992) + uint64(1); [a,b,c] = replicate(u); [d,e] = deal(int8(-7), uint16(60000));",
    )
    .expect("compiled integer deal forms");

    assert!(
        values
            .iter()
            .filter(|value| matches!(value, Value::Int(IntValue::U64(9_007_199_254_740_993))))
            .count()
            >= 4
    );
    assert!(values
        .iter()
        .any(|value| matches!(value, Value::Int(IntValue::I8(-7)))));
    assert!(values
        .iter()
        .any(|value| matches!(value, Value::Int(IntValue::U16(60_000)))));
}

#[test]
fn compiled_deal_enforces_counts_and_accepts_zero_outputs() {
    let error = execute_source("[a,b,c] = deal(1,2);").expect_err("deal count mismatch");
    assert_eq!(
        error.identifier(),
        Some("RunMat:deal:InputOutputCountMismatch")
    );

    execute_source("deal(uint32(7));").expect("single-input zero-output deal");
}

#[test]
fn compiled_dcgain_rejects_integer_sys_but_accepts_integer_tf_coefficients() {
    let error = execute_source("g = dcgain(int64(7));").expect_err("integer sys rejection");
    assert_eq!(error.identifier(), Some("RunMat:dcgain:InvalidModel"));

    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let values = execute_source("g = dcgain(tf(int16(2), int16([1 3])));")
        .expect("dcgain of tf with integer coefficients");
    assert!(values
        .iter()
        .any(|value| matches!(value, Value::Num(gain) if (*gain - 2.0 / 3.0).abs() < 1.0e-12)));
}
