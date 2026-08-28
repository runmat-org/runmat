use runmat_value::{LogicalArray, StringArray, Value};

#[path = "support/mod.rs"]
mod test_helpers;
use test_helpers::execute_source;

fn has_tensor(vars: &[Value], expected: &[f64]) -> bool {
    vars.iter().any(|value| match value {
        Value::Tensor(tensor) => {
            let data = tensor.materialize_f64();
            data.len() == expected.len()
                && data.iter().zip(expected.iter()).all(|(actual, expected)| {
                    if expected.is_nan() {
                        actual.is_nan()
                    } else {
                        (*actual - *expected).abs() < 1e-9
                    }
                })
        }
        _ => false,
    })
}

fn has_logical(vars: &[Value], expected: &[u8]) -> bool {
    vars.iter().any(|value| match value {
        Value::LogicalArray(LogicalArray { data, .. }) => data == expected,
        _ => false,
    })
}

#[test]
fn missing_constructs_and_detects_string_sentinel() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let vars = execute_source("s = missing(1, 2); tf = ismissing(s); anytf = anymissing(s);")
        .expect("missing string script");
    assert!(vars.iter().any(|value| matches!(
        value,
        Value::StringArray(StringArray { data, .. })
            if data == &["<missing>".to_string(), "<missing>".to_string()]
    )));
    assert!(has_logical(&vars, &[1, 1]));
    assert!(vars.iter().any(|value| matches!(value, Value::Bool(true))));
}

#[test]
fn missing_numeric_cleanup_and_fill_execute_from_scripts() {
    let vars = execute_source(
        "A = [1 NaN 3; 4 5 6]; tf = ismissing(A); B = rmmissing(A, 2); C = fillmissing(A, 'constant', 9);",
    )
    .expect("numeric missing cleanup");
    assert!(has_logical(&vars, &[0, 0, 1, 0, 0, 0]));
    assert!(has_tensor(&vars, &[1.0, 4.0, 3.0, 6.0]));
    assert!(has_tensor(&vars, &[1.0, 4.0, 9.0, 5.0, 3.0, 6.0]));
}

#[test]
fn standardize_missing_and_nan_aliases_execute_from_scripts() {
    let vars = execute_source(
        "A = [-99 2 4]; B = standardizeMissing(A, -99); m = nanmean(B, 'all'); s = nansum(B, 'all');",
    )
    .expect("standardize missing and nan aliases");
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (n - 3.0).abs() < 1e-9)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (n - 6.0).abs() < 1e-9)));
}

#[test]
fn movmad_executes_from_scripts() {
    let vars = execute_source("A = [1; 2; 100; 4; 5]; M = movmad(A, 3);").expect("movmad script");
    assert!(has_tensor(&vars, &[0.5, 1.0, 2.0, 1.0, 0.5]));
}

#[test]
fn table_missing_interop_executes_from_scripts() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let vars = execute_source(
        "T = table([1; NaN; 3], strings(3, 1, 'missing'), 'VariableNames', {'A','S'}); M = ismissing(T); R = rmmissing(T);",
    )
    .expect("table missing interop");
    assert!(has_logical(&vars, &[0, 1, 0, 1, 1, 1]));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Object(obj) if obj.class_name == "table")));
}

#[test]
fn numeric_table_fillmissing_executes_from_scripts() {
    let vars = execute_source(
        "T = table([1; NaN; 3], [10; 20; NaN], 'VariableNames', {'A','B'}); [F, M] = fillmissing(T, 'constant', 0);",
    )
    .expect("numeric table fillmissing");
    assert!(has_logical(&vars, &[0, 1, 0, 0, 0, 1]));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Object(obj) if obj.class_name == "table")));
}
