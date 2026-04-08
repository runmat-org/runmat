use runmat_builtins::{StringArray, Value};
use runmat_parser::parse;

mod test_helpers;
use test_helpers::{execute, lower};

#[test]
fn datetime_construction_and_component_access_work_in_scripts() {
    let ast =
        parse("d = datetime(2024, 3, 14); y = year(d); m = month(d); daynum = day(d);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();

    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Object(obj) if obj.class_name == "datetime")));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 2024.0).abs() < f64::EPSILON)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 3.0).abs() < f64::EPSILON)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 14.0).abs() < f64::EPSILON)));
}

#[test]
fn datetime_string_and_indexing_work_for_arrays() {
    let ast =
        parse("d = datetime([2024 2025], [1 6], [15 20]); d2 = d(2); y = year(d2); s = string(d);")
            .unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();

    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 2025.0).abs() < f64::EPSILON)));
    assert!(vars.iter().any(|value| match value {
        Value::StringArray(StringArray { data, .. }) => {
            data.iter().any(|text| text == "15-Jan-2024")
                && data.iter().any(|text| text == "20-Jun-2025")
        }
        _ => false,
    }));
}

#[test]
fn datetime_comparisons_and_format_assignment_work() {
    let ast = parse(
        "a = datetime(2024, 1, 1); \
         b = datetime(2024, 1, 2); \
         ok = a < b; \
         a.Format = 'yyyy-MM-dd'; \
         c = char(a);",
    )
    .unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();

    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 1.0).abs() < f64::EPSILON)));
    assert!(vars.iter().any(|value| match value {
        Value::CharArray(array) => {
            let rendered: String = array.data.iter().collect();
            rendered.trim_end() == "2024-01-01"
        }
        _ => false,
    }));
}
