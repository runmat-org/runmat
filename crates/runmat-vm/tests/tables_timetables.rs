use runmat_builtins::{CellArray, Tensor, Value};

#[path = "support/mod.rs"]
mod test_helpers;
use test_helpers::execute_source;

fn has_bool(vars: &[Value], expected: bool) -> bool {
    vars.iter()
        .any(|value| matches!(value, Value::Bool(value) if *value == expected))
}

fn has_table(vars: &[Value]) -> bool {
    vars.iter()
        .any(|value| matches!(value, Value::Object(object) if object.class_name == "table"))
}

fn has_timetable(vars: &[Value]) -> bool {
    vars.iter()
        .any(|value| matches!(value, Value::Object(object) if object.class_name == "timetable"))
}

fn has_tensor(vars: &[Value], expected: &[f64]) -> bool {
    vars.iter().any(|value| match value {
        Value::Tensor(Tensor { data, .. }) => data == expected,
        _ => false,
    })
}

fn has_logical_array(vars: &[Value], expected: &[u8]) -> bool {
    vars.iter().any(|value| match value {
        Value::LogicalArray(array) => array.data == expected,
        _ => false,
    })
}

fn has_num(vars: &[Value], expected: f64) -> bool {
    vars.iter().any(
        |value| matches!(value, Value::Num(value) if (*value - expected).abs() <= f64::EPSILON),
    )
}

#[test]
fn table_conversion_surface_executes_from_scripts() {
    let vars = execute_source(
        "T = array2table([1 3; 2 4], 'VariableNames', {'A','B'}); tf = istable(T); A = table2array(T); C = table2cell(T); S = table2struct(T); y = S(2).A; T2 = struct2table(S); H = head(T, 1);",
    )
    .expect("table conversion script");
    assert!(has_bool(&vars, true));
    assert!(has_table(&vars));
    assert!(has_tensor(&vars, &[1.0, 2.0, 3.0, 4.0]));
    assert!(has_num(&vars, 2.0));
    assert!(vars.iter().any(|value| matches!(
        value,
        Value::Cell(CellArray {
            rows: 2,
            cols: 2,
            ..
        })
    )));
}

#[test]
fn timetable_conversion_surface_executes_from_scripts() {
    let vars = execute_source(
        "TT = timetable([1; 2], [10; 20], 'VariableNames', {'X'}); tf = istimetable(TT); H = head(TT, 1); tm = H.Time; T = timetable2table(TT, 'ConvertRowTimes', true); TT2 = table2timetable(T);",
    )
    .expect("timetable conversion script");
    assert!(has_bool(&vars, true));
    assert!(has_table(&vars));
    assert!(has_timetable(&vars));
    assert!(has_tensor(&vars, &[1.0]));
}

#[test]
fn categorical_dictionary_and_selector_surface_executes_from_scripts() {
    let vars = execute_source(
        "C = categorical({'red'; 'blue'; 'red'}); tf = iscategorical(C); O = ordinal({'medium'; 'low'}, {'low','medium','high'}); of = isordinal(O); nf = isordinal(C); gtmask = O > 'low'; eqmask = O == 'medium'; lemask = O <= 'medium'; out = [of + 0, nf + 0]; D = dictionary({'a','b'}, {1,2}); R = timerange(1, 3); V = vartype('numeric'); F = rowfilter({'A'}, '@gt0'); DS = arrayDatastore([1; 2]); UI = uitable('Data', [1 2]);",
    )
    .expect("categorical dictionary script");
    assert!(has_bool(&vars, true));
    assert!(has_bool(&vars, false));
    assert!(has_tensor(&vars, &[1.0, 0.0]));
    assert!(has_logical_array(&vars, &[1, 0]));
    assert!(has_logical_array(&vars, &[1, 1]));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Object(object) if object.class_name == "categorical")));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Object(object) if object.class_name == "dictionary")));
}
