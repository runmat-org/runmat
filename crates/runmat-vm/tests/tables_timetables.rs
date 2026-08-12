use runmat_value::{CellArray, IntegerStorage, Value};

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
        Value::Tensor(tensor) => tensor.materialize_f64() == expected,
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

fn has_integer_storage(vars: &[Value], expected: &IntegerStorage) -> bool {
    vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.integer_storage() == Some(expected))
    })
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
fn array2table_compiled_surface_preserves_all_integer_classes() {
    let vars = execute_source(
        "a = table2array(array2table(int8([-128 127]))); b = table2array(array2table(int16([-32768 32767]))); c = table2array(array2table(int32([-2147483648 2147483647]))); d = table2array(array2table(int64([-7 9]))); e = table2array(array2table(uint8([0 255]))); f = table2array(array2table(uint16([0 65535]))); g = table2array(array2table(uint32([0 4294967295]))); base = uint64(9007199254740992); h = table2array(array2table(base + uint64([1 2])));",
    )
    .expect("compiled array2table integer conversion");
    for expected in [
        IntegerStorage::I8(vec![-128, 127]),
        IntegerStorage::I16(vec![-32768, 32767]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![-7, 9]),
        IntegerStorage::U8(vec![0, 255]),
        IntegerStorage::U16(vec![0, 65535]),
        IntegerStorage::U32(vec![0, u32::MAX]),
        IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_994]),
    ] {
        assert!(
            has_integer_storage(&vars, &expected),
            "missing compiled storage {expected:?}"
        );
    }
}

#[test]
fn array2timetable_compiled_surface_preserves_all_integer_classes() {
    let vars = execute_source(
        "a = table2array(timetable2table(array2timetable(int8([-128 127]), 'SampleRate', int8(2)))); b = table2array(timetable2table(array2timetable(int16([-32768 32767]), 'SampleRate', int16(2)))); c = table2array(timetable2table(array2timetable(int32([-2147483648 2147483647]), 'SampleRate', int32(2)))); d = table2array(timetable2table(array2timetable(int64([-7 9]), 'SampleRate', int64(2)))); e = table2array(timetable2table(array2timetable(uint8([0 255]), 'SampleRate', uint8(2)))); f = table2array(timetable2table(array2timetable(uint16([0 65535]), 'SampleRate', uint16(2)))); g = table2array(timetable2table(array2timetable(uint32([0 4294967295]), 'SampleRate', uint32(2)))); base = uint64(9007199254740992); h = table2array(timetable2table(array2timetable(base + uint64([1 2]), 'SampleRate', uint64(2))));",
    )
    .expect("compiled array2timetable integer conversion");
    for expected in [
        IntegerStorage::I8(vec![-128, 127]),
        IntegerStorage::I16(vec![-32768, 32767]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![-7, 9]),
        IntegerStorage::U8(vec![0, 255]),
        IntegerStorage::U16(vec![0, 65535]),
        IntegerStorage::U32(vec![0, u32::MAX]),
        IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_994]),
    ] {
        assert!(
            has_integer_storage(&vars, &expected),
            "missing compiled storage {expected:?}"
        );
    }
}

#[test]
fn array_datastore_compiled_surface_preserves_all_integer_classes() {
    let vars = execute_source(
        "a = arrayDatastore(int8([-128 127]), 'ReadSize', 2, 'OutputType', 'same').Data; b = arrayDatastore(int16([-32768 32767])).Data; c = arrayDatastore(int32([-2147483648 2147483647])).Data; d = arrayDatastore(int64([-7 9])).Data; e = arrayDatastore(uint8([0 255])).Data; f = arrayDatastore(uint16([0 65535])).Data; g = arrayDatastore(uint32([0 4294967295])).Data; base = uint64(9007199254740992); h = arrayDatastore(base + uint64([1 2])).Data;",
    )
    .expect("compiled arrayDatastore integer construction");
    for expected in [
        IntegerStorage::I8(vec![-128, 127]),
        IntegerStorage::I16(vec![-32768, 32767]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![-7, 9]),
        IntegerStorage::U8(vec![0, 255]),
        IntegerStorage::U16(vec![0, 65535]),
        IntegerStorage::U32(vec![0, u32::MAX]),
        IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_994]),
    ] {
        assert!(
            has_integer_storage(&vars, &expected),
            "missing compiled storage {expected:?}"
        );
    }
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

#[test]
fn dictionary_compiled_surface_preserves_wide_integer_keys_and_scalar_expansion() {
    let vars = execute_source(
        "base = uint64(9007199254740992); keys = base + uint64([1 2 1]); D = dictionary(keys, int16(7)); x = D(base + uint64(2)); column = D(base + uint64([2; 1])); D(base + uint64([1 2])) = int16(9); y = D(base + uint64(1)); D(base + uint64(2)) = [];",
    )
    .expect("compiled exact integer dictionary construction and mutation");
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Int(runmat_value::IntValue::I16(7)))));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Int(runmat_value::IntValue::I16(9)))));
    assert!(vars.iter().any(|value| {
        matches!(
            value,
            Value::Tensor(tensor)
                if tensor.shape == [2, 1]
                    && tensor.integer_storage()
                        == Some(&runmat_value::IntegerStorage::I16(vec![7, 7]))
        )
    }));
    let dictionary = vars
        .iter()
        .find_map(|value| match value {
            Value::Object(object) if object.class_name == "dictionary" => Some(object),
            _ => None,
        })
        .expect("dictionary result");
    let Value::Cell(keys) = dictionary.properties.get("Keys").unwrap() else {
        panic!("dictionary keys");
    };
    assert_eq!(keys.data.len(), 1);
    assert_eq!(
        keys.data[0],
        Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993))
    );
}

#[test]
fn categorical_compiled_surface_preserves_exact_integer_identity_and_flags() {
    let vars = execute_source(
        "base = uint64(9007199254740992); C = categorical(base + uint64([1 2]), base + uint64([2 1]), {'two','one'}, 'Ordinal', uint8(1)); tf = isordinal(C);",
    )
    .expect("compiled categorical integer construction");
    assert!(has_bool(&vars, true));
    let object = vars
        .iter()
        .find_map(|value| match value {
            Value::Object(object) if object.class_name == "categorical" => Some(object),
            _ => None,
        })
        .expect("categorical object");
    match object.properties.get("Codes").expect("codes") {
        Value::Tensor(codes) => assert_eq!(codes.materialize_f64(), vec![2.0, 1.0]),
        other => panic!("expected categorical codes, got {other:?}"),
    }
    match object.properties.get("Categories").expect("categories") {
        Value::StringArray(categories) => assert_eq!(categories.data, vec!["two", "one"]),
        other => panic!("expected categorical names, got {other:?}"),
    }
}
