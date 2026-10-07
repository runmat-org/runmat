use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_with_numeric_range() {
    let path = write_temp_file(&["1,2,3", "4,5,6", "7,8,9"]);
    let range = BuiltinTensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]).expect("tensor");
    let args = vec![Value::from(","), Value::Tensor(range)];
    let result =
        dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("dlmread");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![5.0, 8.0, 6.0, 9.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_range_skips_nonnumeric_header_column() {
    let path = write_temp_file(&["Label,Jan,Feb", "alpha,1,2", "beta,3,4"]);
    let range = BuiltinTensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]).expect("tensor");
    let args = vec![Value::from(","), Value::Tensor(range)];
    let result =
        dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("dlmread");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_numeric_range_two_elements() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let path = write_temp_file(&["1,2,3", "4,5,6", "7,8,9"]);
    let range = BuiltinTensor::new(vec![1.0, 1.0], vec![2, 1]).expect("tensor");
    let args = vec![Value::Tensor(range)];
    let result =
        dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("dlmread");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![5.0, 8.0, 6.0, 9.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_excel_style_range_string() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let path = write_temp_file(&["1,2,3,4", "5,6,7,8", "9,10,11,12"]);
    let args = vec![Value::from(","), Value::from("B2:C3")];
    let result =
        dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("dlmread");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![6.0, 10.0, 7.0, 11.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_range_without_delimiter() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let path = write_temp_file(&["1,2,3", "4,5,6", "7,8,9"]);
    let range = BuiltinTensor::new(vec![1.0, 0.0, 2.0, 1.0], vec![4, 1]).expect("tensor");
    let args = vec![Value::Tensor(range)];
    let result =
        dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("dlmread");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![4.0, 7.0, 5.0, 8.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_nonnumeric_token_error() {
    let path = write_temp_file(&["1,foo"]);
    let err = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
        .expect_err("dlmread should fail");
    let message = err.message().to_string();
    assert!(
        message.contains("nonnumeric token 'foo' at row 1, column 2"),
        "unexpected error message: {message}"
    );
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_invalid_range_error() {
    let path = write_temp_file(&["1,2,3", "4,5,6"]);
    let range = BuiltinTensor::new(vec![2.0, 1.0, 1.0, 3.0], vec![4, 1]).expect("tensor");
    let args = vec![Value::from(","), Value::Tensor(range)];
    let err = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args)
        .expect_err("dlmread should fail");
    let message = err.message().to_string();
    assert!(
        message.contains("Range must satisfy R1 <= R2 and C1 <= C2"),
        "unexpected error message: {message}"
    );
    fs::remove_file(path).ok();
}

#[test]
fn dlmread_numeric_range_reads_typed_integer_storage_exactly() {
    let range = BuiltinTensor::new_integer(IntegerStorage::U16(vec![1, 2, 3, 4]), vec![4, 1])
        .expect("range");

    let parsed = parse_range_numeric(&Value::Tensor(range)).expect("range");

    assert_eq!(parsed.start_row, 1);
    assert_eq!(parsed.start_col, 2);
    assert_eq!(parsed.end_row, Some(3));
    assert_eq!(parsed.end_col, Some(4));
}

#[test]
fn dlmread_accepts_documented_double_dot_spreadsheet_range() {
    let range = parse_range_string("A1..B2").unwrap();
    assert_eq!(range.start_row, 0);
    assert_eq!(range.start_col, 0);
    assert_eq!(range.end_row, Some(1));
    assert_eq!(range.end_col, Some(1));
}

#[test]
fn dlmread_numeric_delimiter_is_a_gated_extension() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = parse_delimiter(&Value::Int(IntValue::U8(b',')))
        .expect_err("numeric delimiter must reject in strict mode");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:DlmreadNumericDelimiterExtension")
    );
}

#[test]
fn dlmread_colon_spreadsheet_range_is_a_gated_extension() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = parse_range_string("A1:B2")
        .expect_err("colon spreadsheet range must reject in strict mode");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:DlmreadColonSpreadsheetRangeExtension")
    );
    drop(strict);

    let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let range = parse_range_string("A1:B2").expect("colon range in RunMat mode");
    assert_eq!(range.start_row, 0);
    assert_eq!(range.start_col, 0);
    assert_eq!(range.end_row, Some(1));
    assert_eq!(range.end_col, Some(1));
    drop(extensions);
}
