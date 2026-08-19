use runmat_value::{IntegerStorage, Value};
#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_integer_grid_imag_and_nextpow2_forms_preserve_native_classes() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "a = {constructor}([0 3]); p = nextpow2(a); z = imag(a); [mx,my] = meshgrid(a); [nx,ny] = ndgrid(a);"
        );
        let values = execute_source(&source).expect("compiled integer construction semantics");
        assert!(
            values.iter().any(
                |value| matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some_and(|storage| storage.class_name() == constructor && storage.exact_values().iter().map(|value| value.to_f64()).collect::<Vec<_>>() == vec![0.0, 2.0]))
            ),
            "{constructor} nextpow2 output"
        );
        assert!(
            values.iter().any(
                |value| matches!(value, Value::Tensor(tensor) if tensor.shape == vec![2, 2] && tensor.integer_storage().is_some_and(|storage| storage.class_name() == constructor))
            ),
            "{constructor} grid output"
        );
    }
}

#[test]
fn compiled_magic_uses_first_real_element_floor_and_supports_order_two() {
    let values =
        execute_source("a = magic([3.9 99]); b = magic(2);").expect("compiled magic semantics");
    assert!(values
        .iter()
        .any(|value| matches!(value, Value::Tensor(tensor) if tensor.shape == vec![3, 3])));
    assert!(values.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.shape == vec![2, 2] && tensor.materialize_f64() == vec![1.0, 4.0, 3.0, 2.0])
    }));
}

#[test]
fn compiled_mat2cell_mat2str_and_native2unicode_keep_exact_integer_boundaries() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let values = execute_source(
        "base = uint64(9007199254740992); a = [base+uint64(1) intmax('uint64')]; c = mat2cell(a,uint8(1),uint8([1 1])); s = mat2str(uint16([256 512]),'class'); text = native2unicode(uint16([104 105]));",
    )
    .expect("compiled conversion semantics");
    assert!(values.iter().any(|value| {
        matches!(value, Value::Cell(cell) if cell.data.iter().any(|entry| matches!(entry, Value::Int(integer) if integer.decimal_string() == "18446744073709551615")))
    }));
    assert!(values.iter().any(|value| {
        matches!(value, Value::CharArray(array) if array.data.iter().collect::<String>() == "uint16([256 512])")
    }));
    assert!(values.iter().any(|value| {
        matches!(value, Value::CharArray(array) if array.shape == vec![1, 2] && array.data.iter().collect::<String>() == "hi")
    }));
}

#[test]
fn matlab_mode_rejects_only_the_evidence_bounded_typed_control_extensions() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let partition_error = execute_source("c = mat2cell(uint64(1),uint8(1));")
        .expect_err("typed partition must be gated");
    assert_eq!(
        partition_error.identifier(),
        Some("RunMat:compatibility:Mat2cellIntegerPartitionsExtension")
    );
    let precision_error = execute_source("s = mat2str(uint16(12),uint8(3));")
        .expect_err("typed precision must be gated");
    assert_eq!(
        precision_error.identifier(),
        Some("RunMat:compatibility:Mat2strIntegerPrecisionExtension")
    );
}

#[test]
fn compiled_native2unicode_rejects_out_of_range_integer_bytes() {
    for source in [
        "text = native2unicode(int16(-1));",
        "text = native2unicode(uint16(256));",
    ] {
        let error = execute_source(source).expect_err("invalid byte must reject");
        assert!(error.message().contains("0 through 255"));
    }
}

#[test]
fn compiled_wide_uint64_grid_values_do_not_cross_binary64() {
    let values = execute_source(
        "base = uint64(9007199254740992); a = [base+uint64(1) intmax('uint64')]; [x,y] = meshgrid(a); [p,q] = ndgrid(a);",
    )
    .expect("compiled wide grid semantics");
    assert!(values.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_993, u64::MAX, u64::MAX])))
    }));
}
