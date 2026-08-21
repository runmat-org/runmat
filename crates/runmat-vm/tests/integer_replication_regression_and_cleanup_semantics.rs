use runmat_value::{IntValue, IntegerStorage, NumericDType, Value};
#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

fn contains_integer_storage(values: &[Value], expected: &IntegerStorage) -> bool {
    values.iter().any(|value| match value {
        Value::Tensor(tensor) => tensor.integer_storage() == Some(expected),
        Value::Struct(structure) => contains_integer_storage(
            &structure.fields.values().cloned().collect::<Vec<_>>(),
            expected,
        ),
        Value::Cell(cell) => contains_integer_storage(&cell.data, expected),
        Value::Int(value) => match (value, expected) {
            (IntValue::I8(actual), IntegerStorage::I8(expected)) => {
                expected.as_slice() == [*actual]
            }
            (IntValue::I16(actual), IntegerStorage::I16(expected)) => {
                expected.as_slice() == [*actual]
            }
            (IntValue::I32(actual), IntegerStorage::I32(expected)) => {
                expected.as_slice() == [*actual]
            }
            (IntValue::I64(actual), IntegerStorage::I64(expected)) => {
                expected.as_slice() == [*actual]
            }
            (IntValue::U8(actual), IntegerStorage::U8(expected)) => {
                expected.as_slice() == [*actual]
            }
            (IntValue::U16(actual), IntegerStorage::U16(expected)) => {
                expected.as_slice() == [*actual]
            }
            (IntValue::U32(actual), IntegerStorage::U32(expected)) => {
                expected.as_slice() == [*actual]
            }
            (IntValue::U64(actual), IntegerStorage::U64(expected)) => {
                expected.as_slice() == [*actual]
            }
            _ => false,
        },
        _ => false,
    })
}

#[test]
fn compiled_structural_integer_forms_preserve_exact_storage() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let values = execute_source(
        "w=uint64(9007199254740992)+uint64(1); a=repmat(w,uint8(1),uint16(2)); b=repelem(w,uint32(2)); c=reshape(a,uint8(2),uint8(1)); s=struct('keep',w,'drop',uint8(1)); t=rmfield(s,'drop'); r=rmmissing(a);",
    )
    .expect("compiled structural integer semantics");
    let wide_pair = IntegerStorage::U64(vec![9_007_199_254_740_993; 2]);
    assert!(contains_integer_storage(&values, &wide_pair));
    assert!(contains_integer_storage(
        &values,
        &IntegerStorage::U64(vec![9_007_199_254_740_993])
    ));
}

#[test]
fn compiled_documented_integer_floating_and_image_forms_have_declared_outputs() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let values =
        execute_source("r=rescale(uint16([1 2 3])); g=rgb2gray(uint8(reshape([255 0 0],1,1,3)));")
            .expect("documented integer floating forms");
    assert!(values.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64 && tensor.shape == vec![1, 3])
    }));
    assert!(values.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.integer_storage() == Some(&IntegerStorage::U8(vec![76])))
            || matches!(value, Value::Int(runmat_value::IntValue::U8(76)))
    }));
}

#[test]
fn matlab_mode_rejects_regression_signal_control_and_dimension_extensions() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "b=regress(uint8([1;2;3]),uint8([1 1;1 2;1 3]));",
            "RunMat:compatibility:RegressIntegerDataExtension",
        ),
        (
            "b=ridge(uint8([1;2;3]),uint8([1 1;1 2;1 3]),uint8(1));",
            "RunMat:compatibility:RidgeIntegerDataExtension",
        ),
        (
            "y=resample([1 2 3],uint8(2),1);",
            "RunMat:compatibility:ResampleIntegerOptionsExtension",
        ),
        (
            "r=rmmissing([1 NaN],uint8(2));",
            "RunMat:compatibility:RmmissingIntegerDimensionExtension",
        ),
        (
            "r=rlocus(tf([1],[1 1]),uint8([0 1]));",
            "RunMat:compatibility:RlocusIntegerGainExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict extension must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn runmat_mode_executes_checked_regression_and_control_extensions() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let values = execute_source(
        "b=regress(uint8([1;2;3]),uint8([1 1;1 2;1 3])); d=ridge(uint8([1;2;3]),uint8([1 1;1 2;1 3]),uint8(1)); y=resample([1 2 3],uint8(2),uint8(1));",
    )
    .expect("RunMat integer extensions");
    assert!(values
        .iter()
        .filter(|value| matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64))
        .count()
        >= 3);
}

#[test]
fn checked_floating_boundaries_reject_wide_integers_without_rounding() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for source in [
        "w=uint64(9007199254740992)+uint64(1); b=regress([1;2;3],[1 1;1 2;1 w]);",
        "w=uint64(9007199254740992)+uint64(1); b=ridge([1;2;3],[1 1;1 2;1 3],w);",
        "w=uint64(9007199254740992)+uint64(1); r=rescale(w);",
    ] {
        let error = execute_source(source).expect_err("lossy binary64 boundary must reject");
        assert!(
            error.message().contains("exactly representable"),
            "{source}"
        );
    }
}

#[test]
fn text_only_cleanup_apis_reject_integer_inputs() {
    for source in ["d=removeStopWords(uint8(1));", "p=rmpath(uint8(1));"] {
        execute_source(source).expect_err("integer input must reject as non-text");
    }
}
