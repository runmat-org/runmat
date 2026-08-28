use runmat_value::{NumericDType, Value};
#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_normal_family_executes_every_integer_class_in_runmat_mode() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "x = {constructor}([0 1]); n = norm({constructor}([3 4])); p = normcdf(x); d = normpdf(x); q = norminv({constructor}(0));"
        );
        let values = execute_source(&source).expect("compiled normal-family semantics");
        assert!(
            values.iter().any(
                |value| matches!(value, Value::Num(number) if (*number - 5.0).abs() < 1.0e-12)
            ),
            "{constructor} norm"
        );
        assert!(
            values
                .iter()
                .filter_map(|value| match value {
                    Value::Tensor(tensor) if tensor.shape == vec![1, 2] => Some(tensor),
                    _ => None,
                })
                .count()
                >= 2,
            "{constructor} distribution arrays"
        );
    }
}

#[test]
fn compiled_single_normal_and_normalize_outputs_preserve_single_class() {
    let values = execute_source(
        "x = single([3 4]); n = norm(x); z = normalize(x); p = normcdf(single([0 1]));",
    )
    .expect("compiled single semantics");
    assert!(values.iter().filter(|value| matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)).count() >= 3);
}

#[test]
fn compiled_random_and_regression_integer_extensions_keep_declared_shapes() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let values = execute_source(
        "r = normrnd(int16(0),uint16(1),uint8(2),uint8(3)); m = mvnrnd(int16([0 0]),uint16([1 0;0 1]),uint8(3)); X = int16([0;1;2;3;4]); Y = uint8([1;1;2;2;2]); B = mnrfit(X,Y,'IterationLimit',uint16(200));",
    )
    .expect("compiled random and regression semantics");
    assert!(values
        .iter()
        .any(|value| matches!(value, Value::Tensor(tensor) if tensor.shape == vec![2, 3])));
    assert!(values
        .iter()
        .any(|value| matches!(value, Value::Tensor(tensor) if tensor.shape == vec![3, 2])));
    assert!(values
        .iter()
        .any(|value| matches!(value, Value::Tensor(tensor) if tensor.shape == vec![2, 1])));
}

#[test]
fn matlab_mode_rejects_integer_extensions_before_floating_evaluation() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "n = norm(uint16([3 4]));",
            "RunMat:compatibility:NormIntegerDataExtension",
        ),
        (
            "p = normcdf(uint16(0));",
            "RunMat:compatibility:NormcdfIntegerXExtension",
        ),
        (
            "r = normrnd(uint16(0),1);",
            "RunMat:compatibility:NormrndIntegerMuExtension",
        ),
        (
            "r = mvnrnd(uint16([0 0]),[1 0;0 1]);",
            "RunMat:compatibility:MvnrndIntegerMuExtension",
        ),
        (
            "B = mnrfit(uint16([0;1]),[1;2]);",
            "RunMat:compatibility:MnrfitIntegerXExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict compatibility gate");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_wide_integer_floating_boundaries_reject_without_rounding() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for source in [
        "x = uint64(9007199254740992)+uint64(1); n = norm(x);",
        "x = uint64(9007199254740992)+uint64(1); p = normcdf(x);",
        "x = uint64(9007199254740992)+uint64(1); r = normrnd(x,1);",
    ] {
        let error = execute_source(source).expect_err("lossy binary64 boundary must reject");
        assert!(
            error.message().contains("exactly representable"),
            "{source}"
        );
    }
}
