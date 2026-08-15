#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::{IntegerStorage, NumericDType, Value};
use test_helpers::execute_source;

#[test]
fn compiled_round_and_rot90_preserve_exact_wide_integer_storage() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let values = execute_source(
        "w=uint64(9007199254740992)+uint64(1); a=reshape([w w+uint64(2) uint64(3) uint64(4)],2,2); r=round(a); q=rot90(a,int64(-1));",
    )
    .expect("compiled exact rounding and rotation");
    assert!(values.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_995, 3, 4])))
    }));
    assert!(values.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::U64 && tensor.integer_storage().is_some_and(|storage| storage.exact_values().iter().any(|value| value.try_to_u64() == Some(9_007_199_254_740_993))))
    }));
}

#[test]
fn compiled_strict_mode_rejects_checked_integer_extensions() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "r=roots(uint8([1 0 1]));",
            "RunMat:compatibility:RootsIntegerCoefficientsExtension",
        ),
        (
            "r=rref(uint8([1 0;0 1]));",
            "RunMat:compatibility:RrefIntegerMatrixExtension",
        ),
        (
            "r=round(1.25,uint8(1));",
            "RunMat:compatibility:RoundTypedIntegerDigitsExtension",
        ),
        (
            "h=scatterplot(uint8([1 2]));",
            "RunMat:compatibility:ScatterplotIntegerDataExtension",
        ),
        (
            "h=scatterhist(uint8([1 2]),[1 2]);",
            "RunMat:compatibility:ScatterhistIntegerDataExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict extension must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_runmat_mode_rejects_inexact_floating_boundaries() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for source in [
        "w=uint64(9007199254740992)+uint64(1); r=roots([uint64(1) w]);",
        "w=uint64(9007199254740992)+uint64(1); r=rref([uint64(1) w]);",
        "w=uint64(9007199254740992)+uint64(1); h=scatterplot([uint64(1) w]);",
    ] {
        let error = execute_source(source).expect_err("wide boundary must reject");
        assert!(
            error.message().contains("exactly representable"),
            "{source}"
        );
    }
}

#[test]
fn compiled_runtests_flags_require_exact_zero_or_one() {
    let error =
        execute_source("r=runtests('definitely_missing_test_file','IncludeSubfolders',uint8(2));")
            .expect_err("invalid integer flag must reject before discovery");
    assert_eq!(error.identifier(), Some("RunMat:runtests:InvalidInput"));
    assert!(error.message().contains("0 or 1"));
}
