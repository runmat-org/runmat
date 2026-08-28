#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::{IntegerStorage, Value};
use test_helpers::execute_source;

const INTEGER_CONSTRUCTORS: [&str; 8] = [
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
];

#[test]
fn compiled_full_and_getfield_preserve_every_integer_class() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            "dense = full({constructor}([0 1])); s = struct(); s.values = {constructor}([0 1]); selected = getfield(s, 'values', {{{constructor}([2 1])}});"
        );
        let values = execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor} compiled structural semantics: {error}"));
        assert!(values.iter().any(|value| {
            matches!(value, Value::Tensor(tensor)
                if tensor.integer_storage().is_some_and(|storage| storage.numeric_dtype().class_name() == constructor))
        }));
        assert!(values.iter().any(|value| {
            matches!(value, Value::Tensor(tensor)
            if tensor.integer_storage().is_some_and(|storage| {
                storage.numeric_dtype().class_name() == constructor
                    && storage.len() == 2
                    && storage.value_at(0).is_some_and(|value| value.to_f64() == 1.0)
                    && storage.value_at(1).is_some_and(|value| value.to_f64() == 0.0)
            }))
        }));
    }
}

#[test]
fn compiled_fprintf_formats_wide_integer_data_without_float_conversion() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let values = execute_source(
        "base = uint64(9007199254740992); wide = [base + uint64(1), intmax('uint64')]; count = fprintf('%u %u', wide); retained = wide;",
    )
    .expect("compiled exact integer fprintf");
    assert!(values.iter().any(|value| {
        matches!(value, Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX])))
    }), "{values:?}");
    assert!(values
        .iter()
        .any(|value| matches!(value, Value::Num(count) if *count > 0.0)));
}

#[test]
fn compiled_integer_inapplicable_inputs_reject_without_coercion() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "value = func2str(uint8(1));",
            "RunMat:Func2StrHandleTypeInvalid",
        ),
        (
            "value = functions(uint8(1));",
            "RunMat:FunctionsHandleUnsupported",
        ),
    ] {
        let error = execute_source(source).expect_err("integer-inapplicable input must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
    let error = execute_source("value = getenv(uint8(1));")
        .expect_err("integer environment name must reject");
    assert!(
        error
            .to_string()
            .contains("NAME must be a character vector"),
        "{error}"
    );
}

#[test]
fn compiled_file_control_extensions_have_stable_strict_mode_identifiers() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "value = fopen(uint8(3));",
            "RunMat:compatibility:FopenIntegerIdExtension",
        ),
        (
            "count = fprintf(uint8(1), '%d', 7);",
            "RunMat:compatibility:FprintfIntegerIdExtension",
        ),
        (
            "value = fread(uint8(3));",
            "RunMat:compatibility:FreadIntegerIdExtension",
        ),
        (
            "frewind(uint8(3));",
            "RunMat:compatibility:FrewindIntegerIdExtension",
        ),
        (
            "count = fwrite(uint8(3), uint8(1));",
            "RunMat:compatibility:FwriteIntegerIdExtension",
        ),
        (
            "count = fprintf(single([37 100]), 7);",
            "RunMat:compatibility:FprintfNumericFormatExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict-mode extension must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}
