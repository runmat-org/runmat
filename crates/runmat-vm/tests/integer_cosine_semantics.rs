#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::Value;
use test_helpers::execute_source;

#[test]
fn compiled_cosine_family_integer_extensions_cover_all_classes() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "a = cos({constructor}(0)); b = cosd({constructor}(0)); c = cosh({constructor}(0)); d = cospi({constructor}(1));"
        );
        let values = execute_source(&source).expect("compiled integer cosine family");
        assert_eq!(
            values
                .iter()
                .filter(|value| matches!(value, Value::Num(number) if *number == 1.0))
                .count(),
            3
        );
        assert!(values.contains(&Value::Num(-1.0)));
    }
}

#[test]
fn compiled_cospi_keeps_wide_uint64_parity_exact() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let values = execute_source("x = cospi(intmax(\"uint64\"));").expect("wide cospi");
    assert!(values.contains(&Value::Num(-1.0)));
}

#[test]
fn compiled_integer_cosine_extensions_obey_matlab_mode() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "x = cos(uint8(0));",
            "RunMat:compatibility:CosIntegerInputExtension",
        ),
        (
            "x = cosd(uint8(0));",
            "RunMat:compatibility:CosdIntegerInputExtension",
        ),
        (
            "x = cosh(uint8(0));",
            "RunMat:compatibility:CoshIntegerInputExtension",
        ),
        (
            "x = cospi(uint8(0));",
            "RunMat:compatibility:CospiIntegerInputExtension",
        ),
        (
            "x = cosineSimilarity(uint8([1 0]));",
            "RunMat:compatibility:CosineSimilarityIntegerMatrixExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("MATLAB mode integer gate");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_cos_validates_like_grammar_before_matlab_mode_gate() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let error = execute_source("x = cos(uint8(0), 'like');").expect_err("malformed like form");
    assert_eq!(error.identifier(), Some("RunMat:cos:InvalidOption"));
}

#[test]
fn compiled_cosine_similarity_accepts_integer_matrix_in_runmat_mode() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source("S = cosineSimilarity(uint16([1 0; 0 1]));")
        .expect("compiled integer cosineSimilarity");
}
