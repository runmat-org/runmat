#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::{IntegerStorage, Value};
use test_helpers::execute_source;

#[test]
fn compiled_text_introspection_preserves_wide_integer_passthrough() {
    let vars = execute_source(
        "a = uint64([9223372036854775808 18446744073709551615]); \
         name = class(a); \
         b = convertStringsToChars(a); \
         c = convertCharsToStrings(a); \
         d = convertContainedStringsToChars(a);",
    )
    .expect("execute compiled text/introspection cohort");

    let expected = IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]);
    assert!(vars
        .iter()
        .any(|value| value == &Value::String("uint64".into())));
    assert_eq!(
        vars.iter()
            .filter(|value| {
                matches!(value, Value::Tensor(tensor) if tensor.integer_storage() == Some(&expected))
            })
            .count(),
        4
    );
}

#[test]
fn compiled_integer_controls_and_rendering_sinks_dispatch_without_coercion() {
    let vars = execute_source(
        "lo = bitshift(uint64(1), 53); lo = lo + uint64(1); \
         hi = intmax(\"uint64\"); a = [lo hi]; \
         tf = contains(\"RunMat\", \"run\", \"IgnoreCase\", true); \
         st = dbstack(uint8(0)); \
         disp(a); \
         display(a);",
    )
    .expect("execute compiled integer control/sink cohort");
    assert!(vars.iter().any(|value| value == &Value::Bool(true)));
    assert!(vars.iter().any(|value| matches!(value, Value::Cell(_))));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX])))
    }));
}

#[test]
fn compiled_rendering_sinks_reject_requested_outputs() {
    for (source, expected) in [
        ("x = disp(uint8(1));", "does not return output"),
        ("x = display(uint8(1));", "does not return output"),
        (
            "f = memoize(@sqrt); x = clearCache(f);",
            "does not return output",
        ),
        ("[a,b,c] = dbstack;", "too many output arguments"),
    ] {
        let error = execute_source(source).expect_err("no-output sink assignment must reject");
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn compiled_contains_integer_extensions_obey_compatibility_mode() {
    {
        let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
        let vars = execute_source(
            "a = contains(\"RunMat\", \"run\", \"IgnoreCase\", int8(-1)); \
             b = contains(\"RunMat\", \"run\", int8(-1));",
        )
        .expect("RunMat integer IgnoreCase extensions");
        assert_eq!(
            vars.iter()
                .filter(|value| **value == Value::Bool(true))
                .count(),
            2
        );
    }
    {
        let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
        let error = execute_source("a = contains(\"RunMat\", \"run\", \"IgnoreCase\", int8(-1));")
            .expect_err("MATLAB mode rejects numeric IgnoreCase");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:ContainsNumericIgnoreCaseExtension")
        );
        let error = execute_source("a = contains(\"RunMat\", \"run\", int8(-1));")
            .expect_err("MATLAB mode rejects positional IgnoreCase");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:ContainsPositionalIgnoreCaseExtension")
        );
    }
}
