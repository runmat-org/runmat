#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::{IntegerStorage, Value};
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
    let source = "lo = bitshift(uint64(1), 53); lo = lo + uint64(1); \
         hi = intmax(\"uint64\"); a = [lo hi]; \
         tf = contains(\"RunMat\", \"run\", \"IgnoreCase\", true); \
         st = dbstack(uint8(0)); \
         disp(a); \
         display(a);";
    let bytecode =
        test_helpers::compile_source(source).expect("compile integer control/sink cohort");
    let vars =
        test_helpers::interpret(&bytecode).expect("execute compiled integer control/sink cohort");
    assert!(vars.iter().any(|value| value == &Value::Bool(true)));
    let stack_slot = bytecode
        .var_names
        .iter()
        .find_map(|(&slot, name)| (name == "st").then_some(slot))
        .expect("authored dbstack output slot");
    let validate_frame = |frame: &runmat_value::StructValue| {
        let mut fields = frame.field_names().collect::<Vec<_>>();
        fields.sort_unstable();
        assert_eq!(fields, ["file", "line", "name"]);
        assert!(matches!(frame.fields.get("file"), Some(Value::String(_))));
        assert!(matches!(frame.fields.get("name"), Some(Value::String(_))));
        assert!(matches!(frame.fields.get("line"), Some(Value::Num(line)) if *line >= 0.0));
    };
    match &vars[stack_slot] {
        Value::Struct(frame) => validate_frame(frame),
        Value::StructArray(stack) => {
            assert_eq!(stack.shape().len(), 2);
            assert_eq!(stack.shape()[1], 1);
            let mut fields = stack.field_names().collect::<Vec<_>>();
            fields.sort_unstable();
            assert_eq!(fields, ["file", "line", "name"]);
            for frame in stack.elements() {
                validate_frame(&frame.to_owned());
            }
        }
        other => panic!("dbstack must return its structured schema, got {other:?}"),
    }
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
