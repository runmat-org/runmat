#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::{IntegerStorage, Value};
use test_helpers::execute_source;

const INTEGER_CONSTRUCTORS: [&str; 8] = [
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
];

fn expected_empty_storage(constructor: &str) -> IntegerStorage {
    match constructor {
        "int8" => IntegerStorage::I8(Vec::new()),
        "int16" => IntegerStorage::I16(Vec::new()),
        "int32" => IntegerStorage::I32(Vec::new()),
        "int64" => IntegerStorage::I64(Vec::new()),
        "uint8" => IntegerStorage::U8(Vec::new()),
        "uint16" => IntegerStorage::U16(Vec::new()),
        "uint32" => IntegerStorage::U32(Vec::new()),
        "uint64" => IntegerStorage::U64(Vec::new()),
        _ => unreachable!("known integer constructor"),
    }
}

fn contains_logical(values: &[Value], shape: &[usize], expected: &[u8]) -> bool {
    values.iter().any(|value| {
        matches!(value, Value::LogicalArray(array)
            if array.shape == shape && array.data == expected)
    })
}

fn contains_text(values: &[Value], expected: &str) -> bool {
    values.iter().any(|value| match value {
        Value::String(text) => text == expected,
        Value::CharArray(chars) => chars.data.iter().collect::<String>() == expected,
        _ => false,
    })
}

#[test]
fn compiled_ecdf_eig_and_eigs_accept_every_integer_class() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            r#"
            [f, x] = ecdf({constructor}([1 2 2]));
            d = eig({constructor}([2 0; 0 3]));
            selected = eigs({constructor}([2 0; 0 5]), {constructor}(1));
            if numel(f) ~= 3 || f(1) ~= 0 || f(3) ~= 1 || numel(x) ~= 3
                error('ecdf integer semantics mismatch');
            end
            if numel(d) ~= 2 || d(1) ~= 2 || d(2) ~= 3
                error('eig integer semantics mismatch');
            end
            if numel(selected) ~= 1 || selected(1) ~= 5
                error('eigs integer semantics mismatch');
            end
            "#
        );
        execute_source(&source).unwrap_or_else(|error| {
            panic!("{constructor} compiled numeric cohort failed: {error}")
        });
    }
}

#[test]
fn compiled_static_and_global_empty_preserve_integer_storage() {
    let matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!("value = {constructor}.empty({constructor}(0), uint8(3));");
        let values = execute_source(&source).expect("compiled ClassName.empty semantics");
        let expected = expected_empty_storage(constructor);
        assert!(
            values.iter().any(|value| {
                matches!(value, Value::Tensor(tensor)
                    if tensor.shape == [0, 3]
                        && tensor.integer_storage() == Some(&expected))
            }),
            "{constructor}.empty result"
        );
    }

    drop(matlab);
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let values = execute_source("value = empty(uint8(0), uint8(4), 'uint16');")
        .expect("compiled global empty shorthand");
    assert!(values.iter().any(|value| {
        matches!(value, Value::Tensor(tensor)
            if tensor.shape == [0, 4]
                && tensor.integer_storage() == Some(&IntegerStorage::U16(Vec::new())))
    }));
}

#[test]
fn compiled_encode_and_endswith_cover_integer_flags_and_any_pattern_matching() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            r#"
            bag = bagOfWords(["alpha" "beta"]);
            encoded = encode(bag, "alpha", "ForceCellOutput", {constructor}(1));
            matched = endsWith(["data.GZ" "code.m"], [".xlsx" ".gz"], "IgnoreCase", {constructor}(1));
            "#
        );
        let values = execute_source(&source).expect("compiled text integer-control semantics");
        assert!(
            values.iter().any(|value| matches!(value, Value::Cell(_))),
            "{constructor} encode ForceCellOutput"
        );
        assert!(
            contains_logical(&values, &[1, 2], &[1, 0]),
            "{constructor} endsWith alternatives"
        );
    }
}

#[test]
fn compiled_envelope_and_equality_preserve_integer_boundary_semantics() {
    let runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            r#"
            [upper, lower] = envelope({constructor}([0 3 4 0]), {constructor}(3), 'rms');
            if numel(upper) ~= 4 || numel(lower) ~= 4 || upper(2) <= 0 || lower(2) >= 0
                error('envelope integer semantics mismatch');
            end
            "#
        );
        execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor} envelope failed: {error}"));
    }

    drop(runmat);
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let values = execute_source(
        r#"
        base = uint64(9007199254740992);
        wide = base + uint64(1);
        exact = [eq(wide, base) eq(wide, wide)];
        "#,
    )
    .expect("compiled exact wide equality");
    assert!(contains_logical(&values, &[1, 2], &[0, 1]));
}

#[test]
fn compiled_erase_rejects_numeric_roles_and_erasebetween_accepts_typed_positions() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        ("value = erase(uint8(1), '1');", "RunMat:erase:InvalidInput"),
        (
            "value = erase('abc', uint8(1));",
            "RunMat:erase:PatternType",
        ),
    ] {
        let error = execute_source(source).expect_err("numeric erase role must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }

    for constructor in INTEGER_CONSTRUCTORS {
        let source =
            format!("value = eraseBetween(\"abcdef\", {constructor}(2), {constructor}(5));");
        let values = execute_source(&source).expect("compiled typed eraseBetween positions");
        assert!(contains_text(&values, "af"), "{constructor} positions");
    }
}

#[test]
fn compiled_ecdf_to_erasebetween_extensions_have_stable_strict_mode_identifiers() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "[f,x] = ecdf(uint8([1 2]));",
            "RunMat:compatibility:EcdfIntegerYExtension",
        ),
        (
            "[f,x] = ecdf([1 2], 'Frequency', uint8([1 1]));",
            "RunMat:compatibility:EcdfIntegerFrequencyExtension",
        ),
        (
            "[f,x] = ecdf([1 2], 'Censoring', uint8([0 1]));",
            "RunMat:compatibility:EcdfIntegerCensoringExtension",
        ),
        (
            "value = eig(uint8([1 0; 0 2]));",
            "RunMat:compatibility:EigNonfloatingCoefficientExtension",
        ),
        (
            "value = eigs(uint8([1 0; 0 2]));",
            "RunMat:compatibility:EigsNonfloatingMatrixExtension",
        ),
        (
            "value = eigs([1 0; 0 2], 1, uint8(2));",
            "RunMat:compatibility:EigsIntegerSigmaExtension",
        ),
        (
            "opts = struct('StartVector', uint8([1; 1])); value = eigs([1 0; 0 2], 1, opts);",
            "RunMat:compatibility:EigsIntegerStartVectorExtension",
        ),
        (
            "value = empty(uint8(0), 'uint8');",
            "RunMat:compatibility:EmptyGlobalCallExtension",
        ),
        (
            "bag = bagOfWords([\"alpha\"]); value = encode(bag, \"alpha\", \"ForceCellOutput\", uint8(1));",
            "RunMat:compatibility:EncodeNumericForceCellOutputExtension",
        ),
        (
            "value = endsWith(\"RunMat\", \"mat\", \"IgnoreCase\", uint8(1));",
            "RunMat:compatibility:EndsWithNumericIgnoreCaseExtension",
        ),
        (
            "value = envelope(uint8([1 2 3]));",
            "RunMat:compatibility:EnvelopeIntegerDataExtension",
        ),
        (
            "value = envelope([1 2 3], uint8(3));",
            "RunMat:compatibility:EnvelopeIntegerControlExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict-mode extension rejection");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}
