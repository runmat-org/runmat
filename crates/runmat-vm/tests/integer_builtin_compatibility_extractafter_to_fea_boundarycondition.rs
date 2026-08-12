#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::{IntegerStorage, Value};
use test_helpers::execute_source;

const INTEGER_CONSTRUCTORS: [&str; 8] = [
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
];

fn contains_text(values: &[Value], expected: &str) -> bool {
    values.iter().any(|value| match value {
        Value::String(text) => text == expected,
        Value::StringArray(strings) => strings.data.iter().any(|text| text == expected),
        Value::CharArray(chars) => chars.data.iter().collect::<String>() == expected,
        _ => false,
    })
}

#[test]
fn compiled_extract_boundaries_accept_every_integer_position_class() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            r#"
            before = extractBefore("abcdef", {constructor}(4));
            after = extractAfter("abcdef", {constructor}(3));
            between = extractBetween("abcdef", {constructor}(2), {constructor}(4));
            "#
        );
        let values = execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor} compiled extraction: {error}"));
        assert!(contains_text(&values, "abc"), "{constructor} extractBefore");
        assert!(contains_text(&values, "def"), "{constructor} extractAfter");
        assert!(
            contains_text(&values, "bcd"),
            "{constructor} extractBetween"
        );
    }
}

#[test]
fn compiled_eye_and_false_accept_every_integer_dimension_class() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            r#"
            identity = eye({constructor}(2), "uint64");
            mask = false({constructor}([2 3]));
            "#
        );
        let values = execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor} compiled creation: {error}"));
        assert!(values.iter().any(|value| {
            matches!(value, Value::Tensor(tensor)
                if tensor.shape == [2, 2]
                    && tensor.integer_storage()
                        == Some(&IntegerStorage::U64(vec![1, 0, 0, 1])))
        }));
        assert!(values.iter().any(|value| {
            matches!(value, Value::LogicalArray(array)
                if array.shape == [2, 3] && array.data == vec![0; 6])
        }));
    }
}

#[test]
fn compiled_fcontour_and_boundary_condition_accept_every_integer_class() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            r#"
            handle = fcontour(@(x,y) x + y, {constructor}([0 1 0 1]), ...
                "MeshDensity", {constructor}(3), ...
                "LevelList", {constructor}([1 0]), ...
                "Fill", {constructor}(1));
            boundary = fea.boundaryCondition("wall", "face", "thermalHeatFlux", ...
                "heatFluxWPerM2", {constructor}(2));
            "#
        );
        let values = execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor} compiled plot/FEA: {error}"));
        assert!(values.iter().any(|value| matches!(value, Value::Num(_))));
        assert!(values.iter().any(|value| {
            matches!(value, Value::Object(object)
                if object.class_name == "fea.BoundaryCondition")
        }));
    }
}

#[test]
fn compiled_integer_inapplicable_text_roles_reject_without_coercion() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "value = extractFileText(uint8(1));",
            "RunMat:extractFileText:InvalidInput",
        ),
        (
            "value = extractHTMLText(uint8(1));",
            "RunMat:html:InvalidInput",
        ),
    ] {
        let error = execute_source(source).expect_err("integer text role must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_runmat_only_forms_have_stable_strict_mode_identifiers() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "status = fclose(uint8(3));",
            "RunMat:compatibility:FcloseIntegerIdExtension",
        ),
        (
            "value = eye(uint8([2 2 2]));",
            "RunMat:compatibility:EyeNdDimensionsExtension",
        ),
        (
            "value = false(single([2 2]));",
            "RunMat:compatibility:FalseSingleSizeExtension",
        ),
        (
            "handle = fcontour(@(x,y) x + y, [0 1], uint8(3));",
            "RunMat:compatibility:FcontourPositionalLevelSpecExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict-mode extension must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}
