//! MATLAB-compatible `strtrim` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, StringArray, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::builtins::strings::type_resolvers::text_preserve_type;
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::strtrim")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strtrim",
    op_kind: GpuOpKind::Custom("string-transform"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Executes on the CPU; GPU-resident inputs are gathered to host memory before trimming whitespace.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::strings::transform::strtrim"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strtrim",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String transformation builtin; not eligible for fusion and always gathers GPU inputs.",
};

const BUILTIN_NAME: &str = "strtrim";
const ARG_TYPE_ERROR: &str =
    "strtrim: first argument must be a string array, character array, or cell array of character vectors";
const CELL_ELEMENT_ERROR: &str =
    "strtrim: cell array elements must be string scalars or character vectors";

const STRTRIM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Trimmed text preserving input container kind and shape.",
}];

const STRTRIM_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String/char/cell text input to trim.",
}];

const STRTRIM_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = strtrim(str)",
    inputs: &STRTRIM_INPUTS,
    outputs: &STRTRIM_OUTPUT,
}];

const STRTRIM_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRTRIM.INVALID_INPUT",
    identifier: Some("RunMat:strtrim:InvalidInput"),
    when: "Input is not a string array, character array, or cell array of text scalars.",
    message: ARG_TYPE_ERROR,
};

const STRTRIM_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRTRIM.CELL_ELEMENT",
    identifier: Some("RunMat:strtrim:CellElement"),
    when: "Cell array contains a non-text element or non-row char array element.",
    message: CELL_ELEMENT_ERROR,
};

const STRTRIM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRTRIM.INTERNAL",
    identifier: Some("RunMat:strtrim:InternalError"),
    when: "Internal output container construction failed.",
    message: "strtrim: internal error",
};

const STRTRIM_ERRORS: [BuiltinErrorDescriptor; 3] = [
    STRTRIM_ERROR_INVALID_INPUT,
    STRTRIM_ERROR_CELL_ELEMENT,
    STRTRIM_ERROR_INTERNAL,
];

pub const STRTRIM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRTRIM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STRTRIM_ERRORS,
};

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

fn strtrim_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "strtrim",
    category = "strings/transform",
    summary = "Remove leading and trailing whitespace from strings, character arrays, and cell arrays.",
    keywords = "strtrim,trim,whitespace,strings,character array,text",
    accel = "sink",
    type_resolver(text_preserve_type),
    descriptor(crate::builtins::strings::transform::strtrim::STRTRIM_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::strtrim"
)]
async fn strtrim_builtin(value: Value) -> BuiltinResult<Value> {
    let gathered = gather_if_needed_async(&value).await.map_err(map_flow)?;
    match gathered {
        Value::String(text) => Ok(Value::String(trim_string(text))),
        Value::StringArray(array) => strtrim_string_array(array),
        Value::CharArray(array) => strtrim_char_array(array),
        Value::Cell(cell) => strtrim_cell_array(cell).await,
        _ => Err(strtrim_error_with_message(
            ARG_TYPE_ERROR,
            &STRTRIM_ERROR_INVALID_INPUT,
        )),
    }
}

fn strtrim_string_array(array: StringArray) -> BuiltinResult<Value> {
    let StringArray { data, shape, .. } = array;
    let trimmed = data.into_iter().map(trim_string).collect::<Vec<_>>();
    let out = StringArray::new(trimmed, shape).map_err(|e| {
        strtrim_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRTRIM_ERROR_INTERNAL)
    })?;
    Ok(Value::StringArray(out))
}

fn strtrim_char_array(array: CharArray) -> BuiltinResult<Value> {
    let CharArray { data, rows, cols } = array;
    if rows == 0 {
        return Ok(Value::CharArray(CharArray { data, rows, cols }));
    }

    let mut trimmed_rows: Vec<Vec<char>> = Vec::with_capacity(rows);
    let mut target_cols: usize = 0;
    for row in 0..rows {
        let text = char_row_to_string_slice(&data, cols, row);
        let trimmed = trim_whitespace(&text);
        let chars: Vec<char> = trimmed.chars().collect();
        target_cols = target_cols.max(chars.len());
        trimmed_rows.push(chars);
    }

    let mut new_data: Vec<char> = Vec::with_capacity(rows * target_cols);
    for mut chars in trimmed_rows {
        if chars.len() < target_cols {
            chars.resize(target_cols, ' ');
        }
        new_data.extend(chars);
    }

    CharArray::new(new_data, rows, target_cols)
        .map(Value::CharArray)
        .map_err(|e| {
            strtrim_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRTRIM_ERROR_INTERNAL)
        })
}

async fn strtrim_cell_array(cell: CellArray) -> BuiltinResult<Value> {
    let CellArray {
        data, rows, cols, ..
    } = cell;
    let mut trimmed_values = Vec::with_capacity(rows * cols);
    for value in &data {
        let trimmed = strtrim_cell_element(value).await?;
        trimmed_values.push(trimmed);
    }
    make_cell(trimmed_values, rows, cols).map_err(|e| {
        strtrim_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRTRIM_ERROR_INTERNAL)
    })
}

async fn strtrim_cell_element(value: &Value) -> BuiltinResult<Value> {
    match gather_if_needed_async(value).await.map_err(map_flow)? {
        Value::String(text) => Ok(Value::String(trim_string(text))),
        Value::StringArray(sa) if sa.data.len() == 1 => {
            let text = sa.data.into_iter().next().unwrap();
            Ok(Value::String(trim_string(text)))
        }
        Value::CharArray(ca) if ca.rows <= 1 => {
            if ca.rows == 0 {
                return Ok(Value::CharArray(ca));
            }
            let source = char_row_to_string_slice(&ca.data, ca.cols, 0);
            let trimmed = trim_whitespace(&source);
            let chars: Vec<char> = trimmed.chars().collect();
            let cols = chars.len();
            CharArray::new(chars, ca.rows, cols)
                .map(Value::CharArray)
                .map_err(|e| {
                    strtrim_error_with_message(
                        format!("{BUILTIN_NAME}: {e}"),
                        &STRTRIM_ERROR_INTERNAL,
                    )
                })
        }
        Value::CharArray(_) => Err(strtrim_error_with_message(
            CELL_ELEMENT_ERROR,
            &STRTRIM_ERROR_CELL_ELEMENT,
        )),
        _ => Err(strtrim_error_with_message(
            CELL_ELEMENT_ERROR,
            &STRTRIM_ERROR_CELL_ELEMENT,
        )),
    }
}

fn trim_string(text: String) -> String {
    if is_missing_string(&text) {
        text
    } else {
        trim_whitespace(&text)
    }
}

fn trim_whitespace(text: &str) -> String {
    let trimmed = text.trim_matches(|c: char| c.is_whitespace());
    trimmed.to_string()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{ResolveContext, Type};

    fn run_strtrim(value: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(strtrim_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_string_scalar_trims_whitespace() {
        let result =
            run_strtrim(Value::String("  RunMat  ".into())).expect("strtrim string scalar");
        assert_eq!(result, Value::String("RunMat".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_string_array_preserves_shape() {
        let array = StringArray::new(
            vec![
                " one ".into(),
                "<missing>".into(),
                "two".into(),
                " three ".into(),
            ],
            vec![2, 2],
        )
        .unwrap();
        let result = run_strtrim(Value::StringArray(array)).expect("strtrim string array");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 2]);
                assert_eq!(
                    sa.data,
                    vec![
                        String::from("one"),
                        String::from("<missing>"),
                        String::from("two"),
                        String::from("three")
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_char_array_multiple_rows() {
        let data: Vec<char> = "  cat  ".chars().chain(" dog   ".chars()).collect();
        let array = CharArray::new(data, 2, 7).unwrap();
        let result = run_strtrim(Value::CharArray(array)).expect("strtrim char array");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 3);
                assert_eq!(ca.data, vec!['c', 'a', 't', 'd', 'o', 'g']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_char_array_all_whitespace_yields_zero_width() {
        let array = CharArray::new("   ".chars().collect(), 1, 3).unwrap();
        let result = run_strtrim(Value::CharArray(array)).expect("strtrim char whitespace");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 0);
                assert!(ca.data.is_empty());
            }
            other => panic!("expected empty char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_cell_array_mixed_content() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("  GPU  ")),
                Value::String(" Accelerate ".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = run_strtrim(Value::Cell(cell)).expect("strtrim cell array");
        match result {
            Value::Cell(out) => {
                let first = out.get(0, 0).unwrap();
                let second = out.get(0, 1).unwrap();
                assert_eq!(first, Value::CharArray(CharArray::new_row("GPU")));
                assert_eq!(second, Value::String("Accelerate".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_preserves_missing_strings() {
        let result =
            run_strtrim(Value::String("<missing>".into())).expect("strtrim missing string");
        assert_eq!(result, Value::String("<missing>".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_handles_tabs_and_newlines() {
        let input = Value::String("\tMetrics \n".into());
        let result = run_strtrim(input).expect("strtrim tab/newline");
        assert_eq!(result, Value::String("Metrics".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_trims_unicode_whitespace() {
        let input = Value::String("\u{00A0}RunMat\u{2003}".into());
        let result = run_strtrim(input).expect("strtrim unicode whitespace");
        assert_eq!(result, Value::String("RunMat".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_char_array_zero_rows_stable() {
        let array = CharArray::new(Vec::new(), 0, 0).unwrap();
        let result = run_strtrim(Value::CharArray(array.clone())).expect("strtrim 0x0 char");
        assert_eq!(result, Value::CharArray(array));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_cell_array_accepts_string_scalar() {
        let scalar = StringArray::new(vec![" padded ".into()], vec![1, 1]).unwrap();
        let cell = CellArray::new(vec![Value::StringArray(scalar)], 1, 1).unwrap();
        let trimmed = run_strtrim(Value::Cell(cell)).expect("strtrim cell string scalar");
        match trimmed {
            Value::Cell(out) => {
                let value = out.get(0, 0).expect("cell element");
                assert_eq!(value, Value::String("padded".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_cell_array_rejects_non_text() {
        let cell = CellArray::new(vec![Value::Num(5.0)], 1, 1).unwrap();
        let err = run_strtrim(Value::Cell(cell)).expect_err("strtrim cell non-text");
        assert!(err.to_string().contains("cell array elements"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_errors_on_invalid_input() {
        let err = run_strtrim(Value::Num(1.0)).unwrap_err();
        assert!(err.to_string().contains("strtrim"));
    }

    #[test]
    fn strtrim_type_preserves_text() {
        assert_eq!(
            text_preserve_type(&[Type::String], &ResolveContext::new(Vec::new())),
            Type::String
        );
    }
}
