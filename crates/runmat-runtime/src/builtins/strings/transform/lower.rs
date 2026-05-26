//! MATLAB-compatible `lower` builtin with GPU-aware semantics for RunMat.

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
use crate::builtins::strings::common::{char_row_to_string_slice, lowercase_preserving_missing};
use crate::builtins::strings::type_resolvers::text_preserve_type;
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::lower")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "lower",
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
        "Executes on the CPU; GPU-resident inputs are gathered to host memory before conversion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::lower")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "lower",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String transformation builtin; not eligible for fusion and always gathers GPU inputs.",
};

const BUILTIN_NAME: &str = "lower";

const LOWER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Lowercased text preserving input container kind and shape.",
}];

const LOWER_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String/char/cell text input to transform.",
}];

const LOWER_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = lower(str)",
    inputs: &LOWER_INPUTS,
    outputs: &LOWER_OUTPUT,
}];

const LOWER_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOWER.INVALID_INPUT",
    identifier: Some("RunMat:lower:InvalidInput"),
    when: "Input is not a string array, character array, or cell array of text scalars.",
    message:
        "lower: first argument must be a string array, character array, or cell array of character vectors",
};

const LOWER_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOWER.CELL_ELEMENT",
    identifier: Some("RunMat:lower:CellElement"),
    when: "Cell array contains a non-text element or non-row char array element.",
    message: "lower: cell array elements must be string scalars or character vectors",
};

const LOWER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOWER.INTERNAL",
    identifier: Some("RunMat:lower:InternalError"),
    when: "Internal output container construction failed.",
    message: "lower: internal error",
};

const LOWER_ERRORS: [BuiltinErrorDescriptor; 3] = [
    LOWER_ERROR_INVALID_INPUT,
    LOWER_ERROR_CELL_ELEMENT,
    LOWER_ERROR_INTERNAL,
];

pub const LOWER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOWER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOWER_ERRORS,
};

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

fn lower_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn lower_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    lower_error_with_message(error.message, error)
}

#[runtime_builtin(
    name = "lower",
    category = "strings/transform",
    summary = "Convert strings, character arrays, and cell arrays of character vectors to lowercase.",
    keywords = "lower,lowercase,strings,character array,text",
    accel = "sink",
    type_resolver(text_preserve_type),
    descriptor(crate::builtins::strings::transform::lower::LOWER_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::lower"
)]
async fn lower_builtin(value: Value) -> BuiltinResult<Value> {
    let gathered = gather_if_needed_async(&value).await.map_err(map_flow)?;
    match gathered {
        Value::String(text) => Ok(Value::String(lowercase_preserving_missing(text))),
        Value::StringArray(array) => lower_string_array(array),
        Value::CharArray(array) => lower_char_array(array),
        Value::Cell(cell) => lower_cell_array(cell),
        _ => Err(lower_error(&LOWER_ERROR_INVALID_INPUT)),
    }
}

fn lower_string_array(array: StringArray) -> BuiltinResult<Value> {
    let StringArray { data, shape, .. } = array;
    let lowered = data
        .into_iter()
        .map(lowercase_preserving_missing)
        .collect::<Vec<_>>();
    let lowered_array = StringArray::new(lowered, shape).map_err(|e| {
        lower_error_with_message(format!("{BUILTIN_NAME}: {e}"), &LOWER_ERROR_INTERNAL)
    })?;
    Ok(Value::StringArray(lowered_array))
}

fn lower_char_array(array: CharArray) -> BuiltinResult<Value> {
    let CharArray { data, rows, cols } = array;
    if rows == 0 || cols == 0 {
        return Ok(Value::CharArray(CharArray { data, rows, cols }));
    }

    let mut lowered_rows = Vec::with_capacity(rows);
    let mut target_cols = cols;
    for row in 0..rows {
        let text = char_row_to_string_slice(&data, cols, row).to_lowercase();
        let len = text.chars().count();
        target_cols = target_cols.max(len);
        lowered_rows.push(text);
    }

    let mut lowered_data = Vec::with_capacity(rows * target_cols);
    for row_text in lowered_rows {
        let mut chars: Vec<char> = row_text.chars().collect();
        if chars.len() < target_cols {
            chars.resize(target_cols, ' ');
        }
        lowered_data.extend(chars.into_iter());
    }

    CharArray::new(lowered_data, rows, target_cols)
        .map(Value::CharArray)
        .map_err(|e| {
            lower_error_with_message(format!("{BUILTIN_NAME}: {e}"), &LOWER_ERROR_INTERNAL)
        })
}

fn lower_cell_array(cell: CellArray) -> BuiltinResult<Value> {
    let CellArray {
        data, rows, cols, ..
    } = cell;
    let mut lowered_values = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let lowered = lower_cell_element(&data[idx])?;
            lowered_values.push(lowered);
        }
    }
    make_cell(lowered_values, rows, cols).map_err(|e| {
        lower_error_with_message(format!("{BUILTIN_NAME}: {e}"), &LOWER_ERROR_INTERNAL)
    })
}

fn lower_cell_element(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => Ok(Value::String(lowercase_preserving_missing(text.clone()))),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(Value::String(
            lowercase_preserving_missing(sa.data[0].clone()),
        )),
        Value::CharArray(ca) if ca.rows <= 1 => lower_char_array(ca.clone()),
        Value::CharArray(_) => Err(lower_error(&LOWER_ERROR_CELL_ELEMENT)),
        _ => Err(lower_error(&LOWER_ERROR_CELL_ELEMENT)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{ResolveContext, Type};

    fn run_lower(value: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(lower_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_string_scalar_value() {
        let result = run_lower(Value::String("RunMat".into())).expect("lower");
        assert_eq!(result, Value::String("runmat".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_string_array_preserves_shape() {
        let array = StringArray::new(
            vec![
                "GPU".into(),
                "ACCEL".into(),
                "<missing>".into(),
                "MiXeD".into(),
            ],
            vec![2, 2],
        )
        .unwrap();
        let result = run_lower(Value::StringArray(array)).expect("lower");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 2]);
                assert_eq!(
                    sa.data,
                    vec![
                        String::from("gpu"),
                        String::from("accel"),
                        String::from("<missing>"),
                        String::from("mixed")
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_char_array_multiple_rows() {
        let data: Vec<char> = vec!['C', 'A', 'T', 'D', 'O', 'G'];
        let array = CharArray::new(data, 2, 3).unwrap();
        let result = run_lower(Value::CharArray(array)).expect("lower");
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
    fn lower_char_vector_handles_padding() {
        let array = CharArray::new_row("HELLO ");
        let result = run_lower(Value::CharArray(array)).expect("lower");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 6);
                let expected: Vec<char> = "hello ".chars().collect();
                assert_eq!(ca.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_char_array_unicode_expansion_extends_width() {
        let data: Vec<char> = vec!['İ', 'A'];
        let array = CharArray::new(data, 1, 2).unwrap();
        let result = run_lower(Value::CharArray(array)).expect("lower");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 3);
                let expected: Vec<char> = vec!['i', '\u{307}', 'a'];
                assert_eq!(ca.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_cell_array_mixed_content() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("RUN")),
                Value::String("Mat".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = run_lower(Value::Cell(cell)).expect("lower");
        match result {
            Value::Cell(out) => {
                let first = out.get(0, 0).unwrap();
                let second = out.get(0, 1).unwrap();
                assert_eq!(first, Value::CharArray(CharArray::new_row("run")));
                assert_eq!(second, Value::String("mat".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_errors_on_invalid_input() {
        let err = run_lower(Value::Num(1.0)).unwrap_err();
        assert_eq!(err.to_string(), LOWER_ERROR_INVALID_INPUT.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_cell_errors_on_invalid_element() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err = run_lower(Value::Cell(cell)).unwrap_err();
        assert_eq!(err.to_string(), LOWER_ERROR_CELL_ELEMENT.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_preserves_missing_string() {
        let result = run_lower(Value::String("<missing>".into())).expect("lower");
        assert_eq!(result, Value::String("<missing>".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_cell_allows_empty_char_vector() {
        let empty_char = CharArray::new(Vec::new(), 1, 0).unwrap();
        let cell = CellArray::new(vec![Value::CharArray(empty_char.clone())], 1, 1).unwrap();
        let result = run_lower(Value::Cell(cell)).expect("lower");
        match result {
            Value::Cell(out) => {
                let element = out.get(0, 0).unwrap();
                assert_eq!(element, Value::CharArray(empty_char));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn lower_gpu_tensor_input_gathers_then_errors() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let data = [1.0f64, 2.0];
        let shape = [2usize, 1usize];
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &data,
                shape: &shape,
            })
            .expect("upload");
        let err = run_lower(Value::GpuTensor(handle.clone())).unwrap_err();
        assert_eq!(err.to_string(), LOWER_ERROR_INVALID_INPUT.message);
        provider.free(&handle).ok();
    }

    #[test]
    fn lower_type_preserves_text() {
        assert_eq!(
            text_preserve_type(&[Type::String], &ResolveContext::new(Vec::new())),
            Type::String
        );
    }
}
