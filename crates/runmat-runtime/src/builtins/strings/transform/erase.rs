//! MATLAB-compatible `erase` builtin with GPU-aware semantics for RunMat.
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
use crate::{
    build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult, RuntimeError,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::erase")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "erase",
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
        "Executes on the CPU; GPU-resident inputs are gathered to host memory before substrings are removed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::erase")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "erase",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "String manipulation builtin; not eligible for fusion plans and always gathers GPU inputs before execution.",
};

const BUILTIN_NAME: &str = "erase";

const ERASE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newStr",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text with substring occurrences removed, preserving input container kind.",
}];

const ERASE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "pattern",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text list (scalar or array/cell).",
    },
];

const ERASE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "newStr = erase(str, pattern)",
    inputs: &ERASE_INPUTS,
    outputs: &ERASE_OUTPUT,
}];

const ERASE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE.INVALID_INPUT",
    identifier: Some("RunMat:erase:InvalidInput"),
    when: "First argument is not a string array, char array, or cell array of text scalars.",
    message:
        "erase: first argument must be a string array, character array, or cell array of character vectors",
};

const ERASE_ERROR_PATTERN_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE.PATTERN_TYPE",
    identifier: Some("RunMat:erase:PatternType"),
    when: "Second argument is not a text scalar/array/cell of text scalars.",
    message:
        "erase: second argument must be a string array, character array, or cell array of character vectors",
};

const ERASE_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE.CELL_ELEMENT",
    identifier: Some("RunMat:erase:CellElement"),
    when: "Cell arrays contain non-text elements or non-row char arrays.",
    message: "erase: cell array elements must be string scalars or character vectors",
};

const ERASE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE.INTERNAL",
    identifier: Some("RunMat:erase:InternalError"),
    when: "Internal output container construction failed.",
    message: "erase: internal error",
};

const ERASE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    ERASE_ERROR_INVALID_INPUT,
    ERASE_ERROR_PATTERN_TYPE,
    ERASE_ERROR_CELL_ELEMENT,
    ERASE_ERROR_INTERNAL,
];

pub const ERASE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ERASE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERASE_ERRORS,
};

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

fn erase_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn erase_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    erase_error_with_message(error.message, error)
}

#[runtime_builtin(
    name = "erase",
    category = "strings/transform",
    summary = "Remove substring occurrences from text inputs.",
    keywords = "erase,remove substring,strings,character array,text",
    accel = "sink",
    type_resolver(text_preserve_type),
    descriptor(crate::builtins::strings::transform::erase::ERASE_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::erase"
)]
async fn erase_builtin(text: Value, pattern: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text).await.map_err(map_flow)?;
    let pattern = gather_if_needed_async(&pattern).await.map_err(map_flow)?;

    let patterns = PatternList::from_value(&pattern)?;

    match text {
        Value::String(s) => Ok(Value::String(erase_string_scalar(s, &patterns))),
        Value::StringArray(sa) => erase_string_array(sa, &patterns),
        Value::CharArray(ca) => erase_char_array(ca, &patterns),
        Value::Cell(cell) => erase_cell_array(cell, &patterns),
        _ => Err(erase_error(&ERASE_ERROR_INVALID_INPUT)),
    }
}

struct PatternList {
    entries: Vec<String>,
}

impl PatternList {
    fn from_value(value: &Value) -> BuiltinResult<Self> {
        let entries = match value {
            Value::String(text) => vec![text.clone()],
            Value::StringArray(array) => array.data.clone(),
            Value::CharArray(array) => {
                if array.rows == 0 {
                    Vec::new()
                } else {
                    let mut list = Vec::with_capacity(array.rows);
                    for row in 0..array.rows {
                        list.push(char_row_to_string_slice(&array.data, array.cols, row));
                    }
                    list
                }
            }
            Value::Cell(cell) => {
                let mut list = Vec::with_capacity(cell.data.len());
                for handle in &cell.data {
                    match &handle {
                        Value::String(text) => list.push(text.clone()),
                        Value::StringArray(sa) if sa.data.len() == 1 => {
                            list.push(sa.data[0].clone());
                        }
                        Value::CharArray(ca) if ca.rows == 0 => list.push(String::new()),
                        Value::CharArray(ca) if ca.rows == 1 => {
                            list.push(char_row_to_string_slice(&ca.data, ca.cols, 0));
                        }
                        Value::CharArray(_) => return Err(erase_error(&ERASE_ERROR_CELL_ELEMENT)),
                        _ => return Err(erase_error(&ERASE_ERROR_CELL_ELEMENT)),
                    }
                }
                list
            }
            _ => return Err(erase_error(&ERASE_ERROR_PATTERN_TYPE)),
        };
        Ok(Self { entries })
    }

    fn apply(&self, input: &str) -> String {
        if self.entries.is_empty() {
            return input.to_string();
        }
        let mut current = input.to_string();
        for pattern in &self.entries {
            if pattern.is_empty() {
                continue;
            }
            if current.is_empty() {
                break;
            }
            current = current.replace(pattern, "");
        }
        current
    }
}

fn erase_string_scalar(text: String, patterns: &PatternList) -> String {
    if is_missing_string(&text) {
        text
    } else {
        patterns.apply(&text)
    }
}

fn erase_string_array(array: StringArray, patterns: &PatternList) -> BuiltinResult<Value> {
    let StringArray { data, shape, .. } = array;
    let mut erased = Vec::with_capacity(data.len());
    for entry in data {
        if is_missing_string(&entry) {
            erased.push(entry);
        } else {
            erased.push(patterns.apply(&entry));
        }
    }
    StringArray::new(erased, shape)
        .map(Value::StringArray)
        .map_err(|e| {
            erase_error_with_message(format!("{BUILTIN_NAME}: {e}"), &ERASE_ERROR_INTERNAL)
        })
}

fn erase_char_array(array: CharArray, patterns: &PatternList) -> BuiltinResult<Value> {
    let CharArray { data, rows, cols } = array;
    if rows == 0 {
        return Ok(Value::CharArray(CharArray { data, rows, cols }));
    }

    let mut processed: Vec<String> = Vec::with_capacity(rows);
    let mut target_cols = 0usize;
    for row in 0..rows {
        let slice = char_row_to_string_slice(&data, cols, row);
        let erased = patterns.apply(&slice);
        let len = erased.chars().count();
        if len > target_cols {
            target_cols = len;
        }
        processed.push(erased);
    }

    let mut flattened: Vec<char> = Vec::with_capacity(rows * target_cols);
    for row_text in processed {
        let mut chars: Vec<char> = row_text.chars().collect();
        if chars.len() < target_cols {
            chars.resize(target_cols, ' ');
        }
        flattened.extend(chars);
    }

    CharArray::new(flattened, rows, target_cols)
        .map(Value::CharArray)
        .map_err(|e| {
            erase_error_with_message(format!("{BUILTIN_NAME}: {e}"), &ERASE_ERROR_INTERNAL)
        })
}

fn erase_cell_array(cell: CellArray, patterns: &PatternList) -> BuiltinResult<Value> {
    let shape = cell.shape.clone();
    let mut values = Vec::with_capacity(cell.data.len());
    for handle in &cell.data {
        values.push(erase_cell_element(handle, patterns)?);
    }
    make_cell_with_shape(values, shape).map_err(|e| {
        erase_error_with_message(format!("{BUILTIN_NAME}: {e}"), &ERASE_ERROR_INTERNAL)
    })
}

fn erase_cell_element(value: &Value, patterns: &PatternList) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => Ok(Value::String(erase_string_scalar(text.clone(), patterns))),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(Value::String(erase_string_scalar(
            sa.data[0].clone(),
            patterns,
        ))),
        Value::CharArray(ca) if ca.rows == 0 => Ok(Value::CharArray(ca.clone())),
        Value::CharArray(ca) if ca.rows == 1 => {
            let slice = char_row_to_string_slice(&ca.data, ca.cols, 0);
            let erased = patterns.apply(&slice);
            Ok(Value::CharArray(CharArray::new_row(&erased)))
        }
        Value::CharArray(_) => Err(erase_error(&ERASE_ERROR_CELL_ELEMENT)),
        _ => Err(erase_error(&ERASE_ERROR_CELL_ELEMENT)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{ResolveContext, Type};

    fn erase_builtin(text: Value, pattern: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(super::erase_builtin(text, pattern))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_string_scalar_single_pattern() {
        let result = erase_builtin(
            Value::String("RunMat runtime".into()),
            Value::String(" runtime".into()),
        )
        .expect("erase");
        assert_eq!(result, Value::String("RunMat".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_string_array_multiple_patterns() {
        let strings = StringArray::new(
            vec!["gpu".into(), "cpu".into(), "<missing>".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = erase_builtin(
            Value::StringArray(strings),
            Value::StringArray(StringArray::new(vec!["g".into(), "c".into()], vec![2, 1]).unwrap()),
        )
        .expect("erase");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![3, 1]);
                assert_eq!(
                    sa.data,
                    vec![
                        String::from("pu"),
                        String::from("pu"),
                        String::from("<missing>")
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_string_array_shape_mismatch_applies_all_patterns() {
        let strings =
            StringArray::new(vec!["GPU kernel".into(), "CPU kernel".into()], vec![2, 1]).unwrap();
        let patterns = StringArray::new(vec!["GPU ".into(), "CPU ".into()], vec![1, 2]).unwrap();
        let result = erase_builtin(Value::StringArray(strings), Value::StringArray(patterns))
            .expect("erase");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 1]);
                assert_eq!(
                    sa.data,
                    vec![String::from("kernel"), String::from("kernel")]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_char_array_adjusts_width() {
        let chars = CharArray::new("matrix".chars().collect(), 1, 6).unwrap();
        let result =
            erase_builtin(Value::CharArray(chars), Value::String("tr".into())).expect("erase");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 4);
                let expected: Vec<char> = "maix".chars().collect();
                assert_eq!(out.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_char_array_handles_full_removal() {
        let chars = CharArray::new_row("abc");
        let result = erase_builtin(Value::CharArray(chars.clone()), Value::String("abc".into()))
            .expect("erase");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 0);
                assert!(out.data.is_empty());
            }
            other => panic!("expected empty char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_char_array_multiple_rows_sequential_patterns() {
        let chars = CharArray::new(
            vec![
                'G', 'P', 'U', ' ', 'p', 'i', 'p', 'e', 'l', 'i', 'n', 'e', 'C', 'P', 'U', ' ',
                'p', 'i', 'p', 'e', 'l', 'i', 'n', 'e',
            ],
            2,
            12,
        )
        .unwrap();
        let patterns = CharArray::new_row("GPU ");
        let result =
            erase_builtin(Value::CharArray(chars), Value::CharArray(patterns)).expect("erase");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 12);
                let first = char_row_to_string_slice(&out.data, out.cols, 0);
                let second = char_row_to_string_slice(&out.data, out.cols, 1);
                assert_eq!(first.trim_end(), "pipeline");
                assert_eq!(second.trim_end(), "CPU pipeline");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_cell_array_mixed_content() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("Kernel Planner")),
                Value::String("GPU Fusion".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = erase_builtin(
            Value::Cell(cell),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::String("Kernel ".into()),
                        Value::String("GPU ".into()),
                    ],
                    1,
                    2,
                )
                .unwrap(),
            ),
        )
        .expect("erase");
        match result {
            Value::Cell(out) => {
                let first = out.get(0, 0).unwrap();
                let second = out.get(0, 1).unwrap();
                assert_eq!(first, Value::CharArray(CharArray::new_row("Planner")));
                assert_eq!(second, Value::String("Fusion".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_cell_array_preserves_shape() {
        let cell = CellArray::new(
            vec![
                Value::String("alpha".into()),
                Value::String("beta".into()),
                Value::String("gamma".into()),
                Value::String("delta".into()),
            ],
            2,
            2,
        )
        .unwrap();
        let patterns = StringArray::new(vec!["a".into()], vec![1, 1]).unwrap();
        let result = erase_builtin(Value::Cell(cell), Value::StringArray(patterns)).expect("erase");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 2);
                assert_eq!(out.get(0, 0).unwrap(), Value::String("lph".into()));
                assert_eq!(out.get(1, 1).unwrap(), Value::String("delt".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_preserves_missing_string() {
        let result = erase_builtin(
            Value::String("<missing>".into()),
            Value::String("missing".into()),
        )
        .expect("erase");
        assert_eq!(result, Value::String("<missing>".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_allows_empty_pattern_list() {
        let strings = StringArray::new(vec!["alpha".into(), "beta".into()], vec![2, 1]).unwrap();
        let pattern = StringArray::new(Vec::<String>::new(), vec![0, 0]).unwrap();
        let result = erase_builtin(
            Value::StringArray(strings.clone()),
            Value::StringArray(pattern),
        )
        .expect("erase");
        assert_eq!(result, Value::StringArray(strings));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_errors_on_invalid_first_argument() {
        let err = erase_builtin(Value::Num(1.0), Value::String("a".into())).unwrap_err();
        assert_eq!(err.to_string(), ERASE_ERROR_INVALID_INPUT.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_errors_on_invalid_pattern_type() {
        let err = erase_builtin(Value::String("abc".into()), Value::Num(1.0)).unwrap_err();
        assert_eq!(err.to_string(), ERASE_ERROR_PATTERN_TYPE.message);
    }

    #[test]
    fn erase_type_preserves_text() {
        assert_eq!(
            text_preserve_type(&[Type::String], &ResolveContext::new(Vec::new())),
            Type::String
        );
    }
}
