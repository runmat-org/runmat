//! MATLAB-compatible `strrep` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{CellArray, CharArray, StringArray, Value};
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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::strrep")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strrep",
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
    notes: "Executes on the CPU; GPU-resident inputs are gathered before replacements are applied.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::strrep")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strrep",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String transformation builtin; marked as a sink so fusion skips GPU residency.",
};

const BUILTIN_NAME: &str = "strrep";
const ARGUMENT_TYPE_ERROR: &str =
    "strrep: first argument must be a string array, character array, or cell array of character vectors";
const PATTERN_TYPE_ERROR: &str = "strrep: old and new must be string scalars or character vectors";
const PATTERN_MISMATCH_ERROR: &str = "strrep: old and new must be the same data type";
const CELL_ELEMENT_ERROR: &str =
    "strrep: cell array elements must be string scalars or character vectors";

#[derive(Clone, Copy, PartialEq, Eq)]
enum PatternKind {
    String,
    Char,
}

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "strrep",
    category = "strings/transform",
    summary = "Replace substring occurrences with MATLAB-compatible semantics.",
    keywords = "strrep,replace,strings,character array,text",
    accel = "sink",
    type_resolver(text_preserve_type),
    builtin_path = "crate::builtins::strings::transform::strrep"
)]
async fn strrep_builtin(
    str_value: Value,
    old_value: Value,
    new_value: Value,
) -> BuiltinResult<Value> {
    let gathered_str = gather_if_needed_async(&str_value).await.map_err(map_flow)?;
    let gathered_old = gather_if_needed_async(&old_value).await.map_err(map_flow)?;
    let gathered_new = gather_if_needed_async(&new_value).await.map_err(map_flow)?;

    let (old_text, old_kind) = parse_pattern(gathered_old)?;
    let (new_text, new_kind) = parse_pattern(gathered_new)?;
    if old_kind != new_kind {
        return Err(runtime_error_for(PATTERN_MISMATCH_ERROR));
    }

    match gathered_str {
        Value::String(text) => Ok(Value::String(strrep_string_value(
            text, &old_text, &new_text,
        ))),
        Value::StringArray(array) => strrep_string_array(array, &old_text, &new_text),
        Value::CharArray(array) => strrep_char_array(array, &old_text, &new_text),
        Value::Cell(cell) => strrep_cell_array(cell, &old_text, &new_text),
        _ => Err(runtime_error_for(ARGUMENT_TYPE_ERROR)),
    }
}

fn parse_pattern(value: Value) -> BuiltinResult<(String, PatternKind)> {
    match value {
        Value::String(text) => Ok((text, PatternKind::String)),
        Value::StringArray(array) => {
            if array.data.len() == 1 {
                Ok((array.data[0].clone(), PatternKind::String))
            } else {
                Err(runtime_error_for(PATTERN_TYPE_ERROR))
            }
        }
        Value::CharArray(array) => {
            if array.rows <= 1 {
                let text = if array.rows == 0 {
                    String::new()
                } else {
                    char_row_to_string_slice(&array.data, array.cols, 0)
                };
                Ok((text, PatternKind::Char))
            } else {
                Err(runtime_error_for(PATTERN_TYPE_ERROR))
            }
        }
        _ => Err(runtime_error_for(PATTERN_TYPE_ERROR)),
    }
}

fn strrep_string_value(text: String, old: &str, new: &str) -> String {
    if is_missing_string(&text) {
        text
    } else {
        text.replace(old, new)
    }
}

fn strrep_string_array(array: StringArray, old: &str, new: &str) -> BuiltinResult<Value> {
    let StringArray { data, shape, .. } = array;
    let replaced = data
        .into_iter()
        .map(|text| strrep_string_value(text, old, new))
        .collect::<Vec<_>>();
    let rebuilt = StringArray::new(replaced, shape)
        .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))?;
    Ok(Value::StringArray(rebuilt))
}

fn strrep_char_array(array: CharArray, old: &str, new: &str) -> BuiltinResult<Value> {
    let CharArray { data, rows, cols } = array;
    if rows == 0 || cols == 0 {
        return Ok(Value::CharArray(CharArray { data, rows, cols }));
    }

    let mut replaced_rows = Vec::with_capacity(rows);
    let mut target_cols = 0usize;
    for row in 0..rows {
        let text = char_row_to_string_slice(&data, cols, row);
        let replaced = text.replace(old, new);
        target_cols = target_cols.max(replaced.chars().count());
        replaced_rows.push(replaced);
    }

    let mut new_data = Vec::with_capacity(rows * target_cols);
    for row_text in replaced_rows {
        let mut chars: Vec<char> = row_text.chars().collect();
        if chars.len() < target_cols {
            chars.resize(target_cols, ' ');
        }
        new_data.extend(chars);
    }

    CharArray::new(new_data, rows, target_cols)
        .map(Value::CharArray)
        .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))
}

fn strrep_cell_array(cell: CellArray, old: &str, new: &str) -> BuiltinResult<Value> {
    let CellArray { data, shape, .. } = cell;
    let mut replaced = Vec::with_capacity(data.len());
    for ptr in &data {
        replaced.push(strrep_cell_element(ptr, old, new)?);
    }
    make_cell_with_shape(replaced, shape)
        .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))
}

fn strrep_cell_element(value: &Value, old: &str, new: &str) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => Ok(Value::String(strrep_string_value(text.clone(), old, new))),
        Value::StringArray(array) => strrep_string_array(array.clone(), old, new),
        Value::CharArray(array) => strrep_char_array(array.clone(), old, new),
        _ => Err(runtime_error_for(CELL_ELEMENT_ERROR)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{ResolveContext, Type};

    fn run_strrep(str_value: Value, old_value: Value, new_value: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(strrep_builtin(str_value, old_value, new_value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strrep_string_scalar_basic() {
        let result = run_strrep(
            Value::String("RunMat Ignite".into()),
            Value::String("Ignite".into()),
            Value::String("Interpreter".into()),
        )
        .expect("strrep");
        assert_eq!(result, Value::String("RunMat Interpreter".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strrep_string_array_preserves_missing() {
        let array = StringArray::new(
            vec![
                String::from("gpu"),
                String::from("<missing>"),
                String::from("planner"),
            ],
            vec![3, 1],
        )
        .unwrap();
        let result = run_strrep(
            Value::StringArray(array),
            Value::String("gpu".into()),
            Value::String("GPU".into()),
        )
        .expect("strrep");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![3, 1]);
                assert_eq!(
                    sa.data,
                    vec![
                        String::from("GPU"),
                        String::from("<missing>"),
                        String::from("planner")
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strrep_string_array_with_char_pattern() {
        let array = StringArray::new(
            vec![String::from("alpha"), String::from("beta")],
            vec![2, 1],
        )
        .unwrap();
        let result = run_strrep(
            Value::StringArray(array),
            Value::CharArray(CharArray::new_row("a")),
            Value::CharArray(CharArray::new_row("A")),
        )
        .expect("strrep");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 1]);
                assert_eq!(sa.data, vec![String::from("AlphA"), String::from("betA")]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strrep_char_array_padding() {
        let chars = CharArray::new(vec!['R', 'u', 'n', ' ', 'M', 'a', 't'], 1, 7).unwrap();
        let result = run_strrep(
            Value::CharArray(chars),
            Value::String(" ".into()),
            Value::String("_".into()),
        )
        .expect("strrep");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 7);
                let expected: Vec<char> = "Run_Mat".chars().collect();
                assert_eq!(out.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strrep_char_array_shrinks_rows_pad_with_spaces() {
        let mut data: Vec<char> = "alpha".chars().collect();
        data.extend("beta ".chars());
        let array = CharArray::new(data, 2, 5).unwrap();
        let result = run_strrep(
            Value::CharArray(array),
            Value::String("a".into()),
            Value::String("".into()),
        )
        .expect("strrep");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 4);
                let expected: Vec<char> = vec!['l', 'p', 'h', ' ', 'b', 'e', 't', ' '];
                assert_eq!(out.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strrep_cell_array_char_vectors() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("Kernel Fusion")),
                Value::CharArray(CharArray::new_row("GPU Planner")),
            ],
            1,
            2,
        )
        .unwrap();
        let result = run_strrep(
            Value::Cell(cell),
            Value::String(" ".into()),
            Value::String("_".into()),
        )
        .expect("strrep");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 2);
                assert_eq!(
                    out.get(0, 0).unwrap(),
                    Value::CharArray(CharArray::new_row("Kernel_Fusion"))
                );
                assert_eq!(
                    out.get(0, 1).unwrap(),
                    Value::CharArray(CharArray::new_row("GPU_Planner"))
                );
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strrep_cell_array_string_scalars() {
        let cell = CellArray::new(
            vec![
                Value::String("Planner".into()),
                Value::String("Profiler".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = run_strrep(
            Value::Cell(cell),
            Value::String("er".into()),
            Value::String("ER".into()),
        )
        .expect("strrep");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 2);
                assert_eq!(out.get(0, 0).unwrap(), Value::String("PlannER".into()));
                assert_eq!(out.get(0, 1).unwrap(), Value::String("ProfilER".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strrep_cell_array_invalid_element_error() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err = run_strrep(
            Value::Cell(cell),
            Value::String("1".into()),
            Value::String("one".into()),
        )
        .expect_err("expected cell element error");
        assert!(err.to_string().contains("cell array elements"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strrep_cell_array_char_matrix_element() {
        let mut chars: Vec<char> = "alpha".chars().collect();
        chars.extend("beta ".chars());
        let element = CharArray::new(chars, 2, 5).unwrap();
        let cell = CellArray::new(vec![Value::CharArray(element)], 1, 1).unwrap();
        let result = run_strrep(
            Value::Cell(cell),
            Value::String("a".into()),
            Value::String("A".into()),
        )
        .expect("strrep");
        match result {
            Value::Cell(out) => {
                let nested = out.get(0, 0).unwrap();
                match nested {
                    Value::CharArray(ca) => {
                        assert_eq!(ca.rows, 2);
                        assert_eq!(ca.cols, 5);
                        let expected: Vec<char> =
                            vec!['A', 'l', 'p', 'h', 'A', 'b', 'e', 't', 'A', ' '];
                        assert_eq!(ca.data, expected);
                    }
                    other => panic!("expected char array element, got {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strrep_cell_array_string_arrays() {
        let element = StringArray::new(vec!["alpha".into(), "beta".into()], vec![1, 2]).unwrap();
        let cell = CellArray::new(vec![Value::StringArray(element)], 1, 1).unwrap();
        let result = run_strrep(
            Value::Cell(cell),
            Value::String("a".into()),
            Value::String("A".into()),
        )
        .expect("strrep");
        match result {
            Value::Cell(out) => {
                let nested = out.get(0, 0).unwrap();
                match nested {
                    Value::StringArray(sa) => {
                        assert_eq!(sa.shape, vec![1, 2]);
                        assert_eq!(sa.data, vec![String::from("AlphA"), String::from("betA")]);
                    }
                    other => panic!("expected string array element, got {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strrep_empty_pattern_inserts_replacement() {
        let result = run_strrep(
            Value::String("abc".into()),
            Value::String("".into()),
            Value::String("-".into()),
        )
        .expect("strrep");
        assert_eq!(result, Value::String("-a-b-c-".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strrep_type_mismatch_errors() {
        let err = run_strrep(
            Value::String("abc".into()),
            Value::String("a".into()),
            Value::CharArray(CharArray::new_row("x")),
        )
        .expect_err("expected type mismatch");
        assert!(err.to_string().contains("same data type"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strrep_invalid_pattern_type_errors() {
        let err = run_strrep(
            Value::String("abc".into()),
            Value::Num(1.0),
            Value::String("x".into()),
        )
        .expect_err("expected pattern error");
        assert!(err
            .to_string()
            .contains("string scalars or character vectors"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strrep_first_argument_type_error() {
        let err = run_strrep(
            Value::Num(42.0),
            Value::String("a".into()),
            Value::String("b".into()),
        )
        .expect_err("expected argument type error");
        assert!(err.to_string().contains("first argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn strrep_wgpu_provider_fallback() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            // Unable to initialize the provider in this environment; skip.
            return;
        }
        let result = run_strrep(
            Value::String("Turbine Engine".into()),
            Value::String("Engine".into()),
            Value::String("JIT".into()),
        )
        .expect("strrep");
        assert_eq!(result, Value::String("Turbine JIT".into()));
    }

    #[test]
    fn strrep_type_preserves_text() {
        assert_eq!(
            text_preserve_type(&[Type::String], &ResolveContext::new(Vec::new())),
            Type::String
        );
    }
}
