//! MATLAB-compatible `upper` builtin with GPU-aware semantics for RunMat.
use runmat_builtins::{CellArray, CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{char_row_to_string_slice, uppercase_preserving_missing};
use crate::{gather_if_needed, make_cell};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::upper")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "upper",
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::upper")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "upper",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String transformation builtin; not eligible for fusion and always gathers GPU inputs.",
};

const ARG_TYPE_ERROR: &str =
    "upper: first argument must be a string array, character array, or cell array of character vectors";
const CELL_ELEMENT_ERROR: &str =
    "upper: cell array elements must be string scalars or character vectors";

#[runtime_builtin(
    name = "upper",
    category = "strings/transform",
    summary = "Convert strings, character arrays, and cell arrays of character vectors to uppercase.",
    keywords = "upper,uppercase,strings,character array,text",
    accel = "sink",
    builtin_path = "crate::builtins::strings::transform::upper"
)]
fn upper_builtin(value: Value) -> Result<Value, String> {
    let gathered = gather_if_needed(&value).map_err(|e| format!("upper: {e}"))?;
    match gathered {
        Value::String(text) => Ok(Value::String(uppercase_preserving_missing(text))),
        Value::StringArray(array) => upper_string_array(array),
        Value::CharArray(array) => upper_char_array(array),
        Value::Cell(cell) => upper_cell_array(cell),
        _ => Err(ARG_TYPE_ERROR.to_string()),
    }
}

fn upper_string_array(array: StringArray) -> Result<Value, String> {
    let StringArray { data, shape, .. } = array;
    let uppered = data
        .into_iter()
        .map(uppercase_preserving_missing)
        .collect::<Vec<_>>();
    let upper_array = StringArray::new(uppered, shape).map_err(|e| format!("upper: {e}"))?;
    Ok(Value::StringArray(upper_array))
}

fn upper_char_array(array: CharArray) -> Result<Value, String> {
    let CharArray { data, rows, cols } = array;
    if rows == 0 || cols == 0 {
        return Ok(Value::CharArray(CharArray { data, rows, cols }));
    }

    let mut upper_rows = Vec::with_capacity(rows);
    let mut target_cols = cols;
    for row in 0..rows {
        let text = char_row_to_string_slice(&data, cols, row).to_uppercase();
        let len = text.chars().count();
        target_cols = target_cols.max(len);
        upper_rows.push(text);
    }

    let mut upper_data = Vec::with_capacity(rows * target_cols);
    for row_text in upper_rows {
        let mut chars: Vec<char> = row_text.chars().collect();
        if chars.len() < target_cols {
            chars.resize(target_cols, ' ');
        }
        upper_data.extend(chars.into_iter());
    }

    CharArray::new(upper_data, rows, target_cols)
        .map(Value::CharArray)
        .map_err(|e| format!("upper: {e}"))
}

fn upper_cell_array(cell: CellArray) -> Result<Value, String> {
    let CellArray {
        data, rows, cols, ..
    } = cell;
    let mut upper_values = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let upper = upper_cell_element(&data[idx])?;
            upper_values.push(upper);
        }
    }
    make_cell(upper_values, rows, cols).map_err(|e| format!("upper: {e}"))
}

fn upper_cell_element(value: &Value) -> Result<Value, String> {
    match value {
        Value::String(text) => Ok(Value::String(uppercase_preserving_missing(text.clone()))),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(Value::String(
            uppercase_preserving_missing(sa.data[0].clone()),
        )),
        Value::CharArray(ca) if ca.rows <= 1 => upper_char_array(ca.clone()),
        Value::CharArray(_) => Err(CELL_ELEMENT_ERROR.to_string()),
        _ => Err(CELL_ELEMENT_ERROR.to_string()),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_string_scalar_value() {
        let result = upper_builtin(Value::String("RunMat".into())).expect("upper");
        assert_eq!(result, Value::String("RUNMAT".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_string_array_preserves_shape() {
        let array = StringArray::new(
            vec![
                "gpu".into(),
                "accel".into(),
                "<missing>".into(),
                "MiXeD".into(),
            ],
            vec![2, 2],
        )
        .unwrap();
        let result = upper_builtin(Value::StringArray(array)).expect("upper");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 2]);
                assert_eq!(
                    sa.data,
                    vec![
                        String::from("GPU"),
                        String::from("ACCEL"),
                        String::from("<missing>"),
                        String::from("MIXED")
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_char_array_multiple_rows() {
        let data: Vec<char> = vec!['c', 'a', 't', 'd', 'o', 'g'];
        let array = CharArray::new(data, 2, 3).unwrap();
        let result = upper_builtin(Value::CharArray(array)).expect("upper");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 3);
                assert_eq!(ca.data, vec!['C', 'A', 'T', 'D', 'O', 'G']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_char_vector_handles_padding() {
        let array = CharArray::new_row("hello ");
        let result = upper_builtin(Value::CharArray(array)).expect("upper");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 6);
                let expected: Vec<char> = "HELLO ".chars().collect();
                assert_eq!(ca.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_char_array_unicode_expansion_extends_width() {
        let data: Vec<char> = vec!['ß', 'a'];
        let array = CharArray::new(data, 1, 2).unwrap();
        let result = upper_builtin(Value::CharArray(array)).expect("upper");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 3);
                let expected: Vec<char> = vec!['S', 'S', 'A'];
                assert_eq!(ca.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_cell_array_mixed_content() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("run")),
                Value::String("Mat".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = upper_builtin(Value::Cell(cell)).expect("upper");
        match result {
            Value::Cell(out) => {
                let first = out.get(0, 0).unwrap();
                let second = out.get(0, 1).unwrap();
                assert_eq!(first, Value::CharArray(CharArray::new_row("RUN")));
                assert_eq!(second, Value::String("MAT".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_errors_on_invalid_input() {
        let err = upper_builtin(Value::Num(1.0)).unwrap_err();
        assert_eq!(err, ARG_TYPE_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_cell_errors_on_invalid_element() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err = upper_builtin(Value::Cell(cell)).unwrap_err();
        assert_eq!(err, CELL_ELEMENT_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_preserves_missing_string() {
        let result = upper_builtin(Value::String("<missing>".into())).expect("upper");
        assert_eq!(result, Value::String("<missing>".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn upper_cell_allows_empty_char_vector() {
        let empty_char = CharArray::new(Vec::new(), 1, 0).unwrap();
        let cell = CellArray::new(vec![Value::CharArray(empty_char.clone())], 1, 1).unwrap();
        let result = upper_builtin(Value::Cell(cell)).expect("upper");
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
    fn upper_gpu_tensor_input_gathers_then_errors() {
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
        let err = upper_builtin(Value::GpuTensor(handle.clone())).unwrap_err();
        assert_eq!(err, ARG_TYPE_ERROR);
        provider.free(&handle).ok();
    }
}
