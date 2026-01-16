//! MATLAB-compatible `strfind` builtin for RunMat.

use runmat_builtins::{CellArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::gather_if_needed;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};

use super::text_utils::{value_to_owned_string, TextCollection, TextElement};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::search::strfind")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strfind",
    op_kind: GpuOpKind::Custom("string-search"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Executes entirely on the host; GPU-resident inputs are gathered before substring matching.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::search::strfind")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strfind",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Text operation; not eligible for fusion and materialises host-side numeric or cell outputs.",
};

#[runtime_builtin(
    name = "strfind",
    category = "strings/search",
    summary = "Return the starting indices of pattern matches in text inputs.",
    keywords = "strfind,substring,index,positions,string search",
    accel = "sink",
    builtin_path = "crate::builtins::strings::search::strfind"
)]
fn strfind_builtin(text: Value, pattern: Value, rest: Vec<Value>) -> Result<Value, String> {
    let text = gather_if_needed(&text).map_err(|e| format!("strfind: {e}"))?;
    let pattern = gather_if_needed(&pattern).map_err(|e| format!("strfind: {e}"))?;
    let force_cell_output = parse_force_cell_output(&rest)?;

    let subject = TextCollection::from_subject("strfind", text)?;
    let patterns = TextCollection::from_pattern("strfind", pattern)?;

    evaluate_strfind(&subject, &patterns, force_cell_output)
}

fn evaluate_strfind(
    subject: &TextCollection,
    patterns: &TextCollection,
    force_cell_output: bool,
) -> Result<Value, String> {
    let output_shape = broadcast_shapes("strfind", &subject.shape, &patterns.shape)?;
    let total = tensor::element_count(&output_shape);
    let return_cell = force_cell_output || subject.is_cell || patterns.is_cell || total != 1;

    let subject_strides = compute_strides(&subject.shape);
    let pattern_strides = compute_strides(&patterns.shape);

    let mut matches: Vec<Vec<usize>> = Vec::with_capacity(total);
    for linear in 0..total {
        let subject_idx = broadcast_index(linear, &output_shape, &subject.shape, &subject_strides);
        let pattern_idx = broadcast_index(linear, &output_shape, &patterns.shape, &pattern_strides);
        let result = match (
            &subject.elements[subject_idx],
            &patterns.elements[pattern_idx],
        ) {
            (TextElement::Missing, _) => Vec::new(),
            (_, TextElement::Missing) => Vec::new(),
            (TextElement::Text(text), TextElement::Text(pattern)) => {
                find_indices(text, pattern.as_str())
            }
        };
        matches.push(result);
    }

    if !return_cell {
        let indices = matches.into_iter().next().unwrap_or_default();
        return indices_to_numeric_value(&indices);
    }

    indices_to_cell(matches, &output_shape)
}

fn find_indices(text: &str, pattern: &str) -> Vec<usize> {
    if pattern.is_empty() {
        let len = text.chars().count();
        return (0..=len).map(|pos| pos + 1).collect();
    }

    let text_chars: Vec<char> = text.chars().collect();
    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_len = text_chars.len();
    let pattern_len = pattern_chars.len();

    if pattern_len == 0 || pattern_len > text_len {
        return Vec::new();
    }

    let mut indices = Vec::new();
    for start in 0..=text_len - pattern_len {
        if &text_chars[start..start + pattern_len] == pattern_chars.as_slice() {
            indices.push(start + 1);
        }
    }
    indices
}

fn indices_to_numeric_value(indices: &[usize]) -> Result<Value, String> {
    let data = indices.iter().map(|&pos| pos as f64).collect::<Vec<_>>();
    let cols = indices.len();
    Tensor::new(data, vec![1, cols])
        .map(Value::Tensor)
        .map_err(|e| format!("strfind: {e}"))
}

fn indices_to_tensor(indices: &[usize]) -> Result<Value, String> {
    Tensor::new(
        indices.iter().map(|&pos| pos as f64).collect::<Vec<_>>(),
        vec![1, indices.len()],
    )
    .map(Value::Tensor)
    .map_err(|e| format!("strfind: {e}"))
}

fn indices_to_cell(matches: Vec<Vec<usize>>, shape: &[usize]) -> Result<Value, String> {
    let total = matches.len();
    if total == 0 {
        let (rows, cols) = shape_to_rows_cols(shape);
        return CellArray::new(Vec::new(), rows, cols)
            .map(Value::Cell)
            .map_err(|e| format!("strfind: {e}"));
    }

    let (rows, cols) = shape_to_rows_cols(shape);
    if rows * cols != total {
        return Err("strfind: internal size mismatch while constructing cell output".to_string());
    }

    let mut values = Vec::with_capacity(total);
    for row in 0..rows {
        for col in 0..cols {
            let column_major_idx = row + rows * col;
            let indices = matches
                .get(column_major_idx)
                .ok_or_else(|| "strfind: internal indexing error".to_string())?;
            let cell_value = indices_to_tensor(indices)?;
            values.push(cell_value);
        }
    }

    CellArray::new(values, rows, cols)
        .map(Value::Cell)
        .map_err(|e| format!("strfind: {e}"))
}

fn shape_to_rows_cols(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], 1),
        _ => {
            let rows = shape[0];
            let cols = shape[1..]
                .iter()
                .copied()
                .fold(1usize, |acc, dim| acc.saturating_mul(dim));
            (rows, cols)
        }
    }
}

fn parse_force_cell_output(rest: &[Value]) -> Result<bool, String> {
    if rest.is_empty() {
        return Ok(false);
    }
    if !rest.len().is_multiple_of(2) {
        return Err(
            "strfind: expected name-value pairs after the pattern (e.g., 'ForceCellOutput', true)"
                .to_string(),
        );
    }

    let mut force_cell = None;
    for pair in rest.chunks(2) {
        let name = value_to_owned_string(&pair[0])
            .ok_or_else(|| "strfind: option names must be text scalars".to_string())?;
        if !name.eq_ignore_ascii_case("forcecelloutput") {
            return Err(format!(
                "strfind: unknown option '{name}'; supported option is 'ForceCellOutput'"
            ));
        }
        let value = parse_bool_like(&pair[1])?;
        force_cell = Some(value);
    }
    force_cell.ok_or_else(|| {
        "strfind: expected 'ForceCellOutput' option when providing name-value arguments".to_string()
    })
}

fn parse_bool_like(value: &Value) -> Result<bool, String> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(!i.is_zero()),
        Value::Num(n) => {
            if !n.is_finite() {
                Err("strfind: option values must be finite numeric scalars".to_string())
            } else {
                Ok(*n != 0.0)
            }
        }
        Value::LogicalArray(array) => {
            if array.data.len() != 1 {
                Err(format!(
                    "strfind: option values must be scalar logicals (received {} elements)",
                    array.data.len()
                ))
            } else {
                Ok(array.data[0] != 0)
            }
        }
        Value::Tensor(tensor) => {
            if tensor.data.len() != 1 {
                Err(format!(
                    "strfind: option values must be scalar numeric values (received {} elements)",
                    tensor.data.len()
                ))
            } else if !tensor.data[0].is_finite() {
                Err("strfind: option values must be finite numeric scalars".to_string())
            } else {
                Ok(tensor.data[0] != 0.0)
            }
        }
        other => value_to_owned_string(other)
            .ok_or_else(|| "strfind: option values must be logical or numeric scalars".to_string())
            .and_then(|text| match text.trim().to_ascii_lowercase().as_str() {
                "true" | "on" | "1" => Ok(true),
                "false" | "off" | "0" => Ok(false),
                _ => Err(format!(
                    "strfind: invalid value '{text}' for 'ForceCellOutput'; expected true or false"
                )),
            }),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{CellArray, CharArray, StringArray, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_single_match_returns_row_vector() {
        let result = strfind_builtin(
            Value::String("saturn".into()),
            Value::String("sat".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 1]);
                assert_eq!(tensor.data, vec![1.0]);
            }
            other => panic!("expected 1x1 tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_char_vector_matches() {
        let result = strfind_builtin(
            Value::CharArray(CharArray::new_row("abracadabra")),
            Value::CharArray(CharArray::new_row("abra")),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.data, vec![1.0, 8.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_overlapping_matches() {
        let result = strfind_builtin(
            Value::String("aaaa".into()),
            Value::String("aa".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_eq!(tensor.data, vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_empty_pattern_returns_boundaries() {
        let result = strfind_builtin(
            Value::String("abc".into()),
            Value::String("".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 4]);
                assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_string_array_returns_cell() {
        let strings = StringArray::new(
            vec!["hydrogen".into(), "helium".into(), "lithium".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = strfind_builtin(
            Value::StringArray(strings),
            Value::String("i".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 3);
                assert_eq!(cell.cols, 1);
                let first = cell.get(0, 0).unwrap();
                let second = cell.get(1, 0).unwrap();
                let third = cell.get(2, 0).unwrap();
                match first {
                    Value::Tensor(tensor) => assert!(tensor.data.is_empty()),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match second {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![4.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match third {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![2.0, 5.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_pattern_array_returns_cell() {
        let patterns =
            StringArray::new(vec!["sat".into(), "turn".into(), "moon".into()], vec![1, 3]).unwrap();
        let result = strfind_builtin(
            Value::String("saturn".into()),
            Value::StringArray(patterns),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 3);
                let first = cell.get(0, 0).unwrap();
                let second = cell.get(0, 1).unwrap();
                let third = cell.get(0, 2).unwrap();
                match first {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match second {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![3.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match third {
                    Value::Tensor(tensor) => assert!(tensor.data.is_empty()),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_name_value() {
        let result = strfind_builtin(
            Value::CharArray(CharArray::new_row("mission")),
            Value::CharArray(CharArray::new_row("s")),
            vec![Value::String("ForceCellOutput".into()), Value::Bool(true)],
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![3.0, 4.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_numeric_value() {
        let result = strfind_builtin(
            Value::String("mission".into()),
            Value::String("s".into()),
            vec![Value::String("ForceCellOutput".into()), Value::Num(1.0)],
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![3.0, 4.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_off_string() {
        let result = strfind_builtin(
            Value::String("mission".into()),
            Value::String("s".into()),
            vec![
                Value::String("ForceCellOutput".into()),
                Value::String("off".into()),
            ],
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.data, vec![3.0, 4.0]);
            }
            other => panic!("expected numeric tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_non_scalar_error() {
        let option_value =
            Tensor::new(vec![1.0, 0.0], vec![1, 2]).expect("tensor construction for test");
        let err = strfind_builtin(
            Value::String("mission".into()),
            Value::String("s".into()),
            vec![
                Value::String("ForceCellOutput".into()),
                Value::Tensor(option_value),
            ],
        )
        .expect_err("strfind should error for non-scalar ForceCellOutput");
        assert!(
            err.contains("scalar"),
            "unexpected error message for non-scalar option: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_missing_value_error() {
        let err = strfind_builtin(
            Value::String("mission".into()),
            Value::String("s".into()),
            vec![Value::String("ForceCellOutput".into())],
        )
        .expect_err("strfind should error when ForceCellOutput value missing");
        assert!(
            err.contains("name-value pairs"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_subject_cell_scalar_returns_cell() {
        let subject = CellArray::new(vec![Value::from("needle")], 1, 1).expect("cell construction");
        let result = strfind_builtin(
            Value::Cell(subject),
            Value::String("needle".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_pattern_cell_scalar_returns_cell() {
        let pattern = CellArray::new(vec![Value::from("needle")], 1, 1).expect("cell construction");
        let result = strfind_builtin(
            Value::String("needle".into()),
            Value::Cell(pattern),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_missing_subject_returns_empty() {
        let result = strfind_builtin(
            Value::String("<missing>".into()),
            Value::String("abc".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 0]);
                assert!(tensor.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_missing_pattern_returns_empty_vector() {
        let patterns =
            StringArray::new(vec!["<missing>".into()], vec![1, 1]).expect("string array creation");
        let result = strfind_builtin(
            Value::String("planet".into()),
            Value::StringArray(patterns),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 0]);
                assert!(tensor.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_char_matrix_rows() {
        let data = vec!['c', 'a', 't', 'a', 'd', 'a', 'd', 'o', 'g'];
        let array = CharArray::new(data, 3, 3).unwrap();
        let result = strfind_builtin(
            Value::CharArray(array),
            Value::CharArray(CharArray::new_row("d")),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 3);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert!(tensor.data.is_empty()),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match cell.get(1, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![2.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match cell.get(2, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_invalid_option_name_errors() {
        let err = strfind_builtin(
            Value::String("abc".into()),
            Value::String("a".into()),
            vec![Value::String("IgnoreCase".into()), Value::Bool(true)],
        )
        .expect_err("strfind should error");
        assert!(
            err.contains("unknown option"),
            "unexpected error message: {err}"
        );
    }
}
