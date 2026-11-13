//! MATLAB-compatible `strcat` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, STRCAT_ERROR_INTERNAL, STRCAT_ERROR_INVALID_CELL_ELEMENT,
    STRCAT_ERROR_INVALID_INPUT, STRCAT_ERROR_NOT_ENOUGH_INPUTS, STRCAT_ERROR_SIZE_MISMATCH,
};

use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, StringArray, Value};

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{
    char_row_to_string_slice, contains_numeric_or_resident_text_input,
};
use crate::{
    build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult, RuntimeError,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::strcat")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strcat",
    op_kind: GpuOpKind::Custom("string-transform"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Executes on the CPU with trailing-space trimming; GPU inputs are gathered before concatenation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::strcat")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strcat",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String concatenation runs on the host and is not eligible for fusion.",
};

const BUILTIN_NAME: &str = "strcat";

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

fn strcat_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn strcat_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    strcat_error_with_message(error.message, error)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OperandKind {
    String,
    Cell,
    Char,
}

type TextElement = runmat_value::StringElement;

#[derive(Clone)]
struct TextOperand {
    data: Vec<TextElement>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    kind: OperandKind,
}

impl TextOperand {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::String(s) => Ok(Self::from_string_scalar(s)),
            Value::StringArray(sa) => Ok(Self::from_string_array(sa)),
            Value::CharArray(ca) => Self::from_char_array(&ca),
            Value::Cell(ca) => Self::from_cell_array(&ca),
            _ => Err(strcat_error(&STRCAT_ERROR_INVALID_INPUT)),
        }
    }

    fn from_string_scalar(text: runmat_value::StringElement) -> Self {
        Self {
            data: vec![text],
            shape: vec![1, 1],
            strides: vec![1, 1],
            kind: OperandKind::String,
        }
    }

    fn from_string_array(array: StringArray) -> Self {
        let shape = array.shape().to_vec();
        let (owner, _shape, _rows, _cols) = array.into_parts();
        let values: Vec<_> = match owner.try_into_owned() {
            Ok(values) => values.into_iter().collect(),
            Err(data) => data.iter().cloned().collect(),
        };
        let data = values;
        let strides = compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
            kind: OperandKind::String,
        }
    }

    fn from_char_array(array: &CharArray) -> BuiltinResult<Self> {
        let rows = array.rows;
        let cols = array.cols;
        let mut elements = Vec::with_capacity(rows);
        for row in 0..rows {
            let text = char_row_to_string_slice(array.data(), cols, row).map_err(|e| {
                strcat_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRCAT_ERROR_INTERNAL)
            })?;
            let trimmed = trim_trailing_spaces(&text);
            elements.push(trimmed.into());
        }
        let shape = vec![rows, 1];
        let strides = compute_strides(&shape);
        Ok(Self {
            data: elements,
            shape,
            strides,
            kind: OperandKind::Char,
        })
    }

    fn from_cell_array(array: &CellArray) -> BuiltinResult<Self> {
        let total = array.data().len();
        let mut elements = Vec::with_capacity(total);
        for handle in array.iter_column_major() {
            let text_element = cell_element_to_text(handle)?;
            elements.push(text_element);
        }
        let shape = array.shape.to_vec();
        let strides = compute_strides(&shape);
        Ok(Self {
            data: elements,
            shape,
            strides,
            kind: OperandKind::Cell,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OutputKind {
    Char,
    Cell,
    String,
}

impl OutputKind {
    fn update(self, operand_kind: OperandKind) -> Self {
        match (self, operand_kind) {
            (_, OperandKind::String) => OutputKind::String,
            (OutputKind::String, _) => OutputKind::String,
            (OutputKind::Cell, _) => OutputKind::Cell,
            (_, OperandKind::Cell) => OutputKind::Cell,
            _ => self,
        }
    }
}

fn trim_trailing_spaces(text: &str) -> String {
    text.trim_end_matches(|ch: char| ch.is_ascii_whitespace())
        .to_string()
}

fn cell_element_to_text(value: &Value) -> BuiltinResult<TextElement> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::StringArray(sa) if sa.data().len() == 1 => {
            let text = sa.data()[0].clone();
            Ok(text)
        }
        Value::CharArray(ca) if ca.rows <= 1 => {
            let text = if ca.rows == 0 {
                String::new()
            } else {
                char_row_to_string_slice(ca.data(), ca.cols, 0).map_err(|e| {
                    strcat_error_with_message(
                        format!("{BUILTIN_NAME}: {e}"),
                        &STRCAT_ERROR_INTERNAL,
                    )
                })?
            };
            Ok(trim_trailing_spaces(&text).into())
        }
        Value::CharArray(_) => Err(strcat_error(&STRCAT_ERROR_INVALID_CELL_ELEMENT)),
        _ => Err(strcat_error(&STRCAT_ERROR_INVALID_CELL_ELEMENT)),
    }
}

#[runtime_builtin(
    name = "strcat",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::strcat"
)]
async fn strcat_builtin(rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::STRCAT_DESCRIPTOR,
        &runmat_builtins::STRCAT_ERROR_TOO_MANY_OUTPUTS,
        "strcat",
    )?;
    if rest.is_empty() {
        return Err(strcat_error(&STRCAT_ERROR_NOT_ENOUGH_INPUTS));
    }

    let mut operands = Vec::with_capacity(rest.len());
    let mut output_kind = OutputKind::Char;

    for value in &rest {
        if contains_numeric_or_resident_text_input(value)
            || crate::builtins::strings::common::has_unsupported_character_pages(value)
        {
            return Err(strcat_error(&STRCAT_ERROR_INVALID_INPUT));
        }
        let gathered = gather_if_needed_async(value).await.map_err(map_flow)?;
        let operand = TextOperand::from_value(gathered)?;
        output_kind = output_kind.update(operand.kind);
        operands.push(operand);
    }

    let mut output_shape = operands
        .first()
        .map(|op| op.shape.clone())
        .unwrap_or_else(|| vec![1, 1]);
    for operand in operands.iter().skip(1) {
        output_shape =
            broadcast_shapes(BUILTIN_NAME, &output_shape, &operand.shape).map_err(|e| {
                strcat_error_with_message(
                    format!("{}: {e}", STRCAT_ERROR_SIZE_MISMATCH.message),
                    &STRCAT_ERROR_SIZE_MISMATCH,
                )
            })?;
    }

    let total_len: usize = output_shape.iter().product();
    let mut concatenated = Vec::with_capacity(total_len);

    for linear in 0..total_len {
        let mut buffer = String::new();
        let mut any_missing = false;
        for operand in &operands {
            let idx = broadcast_index(linear, &output_shape, &operand.shape, &operand.strides);
            let element = &operand.data[idx];
            if element.is_missing() {
                any_missing = true;
                continue;
            }
            buffer.push_str(element.as_str().expect("nonmissing operand"));
        }
        if any_missing {
            concatenated.push(runmat_value::StringElement::Missing);
        } else {
            concatenated.push(runmat_value::StringElement::text(buffer));
        }
    }

    match output_kind {
        OutputKind::String => build_string_output(concatenated, &output_shape),
        OutputKind::Cell => build_cell_output(concatenated, &output_shape),
        OutputKind::Char => build_char_output(concatenated),
    }
}

fn build_string_output(
    data: Vec<runmat_value::StringElement>,
    shape: &[usize],
) -> BuiltinResult<Value> {
    if data.is_empty() {
        let array = StringArray::from_elements_unbounded(data, shape.to_vec()).map_err(|e| {
            strcat_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRCAT_ERROR_INTERNAL)
        })?;
        return Ok(Value::StringArray(array));
    }

    let is_scalar = shape.is_empty() || shape.iter().all(|&dim| dim == 1);
    if is_scalar {
        return Ok(Value::String(data[0].clone()));
    }

    let array = StringArray::from_elements_unbounded(data, shape.to_vec()).map_err(|e| {
        strcat_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRCAT_ERROR_INTERNAL)
    })?;
    Ok(Value::StringArray(array))
}

fn build_cell_output(
    data: Vec<runmat_value::StringElement>,
    shape: &[usize],
) -> BuiltinResult<Value> {
    if data.is_empty() {
        return make_cell_with_shape(Vec::new(), shape.to_vec()).map_err(|e| {
            strcat_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRCAT_ERROR_INTERNAL)
        });
    }
    let mut values = Vec::with_capacity(data.len());
    for text in data {
        values.push(match text.as_str() {
            Some(text) => Value::CharArray(CharArray::new_row(text)),
            None => Value::String(runmat_value::StringElement::Missing),
        });
    }
    CellArray::from_column_major(values, shape.to_vec())
        .map(Value::Cell)
        .map_err(|e| {
            strcat_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRCAT_ERROR_INTERNAL)
        })
}

fn build_char_output(data: Vec<runmat_value::StringElement>) -> BuiltinResult<Value> {
    let rows = data.len();
    if rows == 0 {
        let array = CharArray::from_code_units(Vec::new(), 0, 0).map_err(|e| {
            strcat_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRCAT_ERROR_INTERNAL)
        })?;
        return Ok(Value::CharArray(array));
    }

    let max_cols = data
        .iter()
        .map(|s| {
            s.as_str()
                .expect("character output contains only text")
                .encode_utf16()
                .count()
        })
        .max()
        .unwrap_or(0);
    let mut units = Vec::with_capacity(rows * max_cols);
    for text in data {
        let mut row_units = text
            .as_str()
            .ok_or_else(|| strcat_error(&STRCAT_ERROR_INVALID_INPUT))?
            .encode_utf16()
            .map(runmat_value::Utf16CodeUnit::from_raw)
            .collect::<Vec<_>>();
        if row_units.len() < max_cols {
            row_units.resize(max_cols, runmat_value::Utf16CodeUnit::from_raw(b' ' as u16));
        }
        units.extend(row_units);
    }
    let array = CharArray::from_code_units(units, rows, max_cols).map_err(|e| {
        strcat_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRCAT_ERROR_INTERNAL)
    })?;
    Ok(Value::CharArray(array))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_value::Tensor;
    use runmat_value::{CellArray, CharArray, IntValue, StringArray};

    fn run_strcat(rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(strcat_builtin(
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_string_scalar_concatenation() {
        let result = run_strcat(vec![
            Value::String("Run".into()),
            Value::String("Mat".into()),
        ])
        .expect("strcat");
        assert_eq!(result, Value::String("RunMat".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_string_array_broadcasts_scalar() {
        let array = StringArray::new(vec!["core".into(), "runtime".into()], vec![1, 2]).unwrap();
        let result = run_strcat(vec![
            Value::String("runmat-".into()),
            Value::StringArray(array),
        ])
        .expect("strcat");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 2]);
                assert_eq!(
                    sa.data(),
                    vec![String::from("runmat-core"), String::from("runmat-runtime")]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_char_array_multiple_rows_concatenates_per_row() {
        let first = CharArray::from_unicode_scalars(vec!['A', ' ', 'B', 'C'], 2, 2).expect("char");
        let second = CharArray::from_unicode_scalars(vec!['X', 'Y', 'Z', ' '], 2, 2).expect("char");
        let result =
            run_strcat(vec![Value::CharArray(first), Value::CharArray(second)]).expect("strcat");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 3);
                assert_eq!(
                    ca.data().iter().map(|unit| unit.raw()).collect::<Vec<_>>(),
                    vec![
                        b'A' as u16,
                        b'X' as u16,
                        b'Y' as u16,
                        b'B' as u16,
                        b'C' as u16,
                        b'Z' as u16
                    ]
                );
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_char_array_trims_trailing_spaces() {
        let first = CharArray::new_row("GPU ");
        let second = CharArray::new_row(" Accel  ");
        let result =
            run_strcat(vec![Value::CharArray(first), Value::CharArray(second)]).expect("strcat");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 9);
                assert_eq!(
                    runmat_value::decode_utf16_code_units(ca.data()).unwrap(),
                    "GPU Accel"
                );
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_character_width_is_utf16_code_units() {
        let result = run_strcat(vec![
            Value::CharArray(CharArray::new_row("😀")),
            Value::CharArray(CharArray::new_row("X")),
        ])
        .expect("strcat");
        let Value::CharArray(array) = result else {
            panic!("expected character array");
        };
        assert_eq!((array.rows, array.cols), (1, 3));
        assert_eq!(
            runmat_value::decode_utf16_code_units(array.data()).unwrap(),
            "😀X"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_mixed_char_and_string_returns_string_array() {
        let prefixes =
            CharArray::from_unicode_scalars(vec!['A', ' ', 'B', ' '], 2, 2).expect("char");
        let suffixes =
            StringArray::new(vec!["core".into(), "runtime".into()], vec![1, 2]).expect("strings");
        let result = run_strcat(vec![
            Value::CharArray(prefixes),
            Value::StringArray(suffixes),
        ])
        .expect("strcat");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 2]);
                assert_eq!(
                    sa.data(),
                    vec![
                        "Acore".to_string(),
                        "Bcore".to_string(),
                        "Aruntime".to_string(),
                        "Bruntime".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_cell_array_trims_trailing_spaces() {
        let cell = make_cell_with_shape(
            vec![
                Value::CharArray(CharArray::new_row("Run ")),
                Value::CharArray(CharArray::new_row("Mat ")),
            ],
            vec![1, 2],
        )
        .expect("cell");
        let suffix = Value::CharArray(CharArray::new_row("Core "));
        let result = run_strcat(vec![cell, suffix]).expect("strcat");
        match result {
            Value::Cell(ca) => {
                assert_eq!(ca.shape, vec![1, 2]);
                let first: &Value = &ca.data()[0];
                let second: &Value = &ca.data()[1];
                match (first, second) {
                    (Value::CharArray(a), Value::CharArray(b)) => {
                        assert_eq!(
                            runmat_value::decode_utf16_code_units(a.data()).unwrap(),
                            "RunCore"
                        );
                        assert_eq!(
                            runmat_value::decode_utf16_code_units(b.data()).unwrap(),
                            "MatCore"
                        );
                    }
                    other => panic!("unexpected cell contents {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
        let cell = make_cell_with_shape(
            ["a", "b", "c", "d", "e", "f", "g", "h"]
                .into_iter()
                .map(|text| Value::CharArray(CharArray::new_row(text)))
                .collect(),
            vec![2, 2, 2],
        )
        .expect("page-local cell storage");
        let suffixes = Value::StringArray(
            StringArray::new((1..=8).map(|n| n.to_string()).collect(), vec![2, 2, 2])
                .expect("column-major suffixes"),
        );
        let Value::StringArray(mixed) =
            run_strcat(vec![cell.clone(), suffixes]).expect("mixed nd text")
        else {
            panic!("string output");
        };
        assert_eq!(mixed.shape(), &[2, 2, 2]);
        assert_eq!(
            mixed.data().to_vec(),
            vec!["a1", "c2", "b3", "d4", "e5", "g6", "f7", "h8"]
        );
        let Value::Cell(cells) = run_strcat(vec![cell, Value::CharArray(CharArray::new_row("!"))])
            .expect("nd cell output")
        else {
            panic!("cell output");
        };
        assert_eq!(cells.shape(), &[2, 2, 2]);
        for (actual, expected) in cells
            .data()
            .iter()
            .zip(["a!", "b!", "c!", "d!", "e!", "f!", "g!", "h!"])
        {
            assert_eq!(actual, &Value::CharArray(CharArray::new_row(expected)));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_cell_array_two_by_two_preserves_row_major_order() {
        let cell = make_cell_with_shape(
            vec![
                Value::CharArray(CharArray::new_row("Top ")),
                Value::CharArray(CharArray::new_row("Right ")),
                Value::CharArray(CharArray::new_row("Bottom ")),
                Value::CharArray(CharArray::new_row("Last ")),
            ],
            vec![2, 2],
        )
        .expect("cell");
        let suffix = Value::CharArray(CharArray::new_row("X"));
        let result = run_strcat(vec![cell, suffix]).expect("strcat");
        match result {
            Value::Cell(ca) => {
                assert_eq!(ca.shape, vec![2, 2]);
                let v00 = ca.get(0, 0).expect("cell (0,0)");
                let v01 = ca.get(0, 1).expect("cell (0,1)");
                let v10 = ca.get(1, 0).expect("cell (1,0)");
                let v11 = ca.get(1, 1).expect("cell (1,1)");
                match (v00, v01, v10, v11) {
                    (
                        Value::CharArray(a),
                        Value::CharArray(b),
                        Value::CharArray(c),
                        Value::CharArray(d),
                    ) => {
                        assert_eq!(
                            runmat_value::decode_utf16_code_units(a.data()).unwrap(),
                            "TopX"
                        );
                        assert_eq!(
                            runmat_value::decode_utf16_code_units(b.data()).unwrap(),
                            "RightX"
                        );
                        assert_eq!(
                            runmat_value::decode_utf16_code_units(c.data()).unwrap(),
                            "BottomX"
                        );
                        assert_eq!(
                            runmat_value::decode_utf16_code_units(d.data()).unwrap(),
                            "LastX"
                        );
                    }
                    other => panic!("unexpected cell contents {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_missing_strings_propagate() {
        let array = StringArray::from_elements_unbounded(
            vec![runmat_value::StringElement::Missing, "ready".into()],
            vec![1, 2],
        )
        .unwrap();
        let result = run_strcat(vec![
            Value::String("job-".into()),
            Value::StringArray(array),
        ])
        .expect("strcat");
        match result {
            Value::StringArray(sa) => {
                assert!(sa.data()[0].is_missing());
                assert_eq!(sa.data()[1], "job-ready");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_empty_dimension_returns_empty_array() {
        let empty = StringArray::new(Vec::<String>::new(), vec![0, 2]).expect("string array");
        let result = run_strcat(vec![
            Value::StringArray(empty),
            Value::String("prefix".into()),
        ])
        .expect("strcat");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 2]);
                assert!(sa.data().is_empty());
            }
            other => panic!("expected empty string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_errors_on_invalid_input_type() {
        let err = run_strcat(vec![Value::Int(IntValue::I32(4))]).expect_err("expected error");
        assert_eq!(err.to_string(), STRCAT_ERROR_INVALID_INPUT.message);
        assert_eq!(
            err.identifier.as_deref(),
            STRCAT_ERROR_INVALID_INPUT.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_errors_on_mismatched_sizes() {
        let left = CharArray::from_unicode_scalars(vec!['A', 'B'], 2, 1).expect("char");
        let right = CharArray::from_unicode_scalars(vec!['C', 'D', 'E'], 3, 1).expect("char");
        let err = run_strcat(vec![Value::CharArray(left), Value::CharArray(right)])
            .expect_err("expected broadcast error");
        assert!(err
            .to_string()
            .starts_with(STRCAT_ERROR_SIZE_MISMATCH.message));
        assert_eq!(
            err.identifier.as_deref(),
            STRCAT_ERROR_SIZE_MISMATCH.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_errors_on_invalid_cell_element() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell");
        let err = run_strcat(vec![Value::Cell(cell)]).expect_err("expected error");
        assert_eq!(err.to_string(), STRCAT_ERROR_INVALID_CELL_ELEMENT.message);
        assert_eq!(
            err.identifier.as_deref(),
            STRCAT_ERROR_INVALID_CELL_ELEMENT.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_errors_on_empty_argument_list() {
        let err = run_strcat(Vec::new()).expect_err("expected error");
        assert_eq!(err.to_string(), STRCAT_ERROR_NOT_ENOUGH_INPUTS.message);
        assert_eq!(
            err.identifier.as_deref(),
            STRCAT_ERROR_NOT_ENOUGH_INPUTS.identifier
        );
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_gpu_operand_still_errors_on_type() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let err = run_strcat(vec![Value::GpuTensor(handle)]).expect_err("expected error");
            assert_eq!(err.to_string(), STRCAT_ERROR_INVALID_INPUT.message);
        });
    }
}
