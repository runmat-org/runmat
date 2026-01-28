//! MATLAB-compatible `join` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{CellArray, CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::join")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "join",
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
    notes: "Executes on the host; GPU-resident inputs and delimiters are gathered before concatenation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::join")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "join",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Joins operate on CPU-managed text and are ineligible for fusion.",
};

const BUILTIN_NAME: &str = "join";
const INPUT_TYPE_ERROR: &str =
    "join: input must be a string array, string scalar, character array, or cell array of character vectors";
const DELIMITER_TYPE_ERROR: &str =
    "join: delimiter must be a string, character vector, string array, or cell array of character vectors";
const DELIMITER_SIZE_ERROR: &str =
    "join: size of delimiter array must match the size of str, with the join dimension reduced by one";
const DIMENSION_TYPE_ERROR: &str = "join: dimension must be a positive integer scalar";

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "join",
    category = "strings/transform",
    summary = "Combine text across a specified dimension inserting delimiters between elements.",
    keywords = "join,string join,concatenate strings,delimiters,cell array join",
    accel = "none",
    builtin_path = "crate::builtins::strings::transform::join"
)]
async fn join_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text).await.map_err(map_flow)?;
    let mut args = Vec::with_capacity(rest.len());
    for arg in rest {
        args.push(gather_if_needed_async(&arg).await.map_err(map_flow)?);
    }

    let mut input = JoinInput::from_value(text)?;
    let (delimiter_arg, dimension_arg) = parse_arguments(&args)?;

    let mut shape = input.shape.clone();
    if shape.is_empty() {
        shape = vec![1, 1];
    }

    let default_dim = default_dimension(&shape);
    let dimension = match dimension_arg {
        Some(dim) => dim,
        None => default_dim,
    };

    if dimension == 0 {
        return Err(runtime_error_for(DIMENSION_TYPE_ERROR));
    }

    let ndims = input.ndims();
    if dimension > ndims {
        return input.into_value();
    }

    let axis_idx = dimension - 1;
    input.ensure_shape_len(dimension);
    let full_shape = input.shape.clone();

    let delimiter = Delimiter::from_value(delimiter_arg, &full_shape, axis_idx)?;

    let (output_data, output_shape) = perform_join(&input.data, &full_shape, axis_idx, &delimiter);

    input.build_output(output_data, output_shape)
}

fn parse_arguments(args: &[Value]) -> BuiltinResult<(Option<Value>, Option<usize>)> {
    match args.len() {
        0 => Ok((None, None)),
        1 => {
            if let Some(dim) = value_to_dimension(&args[0])? {
                Ok((None, Some(dim)))
            } else {
                Ok((Some(args[0].clone()), None))
            }
        }
        2 => {
            if let Some(dim) = value_to_dimension(&args[1])? {
                Ok((Some(args[0].clone()), Some(dim)))
            } else if let Some(dim) = value_to_dimension(&args[0])? {
                Ok((Some(args[1].clone()), Some(dim)))
            } else {
                Err(runtime_error_for(DIMENSION_TYPE_ERROR))
            }
        }
        _ => Err(runtime_error_for("join: too many input arguments")),
    }
}

fn default_dimension(shape: &[usize]) -> usize {
    for (index, size) in shape.iter().enumerate().rev() {
        if *size != 1 {
            return index + 1;
        }
    }
    2
}

fn value_to_dimension(value: &Value) -> BuiltinResult<Option<usize>> {
    match value {
        Value::Int(i) => {
            let v = i.to_i64();
            if v <= 0 {
                return Err(runtime_error_for(DIMENSION_TYPE_ERROR));
            }
            Ok(Some(v as usize))
        }
        Value::Num(n) => {
            if !n.is_finite() || *n <= 0.0 {
                return Err(runtime_error_for(DIMENSION_TYPE_ERROR));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(runtime_error_for(DIMENSION_TYPE_ERROR));
            }
            Ok(Some(rounded as usize))
        }
        Value::Tensor(t) if t.data.len() == 1 => {
            let val = t.data[0];
            if !val.is_finite() || val <= 0.0 {
                return Err(runtime_error_for(DIMENSION_TYPE_ERROR));
            }
            let rounded = val.round();
            if (rounded - val).abs() > f64::EPSILON {
                return Err(runtime_error_for(DIMENSION_TYPE_ERROR));
            }
            Ok(Some(rounded as usize))
        }
        _ => Ok(None),
    }
}

struct JoinInput {
    data: Vec<String>,
    shape: Vec<usize>,
    kind: OutputKind,
}

#[derive(Clone)]
enum OutputKind {
    StringScalar,
    StringArray,
    CellArray,
}

impl JoinInput {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::String(text) => Ok(Self {
                data: vec![text],
                shape: vec![1, 1],
                kind: OutputKind::StringScalar,
            }),
            Value::StringArray(array) => Ok(Self {
                data: array.data,
                shape: array.shape,
                kind: OutputKind::StringArray,
            }),
            Value::CharArray(array) => {
                let strings = char_array_rows_to_strings(&array);
                Ok(Self {
                    data: strings,
                    shape: vec![array.rows, 1],
                    kind: OutputKind::StringArray,
                })
            }
            Value::Cell(cell) => {
                let (data, shape) = cell_array_to_strings(cell)?;
                Ok(Self {
                    data,
                    shape,
                    kind: OutputKind::CellArray,
                })
            }
            _ => Err(runtime_error_for(INPUT_TYPE_ERROR)),
        }
    }

    fn ndims(&self) -> usize {
        if self.shape.is_empty() {
            2
        } else {
            self.shape.len().max(2)
        }
    }

    fn ensure_shape_len(&mut self, dimension: usize) {
        if self.shape.len() < dimension {
            self.shape.resize(dimension, 1);
        }
    }

    fn into_value(self) -> BuiltinResult<Value> {
        build_value(self.kind, self.data, self.shape)
    }

    fn build_output(&self, data: Vec<String>, shape: Vec<usize>) -> BuiltinResult<Value> {
        build_value(self.kind.clone(), data, shape)
    }
}

fn build_value(kind: OutputKind, data: Vec<String>, shape: Vec<usize>) -> BuiltinResult<Value> {
    match kind {
        OutputKind::StringScalar => Ok(Value::String(data.into_iter().next().unwrap_or_default())),
        OutputKind::StringArray => {
            let array = StringArray::new(data, shape)
                .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))?;
            Ok(Value::StringArray(array))
        }
        OutputKind::CellArray => {
            let rows = shape.first().copied().unwrap_or(0);
            let cols = shape.get(1).copied().unwrap_or(1);
            if rows == 0 || cols == 0 || data.is_empty() {
                return make_cell(Vec::new(), rows, cols)
                    .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")));
            }
            let mut values = Vec::with_capacity(rows * cols);
            for row in 0..rows {
                for col in 0..cols {
                    let idx = row + col * rows;
                    let text = data[idx].clone();
                    let chars: Vec<char> = text.chars().collect();
                    let cols_count = chars.len();
                    let char_array = CharArray::new(chars, 1, cols_count)
                        .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))?;
                    values.push(Value::CharArray(char_array));
                }
            }
            make_cell(values, rows, cols)
                .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))
        }
    }
}

fn char_array_rows_to_strings(array: &CharArray) -> Vec<String> {
    let mut strings = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        strings.push(char_row_to_string_slice(&array.data, array.cols, row));
    }
    strings
}

fn cell_array_to_strings(cell: CellArray) -> BuiltinResult<(Vec<String>, Vec<usize>)> {
    let CellArray {
        data, rows, cols, ..
    } = cell;
    let mut strings = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            let idx = row * cols + col;
            strings.push(
                cell_element_to_string(&data[idx])
                    .ok_or_else(|| runtime_error_for(INPUT_TYPE_ERROR))?,
            );
        }
    }
    Ok((strings, vec![rows, cols]))
}

fn cell_element_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(array) if array.rows <= 1 => {
            if array.rows == 0 {
                Some(String::new())
            } else {
                Some(char_row_to_string_slice(&array.data, array.cols, 0))
            }
        }
        _ => None,
    }
}

#[derive(Clone)]
enum Delimiter {
    Scalar(String),
    Array(DelimiterArray),
}

#[derive(Clone)]
struct DelimiterArray {
    data: Vec<String>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Delimiter {
    fn from_value(
        value: Option<Value>,
        full_shape: &[usize],
        axis_idx: usize,
    ) -> BuiltinResult<Self> {
        match value {
            None => Ok(Self::Scalar(" ".to_string())),
            Some(v) => {
                if let Some(text) = value_to_scalar_string(&v) {
                    return Ok(Self::Scalar(text));
                }
                let (data, shape) = value_to_string_array(v)?;
                let normalized = normalize_delimiter_shape(shape, full_shape, axis_idx)?;
                let strides = compute_strides(&normalized);
                Ok(Self::Array(DelimiterArray {
                    data,
                    shape: normalized,
                    strides,
                }))
            }
        }
    }

    fn value<'a>(&'a self, coords: &[usize], axis_idx: usize, axis_gap: usize) -> &'a str {
        match self {
            Delimiter::Scalar(text) => text.as_str(),
            Delimiter::Array(array) => array.value(coords, axis_idx, axis_gap),
        }
    }
}

impl DelimiterArray {
    fn value<'a>(&'a self, coords: &[usize], axis_idx: usize, axis_gap: usize) -> &'a str {
        let mut offset = 0usize;
        for (dim, stride) in self.strides.iter().enumerate() {
            let size = self.shape[dim];
            let coord = if dim == axis_idx {
                axis_gap.min(size.saturating_sub(1))
            } else if size == 1 {
                0
            } else {
                coords[dim].min(size.saturating_sub(1))
            };
            offset += coord * stride;
        }
        &self.data[offset]
    }
}

fn value_to_scalar_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(array) if array.rows <= 1 => {
            if array.rows == 0 {
                Some(String::new())
            } else {
                Some(char_row_to_string_slice(&array.data, array.cols, 0))
            }
        }
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::Cell(cell) if cell.data.len() == 1 => cell_element_to_string(&cell.data[0]),
        _ => None,
    }
}

fn value_to_string_array(value: Value) -> BuiltinResult<(Vec<String>, Vec<usize>)> {
    match value {
        Value::StringArray(array) => Ok((array.data, array.shape)),
        Value::Cell(cell) => {
            let (data, shape) = cell_array_to_strings(cell)?;
            Ok((data, shape))
        }
        Value::CharArray(array) => {
            let rows = array.rows;
            let strings = char_array_rows_to_strings(&array);
            Ok((strings, vec![rows, 1]))
        }
        _ => Err(runtime_error_for(DELIMITER_TYPE_ERROR)),
    }
}

fn normalize_delimiter_shape(
    mut shape: Vec<usize>,
    full_shape: &[usize],
    axis_idx: usize,
) -> BuiltinResult<Vec<usize>> {
    if shape.len() > full_shape.len() {
        return Err(runtime_error_for(DELIMITER_SIZE_ERROR));
    }
    if shape.len() < full_shape.len() {
        shape.resize(full_shape.len(), 1);
    }

    let axis_len = full_shape[axis_idx].saturating_sub(1);
    if axis_len == 0 {
        shape[axis_idx] = 1;
    } else if shape[axis_idx] != axis_len {
        return Err(runtime_error_for(DELIMITER_SIZE_ERROR));
    }

    for (dim, size) in shape.iter().enumerate() {
        if dim == axis_idx {
            continue;
        }
        let reference = full_shape[dim];
        if *size != reference && *size != 1 {
            return Err(runtime_error_for(DELIMITER_SIZE_ERROR));
        }
    }

    Ok(shape)
}

fn perform_join(
    data: &[String],
    full_shape: &[usize],
    axis_idx: usize,
    delimiter: &Delimiter,
) -> (Vec<String>, Vec<usize>) {
    if full_shape.is_empty() {
        return (vec![String::new()], vec![1, 1]);
    }

    let axis_len = full_shape[axis_idx];
    let mut output_shape = full_shape.to_vec();

    let rest_size = full_shape
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != axis_idx)
        .fold(1usize, |acc, (_, size)| acc.saturating_mul(*size));

    if rest_size == 0 {
        output_shape[axis_idx] = 0;
        return (Vec::new(), output_shape);
    }

    output_shape[axis_idx] = 1;

    let total_output = rest_size;
    let mut output = Vec::with_capacity(total_output);

    let strides = compute_strides(full_shape);
    let axis_stride = strides[axis_idx];
    let dims = full_shape.len();
    let mut coords = vec![0usize; dims];

    for _ in 0..rest_size {
        let mut base_offset = 0usize;
        for dim in 0..dims {
            base_offset += coords[dim] * strides[dim];
        }

        if axis_len == 0 {
            output.push(String::new());
        } else {
            let mut result = String::new();
            let mut missing = false;
            for axis_pos in 0..axis_len {
                let element_offset = base_offset + axis_pos * axis_stride;
                let value = &data[element_offset];
                if is_missing_string(value) {
                    missing = true;
                    break;
                }
                if axis_pos > 0 {
                    let gap = axis_pos - 1;
                    let delim = delimiter.value(&coords, axis_idx, gap);
                    result.push_str(delim);
                }
                result.push_str(value);
            }
            if missing {
                output.push("<missing>".to_string());
            } else {
                output.push(result);
            }
        }

        increment_coords(&mut coords, full_shape, axis_idx);
    }

    (output, output_shape)
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for dim in 1..shape.len() {
        strides[dim] = strides[dim - 1].saturating_mul(shape[dim - 1]);
    }
    strides
}

fn increment_coords(coords: &mut [usize], shape: &[usize], axis_idx: usize) {
    for dim in 0..shape.len() {
        if dim == axis_idx {
            continue;
        }
        coords[dim] += 1;
        if coords[dim] < shape[dim] {
            break;
        }
        coords[dim] = 0;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_backend;
    use runmat_builtins::IntValue;

    fn join_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::join_builtin(text, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_string_array_default_dimension() {
        let array = StringArray::new(
            vec![
                "Carlos".into(),
                "Ella".into(),
                "Diana".into(),
                "Sada".into(),
                "Olsen".into(),
                "Lee".into(),
            ],
            vec![3, 2],
        )
        .unwrap();
        let result = join_builtin(Value::StringArray(array), Vec::new()).expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![3, 1]);
                assert_eq!(
                    sa.data,
                    vec![
                        "Carlos Sada".to_string(),
                        "Ella Olsen".to_string(),
                        "Diana Lee".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_with_custom_scalar_delimiter() {
        let array = StringArray::new(
            vec![
                "x".into(),
                "a".into(),
                "y".into(),
                "b".into(),
                "z".into(),
                "c".into(),
            ],
            vec![2, 3],
        )
        .unwrap();
        let result =
            join_builtin(Value::StringArray(array), vec![Value::String("-".into())]).expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 1]);
                assert_eq!(sa.data, vec![String::from("x-y-z"), String::from("a-b-c")]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_with_delimiter_array_per_row() {
        let array = StringArray::new(
            vec![
                "x".into(),
                "a".into(),
                "y".into(),
                "b".into(),
                "z".into(),
                "c".into(),
            ],
            vec![2, 3],
        )
        .unwrap();
        let delims = StringArray::new(
            vec![" + ".into(), " - ".into(), " = ".into(), " = ".into()],
            vec![2, 2],
        )
        .unwrap();
        let result = join_builtin(Value::StringArray(array), vec![Value::StringArray(delims)])
            .expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 1]);
                assert_eq!(
                    sa.data,
                    vec![String::from("x + y = z"), String::from("a - b = c")]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_with_dimension_argument() {
        let array = StringArray::new(
            vec![
                "Carlos".into(),
                "Ella".into(),
                "Diana".into(),
                "Sada".into(),
                "Olsen".into(),
                "Lee".into(),
            ],
            vec![3, 2],
        )
        .unwrap();
        let result = join_builtin(
            Value::StringArray(array),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 2]);
                assert_eq!(
                    sa.data,
                    vec![
                        String::from("Carlos Ella Diana"),
                        String::from("Sada Olsen Lee"),
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_dimension_greater_than_ndims_returns_input() {
        let array = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
        let result = join_builtin(
            Value::StringArray(array.clone()),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, array.shape);
                assert_eq!(sa.data, array.data);
            }
            other => panic!("expected original array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_cell_array_of_char_vectors() {
        let gpu = CharArray::new_row("GPU");
        let accel = CharArray::new_row("Accelerate");
        let ignition = CharArray::new_row("Ignition");
        let interpreter = CharArray::new_row("Interpreter");
        let values = vec![
            Value::CharArray(gpu),
            Value::CharArray(accel),
            Value::CharArray(ignition),
            Value::CharArray(interpreter),
        ];
        let cell = make_cell(values, 2, 2).expect("cell");
        let result = join_builtin(cell, vec![Value::String(", ".into())]).expect("join cell");
        match result {
            Value::Cell(cell_out) => {
                assert_eq!(cell_out.rows, 2);
                assert_eq!(cell_out.cols, 1);
                let first = unsafe { &*cell_out.data[0].as_raw() };
                let second = unsafe { &*cell_out.data[1].as_raw() };
                match (first, second) {
                    (Value::CharArray(a), Value::CharArray(b)) => {
                        assert_eq!(
                            char_row_to_string_slice(&a.data, a.cols, 0),
                            "GPU, Accelerate"
                        );
                        assert_eq!(
                            char_row_to_string_slice(&b.data, b.cols, 0),
                            "Ignition, Interpreter"
                        );
                    }
                    other => panic!("expected char arrays, got {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_with_numeric_second_argument_uses_default_delimiter() {
        let array = StringArray::new(
            vec!["RunMat".into(), "Accelerate".into(), "Planner".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = join_builtin(
            Value::StringArray(array),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1]);
                assert_eq!(sa.data, vec![String::from("RunMat Accelerate Planner")]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_char_array_input_produces_string_array() {
        let data: Vec<char> = "RunMatGPUDev".chars().collect();
        let char_array = CharArray::new(data, 3, 4).unwrap();
        let result = join_builtin(Value::CharArray(char_array), Vec::new()).expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1]);
                assert_eq!(sa.data, vec![String::from("RunM atGP UDev")]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_with_cell_delimiter_array() {
        let array = StringArray::new(
            vec![
                "g".into(),
                "c".into(),
                "w".into(),
                "gpu".into(),
                "cuda".into(),
                "wgpu".into(),
            ],
            vec![3, 2],
        )
        .unwrap();
        let delimiters = make_cell(
            vec![
                Value::String(String::from(" -> ")),
                Value::String(String::from(" => ")),
                Value::String(String::from(" :: ")),
            ],
            3,
            1,
        )
        .expect("cell");
        let result = join_builtin(
            Value::StringArray(array),
            vec![delimiters, Value::Int(IntValue::I32(2))],
        )
        .expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![3, 1]);
                assert_eq!(
                    sa.data,
                    vec![
                        String::from("g -> gpu"),
                        String::from("c => cuda"),
                        String::from("w :: wgpu")
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_3d_string_array_along_third_dimension() {
        let mut data = Vec::new();
        for page in 0..2 {
            for col in 0..2 {
                for row in 0..2 {
                    data.push(format!("r{row}c{col}p{page}"));
                }
            }
        }
        let array = StringArray::new(data, vec![2, 2, 2]).unwrap();
        let result = join_builtin(
            Value::StringArray(array),
            vec![Value::String(":".into()), Value::Int(IntValue::I32(3))],
        )
        .expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 2, 1]);
                let expected = vec![
                    String::from("r0c0p0:r0c0p1"),
                    String::from("r1c0p0:r1c0p1"),
                    String::from("r0c1p0:r0c1p1"),
                    String::from("r1c1p0:r1c1p1"),
                ];
                assert_eq!(sa.data, expected);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_errors_on_zero_dimension() {
        let array = StringArray::new(vec!["a".into()], vec![1, 1]).unwrap();
        let err = join_builtin(
            Value::StringArray(array),
            vec![Value::Int(IntValue::I32(0))],
        )
        .unwrap_err();
        let err_text = err.to_string();
        assert!(
            err_text.contains("dimension"),
            "expected dimension error, got {err_text}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_errors_on_mismatched_delimiter_shape() {
        let array = StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![1, 3]).unwrap();
        let delims =
            StringArray::new(vec!["+".into(), "-".into(), "=".into()], vec![1, 3]).unwrap();
        let result = join_builtin(Value::StringArray(array), vec![Value::StringArray(delims)]);
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_propagates_missing_strings() {
        let array = StringArray::new(vec!["GPU".into(), "<missing>".into()], vec![1, 2]).unwrap();
        let result = join_builtin(Value::StringArray(array), Vec::new()).expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data, vec![String::from("<missing>")]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_accepts_char_delimiter_scalar() {
        let array = StringArray::new(vec!["A".into(), "B".into()], vec![1, 2]).unwrap();
        let delimiter_chars = CharArray::new("++".chars().collect::<Vec<char>>(), 1, 2).unwrap();
        let result = join_builtin(
            Value::StringArray(array),
            vec![Value::CharArray(delimiter_chars)],
        )
        .expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data, vec![String::from("A++B")]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_handles_empty_axis() {
        let array = StringArray::new(Vec::new(), vec![2, 0]).unwrap();
        let result = join_builtin(Value::StringArray(array), Vec::new()).expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 1]);
                assert_eq!(sa.data, vec![String::from(""), String::from("")]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_missing_dimension_broadcast_delimiters() {
        let array = StringArray::new(
            vec!["aa".into(), "cc".into(), "bb".into(), "dd".into()],
            vec![2, 2],
        )
        .unwrap();
        let delims = StringArray::new(vec!["-".into()], vec![1, 1]).unwrap();
        let result = join_builtin(
            Value::StringArray(array),
            vec![Value::StringArray(delims), Value::Int(IntValue::I32(2))],
        )
        .expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 1]);
                assert_eq!(sa.data, vec![String::from("aa-bb"), String::from("cc-dd")]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn join_executes_with_wgpu_provider_registered() {
        let _ = wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default());
        let array = StringArray::new(vec!["GPU".into(), "Planner".into()], vec![2, 1]).unwrap();
        let result = join_builtin(Value::StringArray(array), Vec::new()).expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data, vec![String::from("GPU Planner")]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }
}
