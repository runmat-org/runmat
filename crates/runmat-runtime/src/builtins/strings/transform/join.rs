//! MATLAB-compatible `join` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, JOIN_ERROR_ARG_COUNT, JOIN_ERROR_DELIMITER_SIZE,
    JOIN_ERROR_DELIMITER_TYPE, JOIN_ERROR_DIMENSION_TYPE, JOIN_ERROR_INPUT_TYPE,
    JOIN_ERROR_INTERNAL, JOIN_RESIDENT_INPUT_EXTENSION, JOIN_REVERSED_ARGUMENTS_EXTENSION,
    JOIN_TYPED_INTEGER_DIMENSION_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, StringArray, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::{
    build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult, RuntimeError,
};

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

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

fn join_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn join_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    join_error_with_message(error.message, error)
}

#[runtime_builtin(
    name = "join",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::transform::join"
)]
async fn join_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    text: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::JOIN_DESCRIPTOR,
        &runmat_builtins::JOIN_ERROR_TOO_MANY_OUTPUTS,
        "join",
    )?;
    preflight_join_extensions(&text, rest.view())?;
    let text = gather_if_needed_async(&text).await.map_err(map_flow)?;
    let args = rest
        .try_map_preserving_roots_async(rest.len(), runtime.as_ref(), async |index| {
            gather_if_needed_async(&rest[index]).await.map_err(map_flow)
        })
        .await?;

    let mut input = JoinInput::from_value(text)?;
    let (delimiter_arg, dimension_arg) = parse_arguments(args.view())?;

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
        return Err(join_error(&JOIN_ERROR_DIMENSION_TYPE));
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

fn preflight_join_extensions(
    text: &Value,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if args.len() > 2 {
        return Err(join_error(&JOIN_ERROR_ARG_COUNT));
    }
    if !is_join_input(text)
        || crate::builtins::strings::common::has_unsupported_character_pages(text)
        || args
            .iter()
            .any(crate::builtins::strings::common::has_unsupported_character_pages)
    {
        return Err(join_error(&JOIN_ERROR_INPUT_TYPE));
    }

    let dimension = match (args.get(0), args.get(1)) {
        (None, None) => None,
        (Some(only), None) if is_dimension_candidate(only) => Some(only),
        (Some(delimiter), None) => {
            if !is_join_delimiter(delimiter) {
                return Err(join_error(&JOIN_ERROR_DELIMITER_TYPE));
            }
            None
        }
        (Some(delimiter), Some(dimension)) if is_dimension_candidate(dimension) => {
            if !is_join_delimiter(delimiter) {
                return Err(join_error(&JOIN_ERROR_DELIMITER_TYPE));
            }
            Some(dimension)
        }
        (Some(dimension), Some(delimiter)) if is_dimension_candidate(dimension) => {
            if !is_join_delimiter(delimiter) {
                return Err(join_error(&JOIN_ERROR_DELIMITER_TYPE));
            }
            crate::compatibility::ensure_builtin_extension_enabled(
                &JOIN_REVERSED_ARGUMENTS_EXTENSION,
                BUILTIN_NAME,
            )?;
            Some(dimension)
        }
        (Some(_), Some(_)) => return Err(join_error(&JOIN_ERROR_DIMENSION_TYPE)),
        _ => unreachable!("join arity checked above"),
    };

    if dimension.is_some_and(is_typed_integer_dimension_candidate) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &JOIN_TYPED_INTEGER_DIMENSION_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args.iter().any(value_contains_explicit_gpu) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &JOIN_RESIDENT_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn is_join_input(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => true,
        Value::Cell(cell) => cell
            .data()
            .iter()
            .all(|value| cell_element_to_string(value).ok().flatten().is_some()),
        _ => false,
    }
}

fn is_join_delimiter(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => true,
        Value::Cell(cell) => cell
            .data()
            .iter()
            .all(|value| cell_element_to_string(value).ok().flatten().is_some()),
        _ => false,
    }
}

fn value_contains_explicit_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_explicit(handle),
        Value::Cell(cell) => cell.data().iter().any(value_contains_explicit_gpu),
        Value::Struct(value) => value.fields.values().any(value_contains_explicit_gpu),
        Value::Object(value) => value.properties().values().any(value_contains_explicit_gpu),
        Value::Closure(value) => value.captures.iter().any(value_contains_explicit_gpu),
        Value::OutputList(values) => values.iter().any(value_contains_explicit_gpu),
        _ => false,
    }
}

fn is_typed_integer_dimension_candidate(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => {
            tensor::is_scalar_tensor(tensor) && tensor.integer_storage().is_some()
        }
        Value::GpuTensor(handle) => {
            handle.shape().iter().all(|&dimension| dimension == 1)
                && runmat_accelerate_api::handle_integer_type(handle).is_some()
        }
        _ => false,
    }
}

fn is_dimension_candidate(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) => true,
        Value::Tensor(tensor) => tensor::is_scalar_tensor(tensor),
        Value::GpuTensor(handle) => handle.shape().iter().all(|&dimension| dimension == 1),
        _ => false,
    }
}

fn parse_arguments(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(Option<Value>, Option<usize>)> {
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
                Err(join_error(&JOIN_ERROR_DIMENSION_TYPE))
            }
        }
        _ => Err(join_error(&JOIN_ERROR_ARG_COUNT)),
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
            let v = i
                .try_to_usize()
                .filter(|value| *value > 0)
                .ok_or_else(|| join_error(&JOIN_ERROR_DIMENSION_TYPE))?;
            Ok(Some(v))
        }
        Value::Num(n) => {
            if !n.is_finite() || *n <= 0.0 {
                return Err(join_error(&JOIN_ERROR_DIMENSION_TYPE));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(join_error(&JOIN_ERROR_DIMENSION_TYPE));
            }
            parse_dimension_float(rounded)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                let dim = int
                    .try_to_usize()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| join_error(&JOIN_ERROR_DIMENSION_TYPE))?;
                return Ok(Some(dim));
            }
            let val = tensor::tensor_value_f64(t, 0);
            if !val.is_finite() || val <= 0.0 {
                return Err(join_error(&JOIN_ERROR_DIMENSION_TYPE));
            }
            let rounded = val.round();
            if (rounded - val).abs() > f64::EPSILON {
                return Err(join_error(&JOIN_ERROR_DIMENSION_TYPE));
            }
            parse_dimension_float(rounded)
        }
        _ => Ok(None),
    }
}

fn parse_dimension_float(rounded: f64) -> BuiltinResult<Option<usize>> {
    if rounded > usize::MAX.saturating_sub(1) as f64 {
        return Err(join_error(&JOIN_ERROR_DIMENSION_TYPE));
    }
    let parsed = rounded as usize;
    if parsed as f64 != rounded || parsed == usize::MAX {
        return Err(join_error(&JOIN_ERROR_DIMENSION_TYPE));
    }
    Ok(Some(parsed))
}

struct JoinInput {
    data: Vec<runmat_value::StringElement>,
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
            Value::StringArray(array) => {
                let shape = array.shape().to_vec();
                let (owner, _shape, _rows, _cols) = array.into_parts();
                let data = match owner.try_into_owned() {
                    Ok(values) => values.into_iter().collect(),
                    Err(data) => data.iter().cloned().collect(),
                };
                Ok(Self {
                    data,
                    shape,
                    kind: OutputKind::StringArray,
                })
            }
            Value::CharArray(array) => {
                let strings = char_array_rows_to_strings(&array)?;
                Ok(Self {
                    data: strings.into_iter().map(Into::into).collect(),
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
            _ => Err(join_error(&JOIN_ERROR_INPUT_TYPE)),
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

    fn build_output(
        &self,
        data: Vec<runmat_value::StringElement>,
        shape: Vec<usize>,
    ) -> BuiltinResult<Value> {
        build_value(self.kind.clone(), data, shape)
    }
}

fn build_value(
    kind: OutputKind,
    data: Vec<runmat_value::StringElement>,
    shape: Vec<usize>,
) -> BuiltinResult<Value> {
    match kind {
        OutputKind::StringScalar => Ok(Value::string(data.into_iter().next().unwrap_or_default())),
        OutputKind::StringArray => {
            let array = StringArray::from_elements_unbounded(
                data.into_iter().map(Into::into).collect(),
                shape,
            )
            .map_err(|e| {
                join_error_with_message(format!("{BUILTIN_NAME}: {e}"), &JOIN_ERROR_INTERNAL)
            })?;
            Ok(Value::StringArray(array))
        }
        OutputKind::CellArray => {
            let rows = shape.first().copied().unwrap_or(0);
            let cols = shape.get(1).copied().unwrap_or(1);
            if rows == 0 || cols == 0 || data.is_empty() {
                return make_cell_with_shape(Vec::new(), shape).map_err(|e| {
                    join_error_with_message(format!("{BUILTIN_NAME}: {e}"), &JOIN_ERROR_INTERNAL)
                });
            }
            let page_len = rows
                .checked_mul(cols)
                .ok_or_else(|| join_error(&JOIN_ERROR_INTERNAL))?;
            if !data.len().is_multiple_of(page_len) {
                return Err(join_error(&JOIN_ERROR_INTERNAL));
            }
            let mut values = Vec::with_capacity(data.len());
            for page in data.chunks_exact(page_len) {
                for row in 0..rows {
                    for col in 0..cols {
                        let idx = row + col * rows;
                        let Some(text) = page[idx].as_str() else {
                            values.push(Value::String(runmat_value::StringElement::Missing));
                            continue;
                        };
                        let chars: Vec<char> = text.chars().collect();
                        let cols_count = text.encode_utf16().count();
                        let char_array = CharArray::from_unicode_scalars(chars, 1, cols_count)
                            .map_err(|e| {
                                join_error_with_message(
                                    format!("{BUILTIN_NAME}: {e}"),
                                    &JOIN_ERROR_INTERNAL,
                                )
                            })?;
                        values.push(Value::CharArray(char_array));
                    }
                }
            }
            make_cell_with_shape(values, shape).map_err(|e| {
                join_error_with_message(format!("{BUILTIN_NAME}: {e}"), &JOIN_ERROR_INTERNAL)
            })
        }
    }
}

fn char_array_rows_to_strings(array: &CharArray) -> BuiltinResult<Vec<String>> {
    let mut strings = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        strings.push(
            char_row_to_string_slice(array.data(), array.cols, row).map_err(|e| {
                join_error_with_message(format!("{BUILTIN_NAME}: {e}"), &JOIN_ERROR_INTERNAL)
            })?,
        );
    }
    Ok(strings)
}

fn cell_array_to_strings(
    cell: CellArray,
) -> BuiltinResult<(Vec<runmat_value::StringElement>, Vec<usize>)> {
    let (data, shape, rows, cols) = cell.into_parts();
    let page_len = rows
        .checked_mul(cols)
        .ok_or_else(|| join_error(&JOIN_ERROR_INTERNAL))?;
    let mut strings = Vec::with_capacity(data.len());
    if page_len == 0 {
        return Ok((strings, shape.to_vec()));
    }
    if !data.len().is_multiple_of(page_len) {
        return Err(join_error(&JOIN_ERROR_INTERNAL));
    }
    for page in data.chunks_exact(page_len) {
        for col in 0..cols {
            for row in 0..rows {
                let idx = row * cols + col;
                strings.push(match &page[idx] {
                    Value::String(text) => text.clone(),
                    Value::StringArray(array) if array.data().len() == 1 => array.data()[0].clone(),
                    value => cell_element_to_string(value)?
                        .ok_or_else(|| join_error(&JOIN_ERROR_INPUT_TYPE))?
                        .into(),
                });
            }
        }
    }
    Ok((strings, shape.to_vec()))
}

fn cell_element_to_string(value: &Value) -> BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(text) => text.as_str().map(str::to_owned),
        Value::StringArray(array) if array.data().len() == 1 => {
            array.data()[0].as_str().map(str::to_owned)
        }
        Value::CharArray(array) if array.rows <= 1 => {
            if array.rows == 0 {
                Some(String::new())
            } else {
                Some(
                    char_row_to_string_slice(array.data(), array.cols, 0).map_err(|e| {
                        join_error_with_message(
                            format!("{BUILTIN_NAME}: {e}"),
                            &JOIN_ERROR_INTERNAL,
                        )
                    })?,
                )
            }
        }
        _ => None,
    })
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
                if let Some(text) = value_to_scalar_string(&v)? {
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

fn value_to_scalar_string(value: &Value) -> BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(text) => text.as_str().map(str::to_owned),
        Value::CharArray(array) if array.rows <= 1 => {
            if array.rows == 0 {
                Some(String::new())
            } else {
                Some(
                    char_row_to_string_slice(array.data(), array.cols, 0).map_err(|e| {
                        join_error_with_message(
                            format!("{BUILTIN_NAME}: {e}"),
                            &JOIN_ERROR_INTERNAL,
                        )
                    })?,
                )
            }
        }
        Value::StringArray(array) if array.data().len() == 1 => {
            array.data()[0].as_str().map(str::to_owned)
        }
        Value::Cell(cell) if cell.data().len() == 1 => cell_element_to_string(&cell.data()[0])?,
        _ => None,
    })
}

fn value_to_string_array(value: Value) -> BuiltinResult<(Vec<String>, Vec<usize>)> {
    match value {
        Value::StringArray(array) => Ok((
            array
                .data()
                .iter()
                .map(|text| {
                    text.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| join_error(&JOIN_ERROR_DELIMITER_TYPE))
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
            array.shape.to_vec(),
        )),
        Value::Cell(cell) => {
            let (data, shape) = cell_array_to_strings(cell)?;
            Ok((
                data.into_iter()
                    .map(|text| {
                        text.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| join_error(&JOIN_ERROR_DELIMITER_TYPE))
                    })
                    .collect::<BuiltinResult<Vec<_>>>()?,
                shape,
            ))
        }
        Value::CharArray(array) => {
            let rows = array.rows;
            let strings = char_array_rows_to_strings(&array)?;
            Ok((strings, vec![rows, 1]))
        }
        _ => Err(join_error(&JOIN_ERROR_DELIMITER_TYPE)),
    }
}

fn normalize_delimiter_shape(
    mut shape: Vec<usize>,
    full_shape: &[usize],
    axis_idx: usize,
) -> BuiltinResult<Vec<usize>> {
    if shape.len() > full_shape.len() {
        return Err(join_error(&JOIN_ERROR_DELIMITER_SIZE));
    }
    if shape.len() < full_shape.len() {
        shape.resize(full_shape.len(), 1);
    }

    let axis_len = full_shape[axis_idx].saturating_sub(1);
    if axis_len == 0 {
        shape[axis_idx] = 1;
    } else if shape[axis_idx] != axis_len {
        return Err(join_error(&JOIN_ERROR_DELIMITER_SIZE));
    }

    for (dim, size) in shape.iter().enumerate() {
        if dim == axis_idx {
            continue;
        }
        let reference = full_shape[dim];
        if *size != reference && *size != 1 {
            return Err(join_error(&JOIN_ERROR_DELIMITER_SIZE));
        }
    }

    Ok(shape)
}

fn perform_join(
    data: &[runmat_value::StringElement],
    full_shape: &[usize],
    axis_idx: usize,
    delimiter: &Delimiter,
) -> (Vec<runmat_value::StringElement>, Vec<usize>) {
    if full_shape.is_empty() {
        return (vec![runmat_value::StringElement::text("")], vec![1, 1]);
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
            output.push(runmat_value::StringElement::text(""));
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
                result.push_str(value.as_str().expect("nonmissing join element"));
            }
            if missing {
                output.push(runmat_value::StringElement::Missing);
            } else {
                output.push(runmat_value::StringElement::text(result));
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
    use crate::make_cell;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_backend;
    use runmat_builtins::{
        BuiltinIntegerInputAvailability, JOIN_DESCRIPTOR, JOIN_EXTENSIONS,
        JOIN_INTEGER_CAPABILITIES,
    };
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    fn join_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::join_builtin(
            None,
            text,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
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
                    sa.data(),
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
                assert_eq!(
                    sa.data(),
                    vec![String::from("x-y-z"), String::from("a-b-c")]
                );
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
                    sa.data(),
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
        let result = join_builtin(Value::StringArray(array), vec![Value::Num(1.0)]).expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 2]);
                assert_eq!(
                    sa.data(),
                    vec![
                        String::from("Carlos Ella Diana"),
                        String::from("Sada Olsen Lee"),
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn join_dimension_parser_preserves_typed_integer_tensor_bounds() {
        let dim = Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).expect("dim");
        assert_eq!(value_to_dimension(&Value::Tensor(dim)).unwrap(), Some(2));

        let zero = Tensor::new_integer(IntegerStorage::U64(vec![0]), vec![1, 1]).expect("dim");
        assert!(value_to_dimension(&Value::Tensor(zero)).is_err());

        let negative =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("negative dim");
        assert!(value_to_dimension(&Value::Tensor(negative)).is_err());
        assert!(value_to_dimension(&Value::Num(1.0e300)).is_err());
    }

    #[test]
    fn join_typed_integer_dimension_is_gated_before_evaluation() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let input = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
        let error = join_builtin(
            Value::StringArray(input),
            vec![Value::Int(IntValue::U64(u64::MAX))],
        )
        .expect_err("typed integer dimension extension");
        assert_eq!(
            error.identifier(),
            JOIN_TYPED_INTEGER_DIMENSION_EXTENSION.error_identifier
        );
    }

    #[test]
    fn join_resident_typed_dimension_gate_precedes_provider_access() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::U64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let input = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
        let error = join_builtin(
            Value::StringArray(input),
            vec![Value::GpuTensor(allocation.handle.clone())],
        )
        .expect_err("resident typed dimension extension");
        allocation
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        assert_eq!(allocation.probe.retirements(), 0);
        assert_eq!(
            error.identifier(),
            JOIN_TYPED_INTEGER_DIMENSION_EXTENSION.error_identifier
        );
    }

    #[test]
    fn join_only_explicit_residency_is_compatibility_gated() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let automatic = allocation
            .handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .unwrap();
        let input = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
        let automatic_error = join_builtin(
            Value::StringArray(input.clone()),
            vec![Value::GpuTensor(automatic.clone())],
        )
        .expect_err("automatic residency may proceed to owner lookup");
        assert_ne!(
            automatic_error.identifier(),
            JOIN_RESIDENT_INPUT_EXTENSION.error_identifier
        );

        let explicit = allocation
            .handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        let explicit_error = join_builtin(
            Value::StringArray(input),
            vec![Value::GpuTensor(explicit.clone())],
        )
        .expect_err("explicit residency must be compatibility gated");
        assert_eq!(
            explicit_error.identifier(),
            JOIN_RESIDENT_INPUT_EXTENSION.error_identifier
        );
        drop(automatic);
        drop(explicit);
        allocation
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        assert_eq!(allocation.probe.retirements(), 0);
    }

    #[test]
    fn join_delimiter_role_error_precedes_dimension_extension() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let input = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
        let error = join_builtin(
            Value::StringArray(input),
            vec![Value::Int(IntValue::U8(1)), Value::Num(2.0)],
        )
        .expect_err("numeric delimiter must reject as a delimiter");
        assert_eq!(error.identifier(), JOIN_ERROR_DELIMITER_TYPE.identifier);
    }

    #[test]
    fn join_dimension_before_delimiter_is_a_gated_extension() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let input = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
        let error = join_builtin(
            Value::StringArray(input),
            vec![Value::Num(2.0), Value::from("-")],
        )
        .expect_err("reversed join syntax extension");
        assert_eq!(
            error.identifier(),
            JOIN_REVERSED_ARGUMENTS_EXTENSION.error_identifier
        );
    }

    #[test]
    fn join_integer_metadata_records_exact_runmat_only_dimension() {
        assert_eq!(JOIN_INTEGER_CAPABILITIES.len(), 1);
        assert_eq!(
            JOIN_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
        assert_eq!(JOIN_EXTENSIONS.len(), 3);
        assert_eq!(JOIN_DESCRIPTOR.signatures.len(), 4);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_dimension_greater_than_ndims_returns_input() {
        let array = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
        let result =
            join_builtin(Value::StringArray(array.clone()), vec![Value::Num(4.0)]).expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, array.shape);
                assert_eq!(sa.data(), array.data());
            }
            other => panic!("expected original array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_cell_array_of_char_vectors() {
        let gpu = CharArray::new_row("GPU");
        let accel = CharArray::new_row("Accelerate");
        let vm_label = CharArray::new_row("VM");
        let interpreter = CharArray::new_row("Interpreter");
        let values = vec![
            Value::CharArray(gpu),
            Value::CharArray(accel),
            Value::CharArray(vm_label),
            Value::CharArray(interpreter),
        ];
        let cell = make_cell(values, 2, 2).expect("cell");
        let result = join_builtin(cell, vec![Value::String(", ".into())]).expect("join cell");
        match result {
            Value::Cell(cell_out) => {
                assert_eq!(cell_out.rows, 2);
                assert_eq!(cell_out.cols, 1);
                let first = &cell_out.data()[0];
                let second = &cell_out.data()[1];
                match (first, second) {
                    (Value::CharArray(a), Value::CharArray(b)) => {
                        assert_eq!(
                            char_row_to_string_slice(a.data(), a.cols, 0).unwrap(),
                            "GPU, Accelerate"
                        );
                        assert_eq!(
                            char_row_to_string_slice(b.data(), b.cols, 0).unwrap(),
                            "VM, Interpreter"
                        );
                    }
                    other => panic!("expected char arrays, got {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn join_preserves_nd_cell_pages_through_both_layout_bridges() {
        let input = make_cell_with_shape(
            vec!["a", "b", "c", "d"]
                .into_iter()
                .map(|text| Value::CharArray(CharArray::new_row(text)))
                .collect(),
            vec![1, 2, 2],
        )
        .expect("nd text cells");
        let Value::Cell(output) =
            join_builtin(input, vec![Value::string(","), Value::Num(2.0)]).expect("join each page")
        else {
            panic!("cell output");
        };
        assert_eq!(output.shape(), &[1, 1, 2]);
        assert_eq!(output.data().len(), 2);
        assert_eq!(
            output.data()[0],
            Value::CharArray(CharArray::new_row("a,b"))
        );
        assert_eq!(
            output.data()[1],
            Value::CharArray(CharArray::new_row("c,d"))
        );
        let pages = Value::CharArray(
            CharArray::from_unicode_scalars_with_shape(vec!['x', 'y'], vec![1, 1, 2])
                .expect("character pages"),
        );
        let nested = make_cell_with_shape(vec![pages], vec![1, 1]).expect("nested character pages");
        let error =
            join_builtin(nested, vec![]).expect_err("do not drop character pages inside cells");
        assert_eq!(error.identifier(), JOIN_ERROR_INPUT_TYPE.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_with_numeric_second_argument_uses_default_delimiter() {
        let array = StringArray::new(
            vec!["RunMat".into(), "Accelerate".into(), "Planner".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = join_builtin(Value::StringArray(array), vec![Value::Num(1.0)]).expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1]);
                assert_eq!(sa.data(), vec![String::from("RunMat Accelerate Planner")]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_char_array_input_produces_string_array() {
        let data: Vec<char> = "RunMatGPUDev".chars().collect();
        let char_array = CharArray::from_unicode_scalars(data, 3, 4).unwrap();
        let result = join_builtin(Value::CharArray(char_array), Vec::new()).expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1]);
                assert_eq!(sa.data(), vec![String::from("RunM atGP UDev")]);
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
                Value::String(String::from(" -> ").into()),
                Value::String(String::from(" => ").into()),
                Value::String(String::from(" :: ").into()),
            ],
            3,
            1,
        )
        .expect("cell");
        let result = join_builtin(Value::StringArray(array), vec![delimiters, Value::Num(2.0)])
            .expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![3, 1]);
                assert_eq!(
                    sa.data(),
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
            vec![Value::String(":".into()), Value::Num(3.0)],
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
                assert_eq!(sa.data(), expected);
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
        let array = StringArray::from_elements_unbounded(
            vec!["GPU".into(), runmat_value::StringElement::Missing],
            vec![1, 2],
        )
        .unwrap();
        let result = join_builtin(Value::StringArray(array), Vec::new()).expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data(), &[runmat_value::StringElement::Missing]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_accepts_char_delimiter_scalar() {
        let array = StringArray::new(vec!["A".into(), "B".into()], vec![1, 2]).unwrap();
        let delimiter_chars =
            CharArray::from_unicode_scalars("++".chars().collect::<Vec<char>>(), 1, 2).unwrap();
        let result = join_builtin(
            Value::StringArray(array),
            vec![Value::CharArray(delimiter_chars)],
        )
        .expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data(), vec![String::from("A++B")]);
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
                assert_eq!(sa.data(), vec![String::from(""), String::from("")]);
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
            vec![Value::StringArray(delims), Value::Num(2.0)],
        )
        .expect("join");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 1]);
                assert_eq!(
                    sa.data(),
                    vec![String::from("aa-bb"), String::from("cc-dd")]
                );
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
                assert_eq!(sa.data(), vec![String::from("GPU Planner")]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }
}
