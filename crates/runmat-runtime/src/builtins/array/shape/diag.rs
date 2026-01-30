//! MATLAB-compatible `diag` builtin.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

const MESSAGE_ID_INVALID_INPUT: &str = "MATLAB:diag:InvalidInput";
const MESSAGE_ID_INVALID_OFFSET: &str = "MATLAB:diag:InvalidOffset";

fn diag_type(args: &[Type]) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { shape: Some(shape) } => {
            if shape.len() == 1
                || (shape.len() >= 2 && (shape[0] == Some(1) || shape[1] == Some(1)))
            {
                let len = shape
                    .get(0)
                    .copied()
                    .flatten()
                    .or_else(|| shape.get(1).copied().flatten());
                if let Some(n) = len {
                    return Type::Tensor {
                        shape: Some(vec![Some(n), Some(n)]),
                    };
                }
            }
            Type::tensor()
        }
        Type::Logical { shape: Some(shape) } => {
            if shape.len() == 1
                || (shape.len() >= 2 && (shape[0] == Some(1) || shape[1] == Some(1)))
            {
                let len = shape
                    .get(0)
                    .copied()
                    .flatten()
                    .or_else(|| shape.get(1).copied().flatten());
                if let Some(n) = len {
                    return Type::Logical {
                        shape: Some(vec![Some(n), Some(n)]),
                    };
                }
            }
            Type::logical()
        }
        Type::Num | Type::Int | Type::Bool => Type::tensor(),
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::diag")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "diag",
    op_kind: GpuOpKind::Custom("diag"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "diag executes on the host and gathers GPU inputs first.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::diag")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "diag",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "diag is a host-only shape helper.",
};

fn diag_error(message_id: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(message_id)
        .with_builtin("diag")
        .build()
}

#[runtime_builtin(
    name = "diag",
    category = "array/shape",
    summary = "Extract or create a diagonal from a vector or matrix.",
    keywords = "diag,diagonal,matrix",
    type_resolver(diag_type),
    builtin_path = "crate::builtins::array::shape::diag"
)]
async fn diag_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: expected at most two inputs",
        ));
    }
    let offset = parse_offset(&rest).await?;
    let gathered = gather_if_needed_async(&value).await?;
    match gathered {
        Value::Tensor(tensor) => diag_tensor_value(tensor, offset).map(Value::Tensor),
        Value::LogicalArray(array) => diag_logical_value(array, offset).map(Value::LogicalArray),
        Value::ComplexTensor(tensor) => {
            diag_complex_value(tensor, offset).map(Value::ComplexTensor)
        }
        other => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            format!("diag: unsupported input {other:?}"),
        )),
    }
}

async fn parse_offset(rest: &[Value]) -> BuiltinResult<isize> {
    if rest.is_empty() {
        return Ok(0);
    }
    let gathered = gather_if_needed_async(&rest[0]).await?;
    scalar_to_isize(&gathered)
}

fn scalar_to_isize(value: &Value) -> BuiltinResult<isize> {
    match value {
        Value::Int(i) => Ok(i.to_i64() as isize),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(diag_error(
                    MESSAGE_ID_INVALID_OFFSET,
                    "diag: diagonal offset must be finite",
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(diag_error(
                    MESSAGE_ID_INVALID_OFFSET,
                    "diag: diagonal offset must be an integer",
                ));
            }
            Ok(rounded as isize)
        }
        Value::Tensor(t) if t.data.len() == 1 => scalar_to_isize(&Value::Num(t.data[0])),
        Value::Bool(flag) => Ok(if *flag { 1 } else { 0 }),
        other => Err(diag_error(
            MESSAGE_ID_INVALID_OFFSET,
            format!("diag: diagonal offset must be a numeric scalar, got {other:?}"),
        )),
    }
}

fn diag_tensor_value(tensor: Tensor, offset: isize) -> BuiltinResult<Tensor> {
    let (rows, cols) = matrix_dims(&tensor.shape)?;
    if rows == 1 || cols == 1 {
        let len = rows.max(cols);
        let data = diag_matrix_from_vector(&tensor.data, len, offset, 0.0);
        let size = len + offset.unsigned_abs();
        Tensor::new(data, vec![size, size])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
    } else {
        let data = diag_vector_from_matrix(&tensor.data, rows, cols, offset);
        let len = data.len();
        Tensor::new(data, vec![len, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
    }
}

fn diag_logical_value(array: LogicalArray, offset: isize) -> BuiltinResult<LogicalArray> {
    let (rows, cols) = matrix_dims(&array.shape)?;
    if rows == 1 || cols == 1 {
        let len = rows.max(cols);
        let data = diag_matrix_from_vector(&array.data, len, offset, 0u8);
        let size = len + offset.unsigned_abs();
        LogicalArray::new(data, vec![size, size])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
    } else {
        let diag = diag_vector_from_matrix(&array.data, rows, cols, offset);
        let len = diag.len();
        LogicalArray::new(diag, vec![len, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
    }
}

fn diag_complex_value(tensor: ComplexTensor, offset: isize) -> BuiltinResult<ComplexTensor> {
    let (rows, cols) = matrix_dims(&tensor.shape)?;
    if rows == 1 || cols == 1 {
        let len = rows.max(cols);
        let data = diag_matrix_from_vector(&tensor.data, len, offset, (0.0, 0.0));
        let size = len + offset.unsigned_abs();
        ComplexTensor::new(data, vec![size, size])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
    } else {
        let diag = diag_vector_from_matrix(&tensor.data, rows, cols, offset);
        let len = diag.len();
        ComplexTensor::new(diag, vec![len, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
    }
}

fn matrix_dims(shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    if shape.len() > 2 {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: only vectors and matrices are supported",
        ));
    }
    let rows = *shape.get(0).unwrap_or(&1);
    let cols = *shape.get(1).unwrap_or(&1);
    Ok((rows, cols))
}

fn diag_matrix_from_vector<T: Copy>(data: &[T], len: usize, offset: isize, zero: T) -> Vec<T> {
    let shift = offset.unsigned_abs();
    let size = len + shift;
    let mut out = vec![zero; size * size];
    for idx in 0..len.min(data.len()) {
        let (row, col) = if offset >= 0 {
            (idx, idx + shift)
        } else {
            (idx + shift, idx)
        };
        out[row + col * size] = data[idx];
    }
    out
}

fn diag_vector_from_matrix<T: Copy>(data: &[T], rows: usize, cols: usize, offset: isize) -> Vec<T> {
    let shift = offset.unsigned_abs();
    let (start_row, start_col) = if offset >= 0 {
        (0usize, shift)
    } else {
        (shift, 0usize)
    };
    if start_row >= rows || start_col >= cols {
        return Vec::new();
    }
    let max_len = (rows - start_row).min(cols - start_col);
    let mut out = Vec::with_capacity(max_len);
    for idx in 0..max_len {
        let row = start_row + idx;
        let col = start_col + idx;
        out.push(data[row + col * rows]);
    }
    out
}
