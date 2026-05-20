//! MATLAB-compatible `repelem` builtin for RunMat.
//!
//! Replicates each element of an array along one or more dimensions, supporting
//! both scalar and per-element replication counts. Mirrors the type coverage of
//! `repmat`: `Tensor` (any numeric dtype), `LogicalArray`, `ComplexTensor`,
//! `StringArray`, `CharArray`, and `CellArray`. RunMat does not maintain a
//! distinct `IntTensor`/`BoolTensor` runtime value (integer dtypes ride on
//! `Tensor` via `NumericDType`, booleans ride on `LogicalArray`), so the same
//! coverage applies here. GPU inputs are gathered to host memory before
//! replication runs because there is no native provider hook today.

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::ResolveContext;
use runmat_builtins::{
    CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::repelem")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "repelem",
    op_kind: GpuOpKind::Custom("repelem"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "repelem executes on the host today; GPU inputs are gathered before replication.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::repelem")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "repelem",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "repelem produces a fresh array; fusion treats it as a residency boundary.",
};

fn repelem_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("repelem").build()
}

fn array_shape(ty: &Type) -> Option<&[Option<usize>]> {
    match ty {
        Type::Tensor { shape: Some(shape) } => Some(shape.as_slice()),
        Type::Logical { shape: Some(shape) } => Some(shape.as_slice()),
        _ => None,
    }
}

/// Output shape for repelem given a known input shape and the number of
/// replication arguments. Like `repmat`, exact dimensions are unknown unless
/// the input is empty, so we encode rank only.
fn repelem_output_shape(
    input_shape: &[Option<usize>],
    reps_len: usize,
) -> Option<Vec<Option<usize>>> {
    let input_rank = input_shape.len();
    let rank = if reps_len == 1 {
        if input_rank == 0 {
            return None;
        }
        input_rank.max(2)
    } else {
        input_rank.max(reps_len.max(2))
    };

    let mut output = Vec::with_capacity(rank);
    for axis in 0..rank {
        if axis < input_rank && input_shape[axis] == Some(0) {
            output.push(Some(0));
        } else {
            output.push(None);
        }
    }
    Some(output)
}

fn repelem_reps_len(args: &[Type]) -> Option<usize> {
    if args.len() < 2 {
        return None;
    }
    if args.len() > 2 {
        return Some(args.len() - 1);
    }
    // Single replication argument: scalar OR vector. Either way reps_len == 1
    // for the output-rank purposes (we replicate along the vector's axis).
    match &args[1] {
        Type::Tensor { shape: Some(_) }
        | Type::Logical { shape: Some(_) }
        | Type::Num
        | Type::Int
        | Type::Bool => Some(1),
        _ => None,
    }
}

fn repelem_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    let reps_len = repelem_reps_len(args);
    let shape = array_shape(input)
        .and_then(|shape| reps_len.and_then(|len| repelem_output_shape(shape, len)));
    match input {
        Type::Tensor { .. } => Type::Tensor { shape },
        Type::Logical { .. } => Type::Logical { shape },
        Type::Bool => Type::logical(),
        Type::Num | Type::Int => Type::tensor(),
        Type::Cell { element_type, .. } => Type::Cell {
            element_type: element_type.clone(),
            length: None,
        },
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

#[runtime_builtin(
    name = "repelem",
    category = "array/shape",
    summary = "Replicate each element of an array along one or more dimensions.",
    keywords = "repelem,replicate,kron,tile,array",
    accel = "custom",
    type_resolver(repelem_type),
    builtin_path = "crate::builtins::array::shape::repelem"
)]
async fn repelem_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        return Err(repelem_error(
            "repelem: replication factors must be specified",
        ));
    }
    let factors = parse_factor_args(&rest).await?;
    // Treat single-argument form as a vector shorthand (only valid when the
    // input array is a vector; the per-type implementations enforce this).
    let single_arg = factors.len() == 1;

    match value {
        Value::Tensor(t) => {
            let out = repelem_tensor(&t, &factors, single_arg)?;
            Ok(tensor::tensor_into_value(out))
        }
        Value::Num(_) | Value::Int(_) => {
            let tensor = tensor::value_into_tensor_for("repelem", value).map_err(repelem_error)?;
            let out = repelem_tensor(&tensor, &factors, single_arg)?;
            Ok(tensor::tensor_into_value(out))
        }
        Value::Bool(flag) => {
            let logical = LogicalArray::new(vec![if flag { 1 } else { 0 }], vec![1, 1])
                .map_err(|e| repelem_error(format!("repelem: {e}")))?;
            let out = repelem_logical(&logical, &factors, single_arg)?;
            Ok(Value::LogicalArray(out))
        }
        Value::LogicalArray(logical) => {
            let out = repelem_logical(&logical, &factors, single_arg)?;
            Ok(Value::LogicalArray(out))
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| repelem_error(format!("repelem: {e}")))?;
            let out = repelem_complex_tensor(&tensor, &factors, single_arg)?;
            Ok(complex_tensor_into_value(out))
        }
        Value::ComplexTensor(ct) => {
            let out = repelem_complex_tensor(&ct, &factors, single_arg)?;
            Ok(complex_tensor_into_value(out))
        }
        Value::String(s) => {
            let array = StringArray::new(vec![s], vec![1, 1])
                .map_err(|e| repelem_error(format!("repelem: {e}")))?;
            let out = repelem_string_array(&array, &factors, single_arg)?;
            Ok(Value::StringArray(out))
        }
        Value::StringArray(sa) => {
            let out = repelem_string_array(&sa, &factors, single_arg)?;
            Ok(Value::StringArray(out))
        }
        Value::CharArray(ca) => {
            let out = repelem_char_array(&ca, &factors, single_arg)?;
            Ok(Value::CharArray(out))
        }
        Value::Cell(ca) => {
            let out = repelem_cell_array(&ca, &factors, single_arg)?;
            Ok(Value::Cell(out))
        }
        Value::GpuTensor(_) => Err(repelem_error(
            "repelem: GPU tensors must be gathered before replication; \
             expected a host residency hint from the planner",
        )),
        other => Err(repelem_error(format!(
            "repelem: unsupported input type {:?}",
            other
        ))),
    }
}

#[derive(Debug, Clone)]
enum RepFactor {
    Scalar(usize),
    Vector(Vec<usize>),
}

async fn parse_factor_args(args: &[Value]) -> crate::BuiltinResult<Vec<RepFactor>> {
    let mut out = Vec::with_capacity(args.len());
    for (idx, value) in args.iter().enumerate() {
        out.push(parse_single_factor(value, idx + 1).await?);
    }
    Ok(out)
}

async fn parse_single_factor(value: &Value, position: usize) -> crate::BuiltinResult<RepFactor> {
    match value {
        Value::Num(n) => Ok(RepFactor::Scalar(coerce_count(*n, position)?)),
        Value::Int(i) => Ok(RepFactor::Scalar(coerce_count(i.to_f64(), position)?)),
        Value::Bool(b) => Ok(RepFactor::Scalar(if *b { 1 } else { 0 })),
        Value::Tensor(tensor) => {
            if tensor.data.len() == 1 {
                Ok(RepFactor::Scalar(coerce_count(tensor.data[0], position)?))
            } else {
                ensure_vector_shape(&tensor.shape, position)?;
                let mut out = Vec::with_capacity(tensor.data.len());
                for &v in &tensor.data {
                    out.push(coerce_count(v, position)?);
                }
                Ok(RepFactor::Vector(out))
            }
        }
        Value::LogicalArray(la) => {
            if la.data.len() == 1 {
                Ok(RepFactor::Scalar(if la.data[0] != 0 { 1 } else { 0 }))
            } else {
                ensure_vector_shape(&la.shape, position)?;
                Ok(RepFactor::Vector(
                    la.data.iter().map(|&b| (b != 0) as usize).collect(),
                ))
            }
        }
        Value::GpuTensor(_) => {
            // Gather to host before reading dimensions so the planner's
            // GatherImmediately hint kicks in for repelem inputs too.
            let raw = tensor::scalar_f64_from_value_async(value)
                .await
                .map_err(|e| repelem_error(format!("repelem: {e}")))?;
            match raw {
                Some(n) => Ok(RepFactor::Scalar(coerce_count(n, position)?)),
                None => Err(repelem_error(format!(
                    "repelem: replication argument {position} must reside on the host"
                ))),
            }
        }
        other => Err(repelem_error(format!(
            "repelem: replication argument {position} must be numeric, got {:?}",
            other
        ))),
    }
}

fn ensure_vector_shape(shape: &[usize], position: usize) -> crate::BuiltinResult<()> {
    let non_singleton = shape.iter().filter(|&&d| d > 1).count();
    if non_singleton > 1 {
        return Err(repelem_error(format!(
            "repelem: replication argument {position} must be a scalar or vector"
        )));
    }
    Ok(())
}

fn coerce_count(value: f64, position: usize) -> crate::BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(repelem_error(format!(
            "repelem: replication count at argument {position} must be finite"
        )));
    }
    let rounded = value.round();
    let tolerance = (f64::EPSILON * value.abs().max(1.0)).min(1e-9);
    if (rounded - value).abs() > tolerance {
        return Err(repelem_error(format!(
            "repelem: replication count at argument {position} must be an integer"
        )));
    }
    if rounded < 0.0 {
        return Err(repelem_error(format!(
            "repelem: replication count at argument {position} must be non-negative"
        )));
    }
    if rounded > (usize::MAX as f64) {
        return Err(repelem_error(format!(
            "repelem: replication count at argument {position} exceeds the maximum supported size"
        )));
    }
    Ok(rounded as usize)
}

/// Determine which axis to replicate along when `repelem(v, n)` is called with
/// a single replication argument. Errors if `shape` is not a vector. Returns
/// the axis (0-based) along which `n` is applied; the other axis stays at 1.
fn vector_replication_axis(shape: &[usize]) -> crate::BuiltinResult<usize> {
    let total: usize = shape.iter().product();
    let non_singleton = shape.iter().filter(|&&d| d > 1).count();
    if non_singleton > 1 {
        return Err(repelem_error(
            "repelem: when called with a single replication count the input must be a vector",
        ));
    }
    // Scalar / empty input -> default to columns (matches MATLAB row-vector semantics).
    if total <= 1 {
        return Ok(1);
    }
    // Column vector: shape[0] > 1 and others == 1.
    if shape.first().copied().unwrap_or(1) > 1 {
        return Ok(0);
    }
    // Row vector or anything else with a single non-singleton axis -> columns.
    Ok(1)
}

fn repelem_tensor(
    tensor: &Tensor,
    factors: &[RepFactor],
    single_arg: bool,
) -> crate::BuiltinResult<Tensor> {
    let (data, shape) = repelem_column_major(&tensor.data, &tensor.shape, factors, single_arg)?;
    let mut out = Tensor::new_with_dtype(data, shape, tensor.dtype)
        .map_err(|e| repelem_error(format!("repelem: {e}")))?;
    out.dtype = tensor.dtype;
    Ok(out)
}

fn repelem_logical(
    logical: &LogicalArray,
    factors: &[RepFactor],
    single_arg: bool,
) -> crate::BuiltinResult<LogicalArray> {
    let (data, shape) = repelem_column_major(&logical.data, &logical.shape, factors, single_arg)?;
    LogicalArray::new(data, shape).map_err(|e| repelem_error(format!("repelem: {e}")))
}

fn repelem_complex_tensor(
    tensor: &ComplexTensor,
    factors: &[RepFactor],
    single_arg: bool,
) -> crate::BuiltinResult<ComplexTensor> {
    let (data, shape) = repelem_column_major(&tensor.data, &tensor.shape, factors, single_arg)?;
    ComplexTensor::new(data, shape).map_err(|e| repelem_error(format!("repelem: {e}")))
}

fn repelem_string_array(
    sa: &StringArray,
    factors: &[RepFactor],
    single_arg: bool,
) -> crate::BuiltinResult<StringArray> {
    let (data, shape) = repelem_column_major(&sa.data, &sa.shape, factors, single_arg)?;
    StringArray::new(data, shape).map_err(|e| repelem_error(format!("repelem: {e}")))
}

fn repelem_char_array(
    ca: &CharArray,
    factors: &[RepFactor],
    single_arg: bool,
) -> crate::BuiltinResult<CharArray> {
    let (rows, cols, plan) = build_2d_plan(ca.rows, ca.cols, factors, single_arg)?;
    let (data, new_rows, new_cols) =
        repelem_row_major(&ca.data, ca.rows, ca.cols, rows, cols, &plan)?;
    CharArray::new(data, new_rows, new_cols).map_err(|e| repelem_error(format!("repelem: {e}")))
}

fn repelem_cell_array(
    cell: &CellArray,
    factors: &[RepFactor],
    single_arg: bool,
) -> crate::BuiltinResult<CellArray> {
    let (new_rows, new_cols, plan) = build_2d_plan(cell.rows, cell.cols, factors, single_arg)?;
    let total = new_rows
        .checked_mul(new_cols)
        .ok_or_else(|| repelem_error("repelem: requested output exceeds maximum size"))?;
    if total == 0 {
        return CellArray::new(Vec::new(), new_rows, new_cols)
            .map_err(|e| repelem_error(format!("repelem: {e}")));
    }
    let mut values = Vec::with_capacity(total);
    for r in 0..new_rows {
        let src_row = plan.row_table[r];
        for c in 0..new_cols {
            let src_col = plan.col_table[c];
            let idx = src_row * cell.cols + src_col;
            values.push((unsafe { &*cell.data[idx].as_raw() }).clone());
        }
    }
    CellArray::new(values, new_rows, new_cols).map_err(|e| repelem_error(format!("repelem: {e}")))
}

#[derive(Debug)]
struct Plan2D {
    row_table: Vec<usize>,
    col_table: Vec<usize>,
}

fn build_2d_plan(
    rows: usize,
    cols: usize,
    factors: &[RepFactor],
    single_arg: bool,
) -> crate::BuiltinResult<(usize, usize, Plan2D)> {
    if single_arg {
        let shape = [rows, cols];
        let axis = vector_replication_axis(&shape)?;
        let factor = &factors[0];
        let (new_rows, row_table, new_cols, col_table) = if axis == 0 {
            let table = expand_axis(rows, factor, 1)?;
            (table.len(), table, cols, identity_table(cols))
        } else {
            let table = expand_axis(cols, factor, 2)?;
            (rows, identity_table(rows), table.len(), table)
        };
        return Ok((
            new_rows,
            new_cols,
            Plan2D {
                row_table,
                col_table,
            },
        ));
    }

    if factors.len() > 2 {
        // Char / cell arrays only support 2-D replication today. Allow extra
        // dimensions only if every additional factor is a scalar 1 (no-op).
        for (idx, factor) in factors.iter().enumerate().skip(2) {
            match factor {
                RepFactor::Scalar(1) => {}
                _ => {
                    return Err(repelem_error(format!(
                        "repelem: char and cell arrays only support replication along the first two dimensions (extra factor at position {} must be 1)",
                        idx + 1
                    )));
                }
            }
        }
    }
    let row_factor = factors.first().cloned().unwrap_or(RepFactor::Scalar(1));
    let col_factor = factors.get(1).cloned().unwrap_or(RepFactor::Scalar(1));
    let row_table = expand_axis(rows, &row_factor, 1)?;
    let col_table = expand_axis(cols, &col_factor, 2)?;
    Ok((
        row_table.len(),
        col_table.len(),
        Plan2D {
            row_table,
            col_table,
        },
    ))
}

fn identity_table(size: usize) -> Vec<usize> {
    (0..size).collect()
}

fn expand_axis(
    size: usize,
    factor: &RepFactor,
    dim_one_based: usize,
) -> crate::BuiltinResult<Vec<usize>> {
    match factor {
        RepFactor::Scalar(m) => {
            let total = size
                .checked_mul(*m)
                .ok_or_else(|| repelem_error("repelem: requested output exceeds maximum size"))?;
            let mut table = Vec::with_capacity(total);
            for i in 0..size {
                for _ in 0..*m {
                    table.push(i);
                }
            }
            Ok(table)
        }
        RepFactor::Vector(v) => {
            if v.len() != size {
                return Err(repelem_error(format!(
                    "repelem: replication vector at dimension {dim_one_based} has length {} but the input dimension has size {}",
                    v.len(),
                    size
                )));
            }
            let total = v
                .iter()
                .try_fold(0usize, |acc, &x| acc.checked_add(x))
                .ok_or_else(|| repelem_error("repelem: requested output exceeds maximum size"))?;
            let mut table = Vec::with_capacity(total);
            for (i, &count) in v.iter().enumerate() {
                for _ in 0..count {
                    table.push(i);
                }
            }
            Ok(table)
        }
    }
}

fn repelem_column_major<T: Clone>(
    data: &[T],
    shape: &[usize],
    factors: &[RepFactor],
    single_arg: bool,
) -> crate::BuiltinResult<(Vec<T>, Vec<usize>)> {
    // Pad the input shape to at least 2-D, matching MATLAB conventions where
    // every array has rank >= 2.
    let mut input_shape = if shape.is_empty() {
        vec![1usize, 1]
    } else {
        shape.to_vec()
    };
    while input_shape.len() < 2 {
        input_shape.push(1);
    }

    // Validate input vs data length (column-major flat storage).
    let orig_total = checked_total(&input_shape)?;
    if !(orig_total == data.len() || (orig_total == 0 && data.is_empty())) {
        return Err(repelem_error(format!(
            "repelem: internal shape mismatch (expected {orig_total} elements, found {})",
            data.len()
        )));
    }

    // Build the per-axis replication plan.
    let mut axis_factors: Vec<RepFactor> = Vec::new();
    let rank;
    if single_arg {
        let axis = vector_replication_axis(&input_shape)?;
        rank = input_shape.len();
        for k in 0..rank {
            if k == axis {
                axis_factors.push(factors[0].clone());
            } else {
                axis_factors.push(RepFactor::Scalar(1));
            }
        }
    } else {
        rank = input_shape.len().max(factors.len()).max(2);
        while input_shape.len() < rank {
            input_shape.push(1);
        }
        for k in 0..rank {
            axis_factors.push(factors.get(k).cloned().unwrap_or(RepFactor::Scalar(1)));
        }
    }

    let mut idx_tables: Vec<Vec<usize>> = Vec::with_capacity(rank);
    let mut output_shape = Vec::with_capacity(rank);
    for (k, factor) in axis_factors.iter().enumerate() {
        let dim_size = input_shape[k];
        let table = expand_axis(dim_size, factor, k + 1)?;
        output_shape.push(table.len());
        idx_tables.push(table);
    }

    let new_total = checked_total(&output_shape)?;
    if new_total == 0 {
        return Ok((Vec::new(), output_shape));
    }

    let src_strides = column_major_strides(&input_shape);
    let mut out = Vec::with_capacity(new_total);
    for idx in 0..new_total {
        let mut rem = idx;
        let mut src_index = 0usize;
        for k in 0..rank {
            let dim_size = output_shape[k];
            let coord = rem % dim_size;
            rem /= dim_size;
            let src_coord = idx_tables[k][coord];
            src_index += src_coord * src_strides[k];
        }
        out.push(data[src_index].clone());
    }
    Ok((out, output_shape))
}

fn repelem_row_major<T: Clone>(
    data: &[T],
    rows: usize,
    cols: usize,
    new_rows: usize,
    new_cols: usize,
    plan: &Plan2D,
) -> crate::BuiltinResult<(Vec<T>, usize, usize)> {
    if rows.checked_mul(cols).unwrap_or(0) != data.len() && !(rows == 0 || cols == 0) {
        return Err(repelem_error(
            "repelem: internal shape mismatch for row-major array",
        ));
    }
    let total = new_rows
        .checked_mul(new_cols)
        .ok_or_else(|| repelem_error("repelem: requested output exceeds maximum size"))?;
    if total == 0 {
        return Ok((Vec::new(), new_rows, new_cols));
    }
    let mut out = Vec::with_capacity(total);
    for r in 0..new_rows {
        let src_row = plan.row_table[r];
        for c in 0..new_cols {
            let src_col = plan.col_table[c];
            let idx = src_row * cols + src_col;
            out.push(data[idx].clone());
        }
    }
    Ok((out, new_rows, new_cols))
}

fn column_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1usize;
    for &dim in dims {
        strides.push(stride);
        stride = stride.saturating_mul(dim.max(1));
    }
    strides
}

fn checked_total(shape: &[usize]) -> crate::BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| repelem_error("repelem: requested output exceeds maximum size"))
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, NumericDType};

    fn repelem_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::repelem_builtin(value, rest))
    }

    #[test]
    fn repelem_type_resolves_tensor_rank() {
        let out = repelem_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Num,
                Type::Num,
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![None, None])
            }
        );
    }

    #[test]
    fn repelem_type_preserves_logical_kind() {
        let out = repelem_type(
            &[
                Type::Logical {
                    shape: Some(vec![Some(2), Some(2)]),
                },
                Type::Num,
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Logical {
                shape: Some(vec![None, None])
            }
        );
    }

    #[test]
    fn row_vector_scalar_replication() {
        let v = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result =
            repelem_builtin(Value::Tensor(v), vec![Value::Int(IntValue::I32(2))]).expect("repelem");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 6]);
                assert_eq!(t.data, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn row_vector_per_element_replication() {
        let v = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let counts = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result =
            repelem_builtin(Value::Tensor(v), vec![Value::Tensor(counts)]).expect("repelem");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 6]);
                assert_eq!(t.data, vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn column_vector_scalar_replication_preserves_orientation() {
        let v = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result =
            repelem_builtin(Value::Tensor(v), vec![Value::Int(IntValue::I32(2))]).expect("repelem");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![6, 1]);
                assert_eq!(t.data, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn column_vector_per_element_replication() {
        let v = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let counts = Tensor::new(vec![3.0, 0.0, 2.0], vec![3, 1]).unwrap();
        let result =
            repelem_builtin(Value::Tensor(v), vec![Value::Tensor(counts)]).expect("repelem");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![5, 1]);
                assert_eq!(t.data, vec![1.0, 1.0, 1.0, 3.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn matrix_scalar_replication_creates_blocks() {
        // magic(3) in MATLAB column-major:
        //   8 1 6
        //   3 5 7
        //   4 9 2
        // Stored as [8,3,4, 1,5,9, 6,7,2] column-major.
        let magic = Tensor::new(
            vec![8.0, 3.0, 4.0, 1.0, 5.0, 9.0, 6.0, 7.0, 2.0],
            vec![3, 3],
        )
        .unwrap();
        let result = repelem_builtin(
            Value::Tensor(magic.clone()),
            vec![Value::Int(IntValue::I32(2)), Value::Int(IntValue::I32(3))],
        )
        .expect("repelem");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![6, 9]);
                let rows = 6usize;
                for c in 0..9 {
                    for r in 0..rows {
                        let src_r = r / 2;
                        let src_c = c / 3;
                        let src_idx = src_r + src_c * 3;
                        let dst_idx = r + c * rows;
                        assert_eq!(
                            t.data[dst_idx], magic.data[src_idx],
                            "mismatch at (r={r}, c={c})"
                        );
                    }
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn matrix_per_axis_vector_replication() {
        // magic(3) again.
        let magic = Tensor::new(
            vec![8.0, 3.0, 4.0, 1.0, 5.0, 9.0, 6.0, 7.0, 2.0],
            vec![3, 3],
        )
        .unwrap();
        let row_counts = Tensor::new(vec![1.0, 2.0, 1.0], vec![1, 3]).unwrap();
        let col_counts = Tensor::new(vec![2.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let result = repelem_builtin(
            Value::Tensor(magic.clone()),
            vec![Value::Tensor(row_counts), Value::Tensor(col_counts)],
        )
        .expect("repelem");
        match result {
            Value::Tensor(t) => {
                // Sum of row counts -> 4 rows. Sum of col counts -> 5 cols.
                assert_eq!(t.shape, vec![4, 5]);

                let row_table = [0usize, 1, 1, 2];
                let col_table = [0usize, 0, 1, 2, 2];
                for c in 0..5 {
                    for r in 0..4 {
                        let src_r = row_table[r];
                        let src_c = col_table[c];
                        let src_idx = src_r + src_c * 3;
                        let dst_idx = r + c * 4;
                        assert_eq!(
                            t.data[dst_idx], magic.data[src_idx],
                            "mismatch at (r={r}, c={c})"
                        );
                    }
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn vector_length_mismatch_errors() {
        let v = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let bad = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let err =
            repelem_builtin(Value::Tensor(v), vec![Value::Tensor(bad)]).expect_err("expected err");
        let msg = err.to_string();
        assert!(
            msg.contains("length 2") && msg.contains("size 3"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn rejects_negative_replication() {
        let v = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let err = repelem_builtin(Value::Tensor(v), vec![Value::Int(IntValue::I32(-1))])
            .expect_err("expected err");
        assert!(err.to_string().contains("non-negative"));
    }

    #[test]
    fn rejects_fractional_replication() {
        let v = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let err =
            repelem_builtin(Value::Tensor(v), vec![Value::Num(1.5)]).expect_err("expected err");
        assert!(err.to_string().contains("integer"));
    }

    #[test]
    fn rejects_single_arg_for_matrix() {
        let m = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let err = repelem_builtin(Value::Tensor(m), vec![Value::Int(IntValue::I32(2))])
            .expect_err("expected err");
        assert!(err.to_string().contains("vector"));
    }

    #[test]
    fn scalar_input_extends_to_high_dim() {
        // MATLAB's `repelem(5, 2, 2, 2)` returns a 2x2x2 array of 5s.
        let result = repelem_builtin(
            Value::Num(5.0),
            vec![
                Value::Int(IntValue::I32(2)),
                Value::Int(IntValue::I32(2)),
                Value::Int(IntValue::I32(2)),
            ],
        )
        .expect("repelem");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2, 2]);
                assert_eq!(t.data, vec![5.0; 8]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn zero_count_vector_yields_empty_axis() {
        let v = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let counts = Tensor::new(vec![0.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let result =
            repelem_builtin(Value::Tensor(v), vec![Value::Tensor(counts)]).expect("repelem");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn column_vector_zero_count_yields_empty_column() {
        let v = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let counts = Tensor::new(vec![0.0, 0.0, 0.0], vec![3, 1]).unwrap();
        let result =
            repelem_builtin(Value::Tensor(v), vec![Value::Tensor(counts)]).expect("repelem");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 1]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn per_element_vector_with_zero_entries_skips_elements() {
        // Documented MATLAB example: repelem([1 2 3 4 5], [0 1 0 2 1])
        // yields a 1x4 row vector [2 4 4 5].
        let v = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
        let counts = Tensor::new(vec![0.0, 1.0, 0.0, 2.0, 1.0], vec![1, 5]).unwrap();
        let result =
            repelem_builtin(Value::Tensor(v), vec![Value::Tensor(counts)]).expect("repelem");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                assert_eq!(t.data, vec![2.0, 4.0, 4.0, 5.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn matrix_mixed_scalar_and_vector_factors() {
        // 2x3 input replicated as row=Scalar(2), col=Vector([1, 2, 0])
        // should produce a 4x3 matrix dropping the third source column entirely.
        let m = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let col_counts = Tensor::new(vec![1.0, 2.0, 0.0], vec![1, 3]).unwrap();
        let result = repelem_builtin(
            Value::Tensor(m),
            vec![Value::Int(IntValue::I32(2)), Value::Tensor(col_counts)],
        )
        .expect("repelem");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4, 3]);
                // Column 0 of source is [1, 4] repeated row-wise twice -> [1, 1, 4, 4].
                // Column 1 of source is [2, 5] used twice across output cols 1..3.
                let rows = 4usize;
                let expected_cols = [
                    [1.0, 1.0, 4.0, 4.0],
                    [2.0, 2.0, 5.0, 5.0],
                    [2.0, 2.0, 5.0, 5.0],
                ];
                for (c, expected) in expected_cols.iter().enumerate() {
                    for (r, value) in expected.iter().enumerate() {
                        let dst_idx = r + c * rows;
                        assert_eq!(t.data[dst_idx], *value, "mismatch at (r={r}, c={c})");
                    }
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn integer_dtype_tensor_preserves_dtype() {
        let mut tensor =
            Tensor::new_with_dtype(vec![1.0, 2.0, 3.0], vec![1, 3], NumericDType::U8).unwrap();
        tensor.dtype = NumericDType::U8;
        let result = repelem_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(2))])
            .expect("repelem");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 6]);
                assert_eq!(t.data, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
                assert_eq!(t.dtype, NumericDType::U8);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn logical_array_replication() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let result = repelem_builtin(
            Value::LogicalArray(logical),
            vec![Value::Int(IntValue::I32(2))],
        )
        .expect("repelem");
        match result {
            Value::LogicalArray(la) => {
                assert_eq!(la.shape, vec![1, 6]);
                assert_eq!(la.data, vec![1, 1, 0, 0, 1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn cell_array_replication_2d() {
        let cell = CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let result = repelem_builtin(
            Value::Cell(cell),
            vec![Value::Int(IntValue::I32(2)), Value::Int(IntValue::I32(2))],
        )
        .expect("repelem");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 4);
                let expected_cols = [1.0, 1.0, 2.0, 2.0];
                for r in 0..out.rows {
                    for c in 0..out.cols {
                        match out.get(r, c).expect("cell element") {
                            Value::Num(n) => assert_eq!(n, expected_cols[c]),
                            other => panic!("expected numeric cell element, got {other:?}"),
                        }
                    }
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn char_array_replication_2d() {
        let ca = CharArray::new("hi".chars().collect(), 1, 2).unwrap();
        let result = repelem_builtin(
            Value::CharArray(ca),
            vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(3))],
        )
        .expect("repelem");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 6);
                let s: String = out.data.iter().collect();
                assert_eq!(s, "hhhiii");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn complex_tensor_replication() {
        let ct = ComplexTensor::new(vec![(1.0, -1.0), (0.0, 2.0)], vec![1, 2]).unwrap();
        let result = repelem_builtin(
            Value::ComplexTensor(ct),
            vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(2))],
        )
        .expect("repelem");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 4]);
                assert_eq!(
                    out.data,
                    vec![(1.0, -1.0), (1.0, -1.0), (0.0, 2.0), (0.0, 2.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn string_array_replication() {
        let sa = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
        let result = repelem_builtin(
            Value::StringArray(sa),
            vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(2))],
        )
        .expect("repelem");
        match result {
            Value::StringArray(out) => {
                assert_eq!(out.shape, vec![1, 4]);
                assert_eq!(
                    out.data,
                    vec![
                        "a".to_string(),
                        "a".to_string(),
                        "b".to_string(),
                        "b".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }
}
