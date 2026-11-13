//! One pre-query resource admission for the typed two-dimensional spline.

use super::*;

pub(in crate::builtins::math::interpolation::gridded_interpolant) fn admit_spline2_from_object(
    receiver: &Value,
    args: crate::call::FunctionArgumentView<'_>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Option<HostAllocationReceipt>> {
    let Some(runtime) = runtime else {
        return Ok(None);
    };
    let Some(parallel) = runtime.service_ports().parallel() else {
        return Ok(None);
    };
    if parallel
        .placement_resources()
        .memory_available_bytes
        .is_none()
    {
        return Ok(None);
    }
    let Value::Object(object) = receiver else {
        return Ok(None);
    };
    if !object.is_class(CLASS_IDENTITY) {
        return Ok(None);
    }
    let Some(method) = object
        .properties()
        .get(METHOD)
        .map(text_from_value)
        .transpose()?
        .flatten()
    else {
        return Ok(None);
    };
    let Some(method) = MethodKeyword::parse(&method) else {
        return Ok(None);
    };
    if parse_interp_method(method)? != InterpMethod::Spline {
        return Ok(None);
    }
    let Some(Value::Cell(grids)) = object.properties().get(GRID_VECTORS) else {
        return Ok(None);
    };
    if grids.data().len() != 2 {
        return Ok(None);
    }
    let Some(Value::Tensor(values)) = object.properties().get(VALUES) else {
        return Ok(None);
    };
    let rows = raw_count(&grids.data()[0])
        .map_err(|error| super::super::super::pp::host_allocation_error(BUILTIN_NAME, error))?;
    let columns = raw_count(&grids.data()[1])
        .map_err(|error| super::super::super::pp::host_allocation_error(BUILTIN_NAME, error))?;
    let mut extra_shape = values.shape().get(2..).unwrap_or(&[]).to_vec();
    while extra_shape.last() == Some(&1) {
        extra_shape.pop();
    }
    let extra_count = extra_shape
        .iter()
        .try_fold(1usize, |n, dim| n.checked_mul(*dim))
        .ok_or_else(|| invalid("value size exceeds platform limits"))?;
    let axis_copy = rows
        .checked_add(columns)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| invalid("grid vector copy size exceeds platform limits"))?;
    let axis_copy = u64::try_from(axis_copy)
        .map_err(|_| invalid("grid vector copy size exceeds platform limits"))?;
    let source_dtype = values.numeric_dtype();
    let dtype = if matches!(source_dtype, NumericDType::F32 | NumericDType::F64) {
        source_dtype
    } else {
        NumericDType::F64
    };
    let integer_projection = if dtype != source_dtype {
        u64::try_from(values.len())
            .map_err(|_| invalid("value size overflow"))?
            .checked_mul(8)
            .ok_or_else(|| invalid("value size overflow"))?
    } else {
        0
    };
    let object_scratch = axis_copy
        .checked_add(integer_projection)
        .ok_or_else(|| invalid("object workspace size overflow"))?;
    admit_geometry(
        rows,
        columns,
        values.len(),
        dtype,
        extra_count,
        extra_shape.len(),
        args,
        object_scratch,
        Some(runtime),
    )
}

pub(super) fn admit_spline2_before_query(
    spec: &InterpolantSpec,
    args: crate::call::FunctionArgumentView<'_>,
    grid_size: usize,
    extra_count: usize,
    extra_rank: usize,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Option<HostAllocationReceipt>> {
    if spec.method != InterpMethod::Spline || spec.grid_vectors.len() != 2 {
        return Ok(None);
    }
    debug_assert_eq!(
        grid_size,
        spec.grid_vectors[0].len() * spec.grid_vectors[1].len()
    );
    admit_geometry(
        spec.grid_vectors[0].len(),
        spec.grid_vectors[1].len(),
        spec.values.len(),
        spec.values.numeric_dtype(),
        extra_count,
        extra_rank,
        args,
        0,
        runtime,
    )
}

fn admit_geometry(
    rows: usize,
    columns: usize,
    value_len: usize,
    dtype: NumericDType,
    extra_count: usize,
    extra_rank: usize,
    args: crate::call::FunctionArgumentView<'_>,
    object_scratch: u64,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Option<HostAllocationReceipt>> {
    let Some(runtime) = runtime else {
        return Ok(None);
    };
    let Some(parallel) = runtime.service_ports().parallel() else {
        return Ok(None);
    };
    if parallel
        .placement_resources()
        .memory_available_bytes
        .is_none()
    {
        return Ok(None);
    }
    let Some(geometry) = query_geometry(args) else {
        // The ordinary query parser owns malformed-form diagnostics. Every
        // supported two-dimensional query form has a geometry above. Reject
        // malformed resident forms before their otherwise unnecessary gather.
        if args.iter().any(crate::dispatcher::value_contains_gpu) {
            return Err(invalid("unsupported two-dimensional spline query form"));
        }
        return Ok(None);
    };
    let (query_count, query_rank, query_bytes) = geometry
        .map_err(|error| super::super::super::pp::host_allocation_error(BUILTIN_NAME, error))?;
    let output_count = query_count
        .checked_mul(extra_count)
        .ok_or_else(|| invalid("interpolant output size exceeds platform limits"))?;
    let output_rank = query_rank
        .checked_add(extra_rank)
        .ok_or_else(|| invalid("interpolant output rank exceeds platform limits"))?;
    let value_projection = if dtype == NumericDType::F32 {
        u64::try_from(value_len)
            .map_err(|_| invalid("value size overflow"))?
            .checked_mul(8)
            .ok_or_else(|| invalid("value size overflow"))?
    } else {
        0
    };
    let input_scratch = query_bytes
        .checked_add(value_projection)
        .and_then(|bytes| bytes.checked_add(object_scratch))
        .ok_or_else(|| invalid("spline workspace size overflow"))?;
    let plan = tensor_spline_allocation_plan(
        rows,
        columns,
        1,
        output_count,
        output_rank,
        dtype,
        input_scratch,
    )
    .map_err(|error| super::super::super::pp::host_allocation_error(BUILTIN_NAME, error))?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|error| super::super::super::pp::host_allocation_error(BUILTIN_NAME, error))?;
    receipt.map(Some).ok_or_else(|| {
        super::super::super::pp::host_allocation_error(
            BUILTIN_NAME,
            HostAllocationError::StaleAuthority,
        )
    })
}

fn query_geometry(
    args: crate::call::FunctionArgumentView<'_>,
) -> Option<Result<(usize, usize, u64), HostAllocationError>> {
    let operands = (args.len(), args.get(0), args.get(1));
    let supported = matches!(operands, (1, Some(Value::Cell(cell)), _) if cell.data().len() == 2)
        || matches!(operands, (1, Some(matrix), _) if raw_shape(matrix).len() == 2 && raw_shape(matrix)[1] == 2)
        || matches!(operands, (2, Some(_), Some(_)));
    if !supported {
        return None;
    }
    let checked = |n: usize| u64::try_from(n).map_err(|_| HostAllocationError::Overflow);
    Some((|| match operands {
        (1, Some(Value::Cell(cell)), _) if cell.data().len() == 2 => {
            let x = raw_count(&cell.data()[0])?;
            let y = raw_count(&cell.data()[1])?;
            let total = x.checked_mul(y).ok_or(HostAllocationError::Overflow)?;
            let vectors = x.checked_add(y).ok_or(HostAllocationError::Overflow)?;
            let bytes = checked(vectors)?
                .checked_mul(8)
                .ok_or(HostAllocationError::Overflow)?;
            let downloads = resident_bytes(&cell.data()[0])?
                .checked_add(resident_bytes(&cell.data()[1])?)
                .ok_or(HostAllocationError::Overflow)?;
            Ok((
                total,
                2,
                bytes
                    .checked_add(downloads)
                    .ok_or(HostAllocationError::Overflow)?,
            ))
        }
        (1, Some(matrix), _) if raw_shape(matrix).len() == 2 && raw_shape(matrix)[1] == 2 => {
            let count = raw_shape(matrix)[0];
            let bytes = checked(count)?
                .checked_mul(56)
                .ok_or(HostAllocationError::Overflow)?;
            let download = resident_bytes(matrix)?;
            Ok((
                count,
                2,
                bytes
                    .checked_add(download)
                    .ok_or(HostAllocationError::Overflow)?,
            ))
        }
        (2, Some(left), Some(right)) => {
            let count = raw_count(left)?;
            let other = raw_count(right)?;
            let temporary = count
                .checked_add(other)
                .and_then(|n| n.checked_mul(8))
                .ok_or(HostAllocationError::Overflow)?;
            let points = count.checked_mul(40).ok_or(HostAllocationError::Overflow)?;
            let bytes = temporary
                .checked_add(points)
                .ok_or(HostAllocationError::Overflow)?;
            let downloads = resident_bytes(left)?
                .checked_add(resident_bytes(right)?)
                .ok_or(HostAllocationError::Overflow)?;
            Ok((
                count,
                raw_shape(left).len().max(2),
                checked(bytes)?
                    .checked_add(downloads)
                    .ok_or(HostAllocationError::Overflow)?,
            ))
        }
        _ => unreachable!("supported query form"),
    })())
}

fn raw_shape(value: &Value) -> &[usize] {
    match value {
        Value::Tensor(tensor) => tensor.shape(),
        Value::GpuTensor(handle) => handle.shape(),
        _ => &[],
    }
}

fn resident_bytes(value: &Value) -> Result<u64, HostAllocationError> {
    let Value::GpuTensor(handle) = value else {
        return Ok(0);
    };
    let count = u64::try_from(raw_count(value)?).map_err(|_| HostAllocationError::Overflow)?;
    let lanes = if handle.physical().storage()
        == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    {
        2
    } else {
        1
    };
    count
        .checked_mul(handle.physical().element_type().element_size() as u64)
        .and_then(|bytes| bytes.checked_mul(lanes))
        .ok_or(HostAllocationError::Overflow)
}

fn raw_count(value: &Value) -> Result<usize, HostAllocationError> {
    raw_shape(value)
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or(HostAllocationError::Overflow)
}
