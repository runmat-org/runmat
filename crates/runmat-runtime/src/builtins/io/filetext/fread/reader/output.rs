//! One final exact output, with the existing fread padding and layout rules.

use super::*;
use crate::builtins::io::filetext::cooperation::FileCooperation;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use presentation::{Destination, Projection};
use runmat_value::{
    ExactArrayStorage, ExactHostLogicalTensorAllocation, ExactHostTensorAllocation, SharedShape,
};
use scalar::ReadValue;
use workspace::{allocation, StreamValues};

const OUTPUT_FILL_CHUNK: usize = 4096;

fn geometry(
    runtime: Option<&crate::context::RuntimeContext>,
    size: &SizeSpec,
    count: usize,
) -> BuiltinResult<(usize, usize, usize)> {
    let allocation = |source| allocation(runtime, source);
    let (rows, columns) = match size {
        SizeSpec::All | SizeSpec::Count(_) => (count, 1),
        SizeSpec::Matrix {
            rows,
            cols: Some(columns),
        } => (*rows, *columns),
        SizeSpec::Matrix { rows, cols: None } => {
            (*rows, if *rows == 0 { 0 } else { count.div_ceil(*rows) })
        }
    };
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| allocation(HostAllocationError::Overflow))?;
    Ok((rows, columns, count))
}

fn character_plan(
    runtime: Option<&crate::context::RuntimeContext>,
    count: usize,
) -> BuiltinResult<u64> {
    let allocation = |source| allocation(runtime, source);
    let payload = ExactArrayStorage::<runmat_value::Utf16CodeUnit>::requested_bytes(count)
        .map_err(|source| allocation(HostAllocationError::Numeric(source)))?;
    let dimensions = SharedShape::requested_bytes_for_rank(2)
        .map_err(|source| allocation(HostAllocationError::Numeric(source)))?;
    let physical = payload
        .checked_add(dimensions)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| allocation(HostAllocationError::Overflow))?;
    HostAllocationPlan::retained_exact_output_bytes(physical, usize::from(count != 0) + 1)
        .map_err(allocation)
}

fn output_plan(
    runtime: Option<&crate::context::RuntimeContext>,
    destination: Destination,
    shape: &[usize; 2],
) -> BuiltinResult<u64> {
    let allocation = |source| allocation(runtime, source);
    match destination {
        Destination::Character => character_plan(
            runtime,
            shape[0]
                .checked_mul(shape[1])
                .ok_or_else(|| allocation(HostAllocationError::Overflow))?,
        ),
        Destination::Numeric(dtype) => {
            HostAllocationPlan::retained_tensor_bytes(dtype, shape).map_err(allocation)
        }
        Destination::Logical => {
            HostAllocationPlan::retained_logical_tensor_bytes(shape).map_err(allocation)
        }
    }
}

async fn checkpoint<T: Copy>(
    cooperation: &mut FileCooperation,
    values: &StreamValues<'_, T>,
    receipt: Option<&HostAllocationReceipt>,
    completed: usize,
) -> BuiltinResult<()> {
    let runtime = values.runtime();
    let allocation = |source| allocation(runtime, source);
    if completed != 0 {
        cooperation.completed(completed).await?;
    }
    values.check()?;
    if let Some(receipt) = receipt {
        receipt
            .checkpoint(values.runtime().expect("managed output has runtime"))
            .map_err(allocation)?;
    }
    Ok(())
}

pub(super) async fn finish<T: ReadValue>(
    values: &StreamValues<'_, T>,
    size: &SizeSpec,
    projection: Projection<'_>,
    cooperation: &mut FileCooperation,
) -> BuiltinResult<Value> {
    let runtime = values.runtime();
    let allocation = |source| allocation(runtime, source);
    let (rows, columns, _) = geometry(runtime, size, values.values().len())?;
    let shape = [rows, columns];
    values.check()?;
    let receipt = HostAllocationReceipt::admit(
        runtime,
        HostAllocationPlan::new(0, output_plan(runtime, projection.destination, &shape)?)
            .map_err(allocation)?,
    )
    .map_err(allocation)?;
    match projection.destination {
        Destination::Character => {
            finish_character(values, shape, projection, receipt, cooperation).await
        }
        Destination::Numeric(dtype) => {
            finish_numeric(values, shape, projection, dtype, receipt, cooperation).await
        }
        Destination::Logical => {
            finish_logical(values, shape, projection, receipt, cooperation).await
        }
    }
}

async fn finish_character<T: ReadValue>(
    values: &StreamValues<'_, T>,
    shape: [usize; 2],
    projection: Projection<'_>,
    receipt: Option<HostAllocationReceipt>,
    cooperation: &mut FileCooperation,
) -> BuiltinResult<Value> {
    let runtime = values.runtime();
    let allocation = |source| allocation(runtime, source);
    let [rows, columns] = shape;
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| allocation(HostAllocationError::Overflow))?;
    let mut managed = if let Some(receipt) = &receipt {
        Some(
            receipt
                .with_output_allocation(
                    values.runtime().expect("managed output has runtime"),
                    |permit| ExactArrayStorage::try_begin(count, permit),
                )
                .map_err(allocation)?,
        )
    } else {
        None
    };
    let mut ordinary = Vec::new();
    if managed.is_none() {
        ordinary.try_reserve_exact(count).map_err(|source| {
            build_runtime_error("fread: character output allocation failed")
                .with_builtin(BUILTIN_NAME)
                .with_source(source)
                .build()
        })?;
    }
    let mut pending = 0usize;
    for index in 0..count {
        let source = if columns == 0 {
            0
        } else {
            (index % columns) * rows + index / columns
        };
        let value = values
            .values()
            .get(source)
            .copied()
            .map(|value| projection.character(value))
            .transpose()
            .map_err(ReadFailure::Semantic)?
            .unwrap_or(runmat_value::Utf16CodeUnit::from_raw(0));
        if let Some(managed) = &mut managed {
            managed.try_push(value).map_err(|_| {
                ReadFailure::Semantic(
                    "fread: character output exceeds admitted geometry".to_string(),
                )
            })?;
        } else {
            ordinary.push(value);
        }
        pending += 1;
        if pending == OUTPUT_FILL_CHUNK {
            checkpoint(cooperation, values, receipt.as_ref(), pending).await?;
            pending = 0;
        }
    }
    checkpoint(cooperation, values, receipt.as_ref(), pending).await?;
    let array = if let Some(managed) = managed {
        let admitted_shape = receipt
            .as_ref()
            .expect("exact output has receipt")
            .with_output_allocation(
                values.runtime().expect("managed output has runtime"),
                |permit| SharedShape::new_admitted_from_slice(&shape, permit),
            )
            .map_err(allocation)?;
        CharArray::from_admitted(managed.finish_initialized(), admitted_shape)
    } else {
        CharArray::from_code_units(ordinary, rows, columns)
    };
    array
        .map(Value::CharArray)
        .map_err(|detail| ReadFailure::Semantic(format!("fread: {detail}")).into())
}

async fn finish_numeric<T: ReadValue>(
    values: &StreamValues<'_, T>,
    shape: [usize; 2],
    projection: Projection<'_>,
    dtype: NumericDType,
    receipt: Option<HostAllocationReceipt>,
    cooperation: &mut FileCooperation,
) -> BuiltinResult<Value> {
    let runtime = values.runtime();
    let allocation = |source| allocation(runtime, source);
    let count = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| allocation(HostAllocationError::Overflow))?;
    let mut managed: Option<ExactHostTensorAllocation> = if let Some(receipt) = &receipt {
        Some(
            receipt
                .exact_output_tensor(
                    values.runtime().expect("managed output has runtime"),
                    dtype,
                    &shape,
                )
                .map_err(allocation)?,
        )
    } else {
        None
    };
    let mut ordinary = if managed.is_none() {
        Some(
            NumericStorage::zeros_exact(dtype, count)
                .map_err(|source| allocation(HostAllocationError::Numeric(source)))?,
        )
    } else {
        None
    };
    let mut pending = 0usize;
    for index in 0..count {
        let scalar = values
            .values()
            .get(index)
            .copied()
            .map(|value| projection.numeric(value, dtype))
            .unwrap_or_else(|| projection.numeric(0_u8, dtype));
        if let Some(managed) = &mut managed {
            managed
                .set_value(index, scalar)
                .map_err(ReadFailure::Semantic)?;
        } else {
            ordinary
                .as_mut()
                .expect("ordinary output")
                .set_value(index, scalar)
                .map_err(ReadFailure::Semantic)?;
        }
        pending += 1;
        if pending == OUTPUT_FILL_CHUNK {
            checkpoint(cooperation, values, receipt.as_ref(), pending).await?;
            pending = 0;
        }
    }
    checkpoint(cooperation, values, receipt.as_ref(), pending).await?;
    let tensor = if let Some(managed) = managed {
        managed
            .into_tensor()
            .map_err(|source| allocation(HostAllocationError::Numeric(source)))?
    } else {
        Tensor::from_numeric_storage(ordinary.expect("ordinary output"), shape.to_vec())
            .map_err(|detail| ReadFailure::Semantic(format!("fread: {detail}")))?
    };
    Ok(Value::Tensor(tensor))
}

async fn finish_logical<T: ReadValue>(
    values: &StreamValues<'_, T>,
    shape: [usize; 2],
    projection: Projection<'_>,
    receipt: Option<HostAllocationReceipt>,
    cooperation: &mut FileCooperation,
) -> BuiltinResult<Value> {
    let runtime = values.runtime();
    let allocation = |source| allocation(runtime, source);
    let count = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| allocation(HostAllocationError::Overflow))?;
    let mut managed: Option<ExactHostLogicalTensorAllocation> = if let Some(receipt) = &receipt {
        Some(
            receipt
                .with_output_allocation(
                    values.runtime().expect("managed output has runtime"),
                    |permit| ExactHostLogicalTensorAllocation::new(&shape, permit),
                )
                .map_err(allocation)?,
        )
    } else {
        None
    };
    let mut ordinary = if managed.is_none() {
        let mut bits = Vec::new();
        bits.try_reserve_exact(count).map_err(|source| {
            build_runtime_error("fread: logical output allocation failed")
                .with_builtin(BUILTIN_NAME)
                .with_source(source)
                .build()
        })?;
        bits.resize(count, 0);
        Some(bits)
    } else {
        None
    };
    let mut pending = 0usize;
    for index in 0..count {
        let value = values
            .values()
            .get(index)
            .copied()
            .map(|value| projection.logical(value))
            .transpose()
            .map_err(ReadFailure::Semantic)?
            .unwrap_or(0);
        if let Some(managed) = &mut managed {
            managed.as_mut_slice()[index] = value;
        } else {
            ordinary.as_mut().expect("ordinary output")[index] = value;
        }
        pending += 1;
        if pending == OUTPUT_FILL_CHUNK {
            checkpoint(cooperation, values, receipt.as_ref(), pending).await?;
            pending = 0;
        }
    }
    checkpoint(cooperation, values, receipt.as_ref(), pending).await?;
    let array = if let Some(managed) = managed {
        managed
            .into_array()
            .map_err(|source| allocation(HostAllocationError::Numeric(source)))?
    } else {
        LogicalArray::new(ordinary.expect("ordinary output"), shape.to_vec())
            .map_err(|detail| ReadFailure::Semantic(format!("fread: {detail}")))?
    };
    Ok(Value::LogicalArray(array))
}
