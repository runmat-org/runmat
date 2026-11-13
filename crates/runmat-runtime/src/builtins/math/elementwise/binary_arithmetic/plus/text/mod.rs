//! String addition converts borrowed elements into one admitted final result.
mod operand;
mod output;
mod error;
use error::TextError;
#[cfg(test)]
mod tests;

use operand::Operand;
use runmat_value::{ExactArrayStorage, SharedShape, StringArray, StringElement, Utf8Text, Value};

use crate::builtins::common::broadcast::{BroadcastPlan, PreparedBroadcastPlan};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};

fn invalid_input(message: impl AsRef<str>) -> RuntimeError {
    super::plus_error_with_detail(&runmat_builtins::PLUS_ERROR_INVALID_INPUT, message)
}

fn text_error(error: TextError) -> RuntimeError {
    let mut builder = crate::build_runtime_error(format!("plus: {error}"))
        .with_builtin(super::BUILTIN_NAME)
        .with_source(error);
    if let Some(identifier) = runmat_builtins::PLUS_ERROR_INVALID_INPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn admission(runtime: &RuntimeContext, error: HostAllocationError) -> RuntimeError {
    crate::builtins::strings::common::text_input_admission("plus", runtime, error)
}

fn numeric(error: runmat_value::ExactHostNumericAllocationError) -> HostAllocationError {
    HostAllocationError::Numeric(error)
}

fn retained(physical: usize, claims: usize) -> Result<u64, HostAllocationError> {
    HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical).map_err(|_| HostAllocationError::Overflow)?,
        claims,
    )
}

pub(super) async fn concatenate(
    runtime: Option<&RuntimeContext>,
    left: &Value,
    right: &Value,
) -> BuiltinResult<Value> {
    let left = Operand::new(left)?;
    let right = Operand::new(right)?;
    let dimensions =
        runmat_types::geometry::ConcreteBroadcastDimensions::new(left.shape(), right.shape())
            .map_err(|error| {
                super::plus_error_with_detail(
                    &runmat_builtins::PLUS_ERROR_SIZE_MISMATCH,
                    error.to_string(),
                )
            })?;
    let count = dimensions
        .element_count()
        .ok_or_else(|| invalid_input("string array cardinality overflow"))?;
    let Some(runtime) = runtime else {
        return ordinary(None, &left, &right).await;
    };
    let shape_bytes = SharedShape::requested_bytes_for_rank(dimensions.rank())
        .map_err(|error| admission(runtime, numeric(error)))?;
    let plan = HostAllocationPlan::new(
        PreparedBroadcastPlan::quoted_bytes(dimensions.rank())
            .map_err(|error| admission(runtime, error))?,
        retained(shape_bytes, 1).map_err(|error| admission(runtime, error))?,
    )
    .map_err(|error| admission(runtime, error))?;
    let Some(metadata) = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|error| admission(runtime, error))?
    else {
        return ordinary(Some(runtime), &left, &right).await;
    };
    let shape = metadata
        .with_output_allocation(runtime, |permit| {
            let mut shape = SharedShape::begin_exact(dimensions.rank(), permit)?;
            for axis in 0..dimensions.rank() {
                shape
                    .try_push(
                        dimensions
                            .dimension(axis)
                            .expect("validated broadcast axis"),
                    )
                    .map_err(|_| runmat_value::ExactHostNumericAllocationError::LayoutOverflow)?;
            }
            shape
                .try_finish()
                .map_err(|_| runmat_value::ExactHostNumericAllocationError::LayoutOverflow)
        })
        .map_err(|error| admission(runtime, error))?;
    let mut broadcast =
        PreparedBroadcastPlan::prepare(&metadata, runtime, left.shape(), right.shape(), &shape)
            .map_err(|error| admission(runtime, error))?;
    let mut physical = ExactArrayStorage::<StringElement>::requested_bytes(count)
        .map_err(|error| admission(runtime, numeric(error)))?;
    let mut claims = usize::from(count != 0);
    let mut cooperation = runtime.cpu_cooperation();
    for (_, a, b) in broadcast.iter() {
        metadata
            .checkpoint(runtime)
            .map_err(|error| admission(runtime, error))?;
        let bytes =
            output::byte_length(&left.element(a)?, &right.element(b)?).map_err(text_error)?;
        physical = physical
            .checked_add(
                Utf8Text::requested_bytes_for_capacity(bytes)
                    .map_err(|error| admission(runtime, numeric(error)))?,
            )
            .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?;
        claims = claims
            .checked_add(usize::from(bytes != 0))
            .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?;
        cooperation
            .checkpoint(u64::try_from(bytes).unwrap_or(u64::MAX).saturating_add(1))
            .await
            .map_err(|error| error.into_runtime_error())?;
    }
    metadata
        .checkpoint(runtime)
        .map_err(|error| admission(runtime, error))?;
    let plan = HostAllocationPlan::new(
        0,
        retained(physical, claims).map_err(|error| admission(runtime, error))?,
    )
    .map_err(|error| admission(runtime, error))?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|error| admission(runtime, error))?
        .ok_or_else(|| invalid_input("host allocation authority changed during string addition"))?;
    let permit = receipt
        .output_permit(runtime)
        .map_err(|error| admission(runtime, error))?;
    let mut data = ExactArrayStorage::try_begin(count, &permit)
        .map_err(|error| admission(runtime, numeric(error)))?;
    for (_, a, b) in broadcast.iter() {
        metadata
            .checkpoint(runtime)
            .map_err(|error| admission(runtime, error))?;
        receipt
            .checkpoint(runtime)
            .map_err(|error| admission(runtime, error))?;
        let text = output::concatenate(&left.element(a)?, &right.element(b)?, Some(&permit))
            .map_err(text_error)?;
        let bytes = text.as_str().map_or(0, str::len);
        data.try_push(text)
            .map_err(|_| admission(runtime, HostAllocationError::Overflow))?;
        cooperation
            .checkpoint(u64::try_from(bytes).unwrap_or(u64::MAX).saturating_add(1))
            .await
            .map_err(|error| error.into_runtime_error())?;
    }
    metadata
        .checkpoint(runtime)
        .map_err(|error| admission(runtime, error))?;
    receipt
        .checkpoint(runtime)
        .map_err(|error| admission(runtime, error))?;
    let data = data
        .try_finish()
        .map_err(|_| admission(runtime, HostAllocationError::Overflow))?;
    let array = StringArray::from_admitted(data, shape).map_err(invalid_input)?;
    receipt
        .checkpoint(runtime)
        .map_err(|error| admission(runtime, error))?;
    Ok(Value::StringArray(array))
}

async fn ordinary(
    runtime: Option<&RuntimeContext>,
    left: &Operand<'_>,
    right: &Operand<'_>,
) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(left.shape(), right.shape()).map_err(|error| {
        super::plus_error_with_detail(&runmat_builtins::PLUS_ERROR_SIZE_MISMATCH, error)
    })?;
    let mut values = runmat_value::exact_capacity_vec(plan.len())
        .map_err(|error| text_error(TextError::Allocation(error)))?;
    let mut cooperation = runtime.map(RuntimeContext::cpu_cooperation);
    for (_, a, b) in plan.iter() {
        let text =
            output::concatenate(&left.element(a)?, &right.element(b)?, None).map_err(text_error)?;
        if let Some(cooperation) = cooperation.as_mut() {
            cooperation
                .checkpoint(
                    u64::try_from(text.as_str().map_or(0, str::len))
                        .unwrap_or(u64::MAX)
                        .saturating_add(1),
                )
                .await
                .map_err(|error| error.into_runtime_error())?;
        }
        values.push(text);
    }
    let array = StringArray::from_shared_storage(
        values.into(),
        SharedShape::new_unmanaged(plan.output_shape().to_vec()),
    )
    .map_err(invalid_input)?;
    Ok(Value::StringArray(array))
}
