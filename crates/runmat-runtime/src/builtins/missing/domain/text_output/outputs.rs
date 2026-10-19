//! Exact output-list publication for the string-array fill result.

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_value::{LogicalArray, OutputListInitializationError, OutputListStorage, Value};

use super::super::errors::allocation_error;

pub(in crate::builtins::missing::domain) fn fill_outputs(
    runtime: Option<&RuntimeContext>,
    result: Value,
    mask: LogicalArray,
) -> BuiltinResult<Value> {
    let Some(count) = crate::output_count::current_output_count() else {
        return Ok(result);
    };
    let Some(runtime) = runtime else {
        return Ok(ordinary(count, result, mask));
    };
    if HostAllocationReceipt::policy(Some(runtime))
        .map_err(|cause| allocation_error(runtime, "fillmissing", cause))?
        == HostAllocationPolicy::Unbounded
    {
        return Ok(ordinary(count, result, mask));
    }
    let raw = u64::try_from(OutputListStorage::requested_bytes(count).map_err(|cause| {
        allocation_error(runtime, "fillmissing", HostAllocationError::Numeric(cause))
    })?)
    .map_err(|_| allocation_error(runtime, "fillmissing", HostAllocationError::Overflow))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(raw, usize::from(count != 0))
        .map_err(|cause| allocation_error(runtime, "fillmissing", cause))?;
    let plan = HostAllocationPlan::new(0, retained)
        .map_err(|cause| allocation_error(runtime, "fillmissing", cause))?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|cause| allocation_error(runtime, "fillmissing", cause))?
        .ok_or_else(|| {
            allocation_error(
                runtime,
                "fillmissing",
                HostAllocationError::MissingManagedOwner,
            )
        })?;
    let mut result = Some(result);
    let mut mask = Some(mask);
    let values = receipt
        .with_output_construction(runtime, |permit| {
            OutputListStorage::try_initialize(count, permit, |index| {
                Ok::<Value, runmat_value::ExactHostNumericAllocationError>(match index {
                    0 => result.take().expect("first fillmissing output"),
                    1 => Value::LogicalArray(mask.take().expect("second fillmissing output")),
                    _ => Value::Num(0.0),
                })
            })
        })
        .map_err(|cause| allocation_error(runtime, "fillmissing", cause))?
        .map_err(|cause| {
            allocation_error(
                runtime,
                "fillmissing",
                match cause {
                    OutputListInitializationError::Allocation(cause)
                    | OutputListInitializationError::Element(cause) => {
                        HostAllocationError::Numeric(cause)
                    }
                },
            )
        })?;
    receipt
        .checkpoint(runtime)
        .map_err(|cause| allocation_error(runtime, "fillmissing", cause))?;
    Ok(Value::OutputList(values))
}

fn ordinary(count: usize, result: Value, mask: LogicalArray) -> Value {
    match count {
        0 => Value::output_list(Vec::new()),
        1 => Value::output_list(vec![result]),
        n => crate::output_count::output_list_with_padding(
            n,
            vec![result, Value::LogicalArray(mask)],
        ),
    }
}
