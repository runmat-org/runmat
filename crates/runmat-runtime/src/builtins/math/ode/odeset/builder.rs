//! One shared empty default and one final admitted structure publication.
use super::*;
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationPolicy,
    HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use runmat_value::{NumericDType, Tensor};

pub(super) fn finish(
    runtime: Option<&RuntimeContext>,
    values: [Option<&Value>; OdeOptionField::ALL.len()],
) -> BuiltinResult<crate::execution::RootedValue> {
    let empty = if values.iter().any(Option::is_none) {
        Some(empty(runtime)?)
    } else {
        None
    };
    crate::execution::structure::construct_struct_from_fields_in_optional_context(
        runtime,
        OdeOptionField::ALL
            .into_iter()
            .zip(values)
            .map(|(field, value)| {
                (
                    field.public_name(),
                    value.unwrap_or_else(|| {
                        empty
                            .as_ref()
                            .expect("missing fields have one empty default")
                            .value()
                    }),
                )
            }),
    )
}

fn empty(runtime: Option<&RuntimeContext>) -> BuiltinResult<crate::execution::RootedValue> {
    let allocation = |source| {
        host_allocation_runtime_error(
            runtime,
            "RunMat:odeset:Allocation",
            "odeset: default option backing admission failed",
            source,
        )
    };
    let shape = [0, 0];
    let bytes =
        HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &shape).map_err(allocation)?;
    let plan = HostAllocationPlan::new(0, bytes).map_err(allocation)?;
    let receipt = HostAllocationReceipt::admit(runtime, plan).map_err(allocation)?;
    let tensor = match (runtime, receipt) {
        (Some(runtime), Some(receipt)) => {
            let tensor = receipt
                .exact_output_tensor(runtime, NumericDType::F64, &shape)
                .map_err(allocation)?
                .into_tensor()
                .map_err(|source| allocation(HostAllocationError::Numeric(source)))?;
            receipt.checkpoint(runtime).map_err(allocation)?;
            tensor
        }
        (_, None) => {
            if HostAllocationReceipt::policy(runtime).map_err(allocation)?
                != HostAllocationPolicy::Unbounded
            {
                return Err(allocation(HostAllocationError::StaleAuthority));
            }
            let tensor = Tensor::zeros(shape.to_vec());
            if HostAllocationReceipt::policy(runtime).map_err(allocation)?
                != HostAllocationPolicy::Unbounded
            {
                return Err(allocation(HostAllocationError::StaleAuthority));
            }
            tensor
        }
        (None, Some(_)) => return Err(allocation(HostAllocationError::MissingManagedOwner)),
    };
    crate::execution::RootedValue::in_optional_context(Value::Tensor(tensor), runtime)
}
