//! Preserve parsed field-selector values without binary64 index conversion.
//! Selection/stride arithmetic remains exclusively in the canonical planner.
use super::super::{FieldStep, IndexComponent};
use super::{invalid, Prepared};
use crate::object::indexing::{
    ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use crate::{
    context::host_allocation::{
        host_allocation_runtime_error, prepare_host_array_optional, HostAllocationError,
        HostAllocationPlan, HostAllocationReceipt,
    },
    BuiltinResult,
};
use runmat_value::{IntValue, NumericDType, NumericScalar, NumericStorage, Tensor, Value};

pub(super) async fn path(
    owner: &mut Prepared,
    fields: &[FieldStep],
) -> BuiltinResult<ObjectSubscriptPath> {
    let count = fields
        .iter()
        .try_fold(0usize, |count, field| {
            count.checked_add(1 + usize::from(field.index.is_some()))
        })
        .ok_or_else(|| invalid("native field path extent overflow"))?;
    let mut steps = prepare_host_array_optional(owner.runtime.as_ref(), count)?;
    for field in fields {
        let name = crate::call::argument_text::copy_str(&field.name, owner.runtime.as_ref())?;
        steps
            .try_push(ObjectSubscript::member_retained(name))
            .map_err(|_| invalid("native field path extent changed"))?;
        if let Some(index) = &field.index {
            let mut components =
                prepare_host_array_optional(owner.runtime.as_ref(), index.components.len())?;
            for component in &index.components {
                if matches!(component, IndexComponent::End) {
                    components
                        .try_push(ObjectIndexComponent::End)
                        .map_err(|_| invalid("native selector extent changed"))?;
                    owner.checkpoint(1).await?;
                    continue;
                }
                let value = match component {
                    IndexComponent::Scalar(value) => Value::Int(IntValue::U64(
                        u64::try_from(*value).map_err(|_| invalid("index exceeds uint64"))?,
                    )),
                    IndexComponent::Vector(values, shape) => {
                        let mut values = values.iter().copied();
                        numeric(owner, shape, || values.next()).await?
                    }
                    IndexComponent::LogicalScalar(value) => Value::Bool(*value),
                    IndexComponent::Logical(mask) => {
                        // Preserve the admitted logical mask and its shape for
                        // the canonical object's linear-index planner. A
                        // numeric index tensor would lose mask orientation.
                        Value::LogicalArray(mask.clone())
                    }
                    IndexComponent::End => {
                        unreachable!("deferred selector retained before numeric conversion")
                    }
                };
                components
                    .try_push(ObjectIndexComponent::Value(value))
                    .map_err(|_| invalid("native selector extent changed"))?;
                owner.checkpoint(1).await?;
            }
            let components = components
                .try_finish()
                .map_err(|_| invalid("native selector extent changed"))?;
            steps
                .try_push(ObjectSubscript::parentheses(
                    ObjectIndexSelector::IndexValues { components },
                ))
                .map_err(|_| invalid("native field path extent changed"))?;
        }
        owner.checkpoint(1).await?;
    }
    ObjectSubscriptPath::from_storage(
        steps
            .try_finish()
            .map_err(|_| invalid("native field path extent changed"))?,
    )
}
async fn numeric(
    owner: &mut Prepared,
    shape: &[usize],
    mut next: impl FnMut() -> Option<usize>,
) -> BuiltinResult<Value> {
    let count = shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| invalid("index shape overflow"))?;
    let allocation = |source| {
        host_allocation_runtime_error(
            owner.runtime.as_ref(),
            "RunMat:Object:FieldPathAllocation",
            "native field selector storage",
            source,
        )
    };
    let bytes =
        HostAllocationPlan::retained_tensor_bytes(NumericDType::U64, shape).map_err(allocation)?;
    let plan = HostAllocationPlan::new(0, bytes).map_err(allocation)?;
    let receipt = HostAllocationReceipt::admit(owner.runtime.as_ref(), plan).map_err(allocation)?;
    // The builder owns final selector backing, not a tracking-only copy. A
    // managed result never falls back to an unclaimed ordinary destination.
    let mut managed = match &receipt {
        Some(receipt) => Some(
            receipt
                .exact_output_tensor(
                    owner
                        .runtime
                        .as_ref()
                        .ok_or_else(|| allocation(HostAllocationError::MissingManagedOwner))?,
                    NumericDType::U64,
                    shape,
                )
                .map_err(allocation)?,
        ),
        None => None,
    };
    let mut ordinary = if managed.is_none() {
        Some(runmat_value::exact_capacity_vec(count).map_err(|source| allocation(source.into()))?)
    } else {
        None
    };
    for index in 0..count {
        let value = u64::try_from(
            next().ok_or_else(|| invalid("index shape differs from selector count"))?,
        )
        .map_err(|_| invalid("index exceeds uint64"))?;
        if let Some(output) = managed.as_mut() {
            output
                .set_value(index, NumericScalar::U64(value))
                .map_err(|_| invalid("native index destination rejected exact uint64"))?;
        } else {
            ordinary.as_mut().expect("ordinary destination").push(value);
        }
        owner.checkpoint(1).await?;
    }
    if next().is_some() {
        return Err(invalid("index shape differs from selector count"));
    }
    let allocation = |source| {
        host_allocation_runtime_error(
            owner.runtime.as_ref(),
            "RunMat:Object:FieldPathAllocation",
            "native field selector storage",
            source,
        )
    };
    let tensor = match managed {
        Some(output) => output
            .into_tensor()
            .map_err(|source| allocation(source.into()))?,
        None => Tensor::from_numeric_storage(
            NumericStorage::U64(ordinary.expect("ordinary destination")),
            shape.to_vec(),
        )
        .map_err(|_| invalid("native selector shape is invalid"))?,
    };
    if let Some(receipt) = receipt {
        receipt
            .checkpoint(
                owner
                    .runtime
                    .as_ref()
                    .ok_or_else(|| allocation(HostAllocationError::MissingManagedOwner))?,
            )
            .map_err(allocation)?;
    }
    Ok(Value::Tensor(tensor))
}
