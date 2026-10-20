//! Explicit private destination for transformations that preserve GC identities.
use super::*;
use crate::execution::{RootedValue, RootedValueLease};

pub(crate) fn copy_generation(
    source: &ObjectInstance,
    runtime: Option<&crate::context::RuntimeContext>,
) -> Result<(ObjectInstance, RootedValueLease), crate::RuntimeError> {
    let allocation = |message, error| {
        crate::context::host_allocation::host_allocation_runtime_error(
            runtime,
            "RunMat:Object:Transformation",
            message,
            error,
        )
    };
    let semantic =
        |message| crate::runtime_error::semantic_error("RunMat:Object:Transformation", message);
    let owner = HostAllocationReceipt::capture_owner(runtime)
        .map_err(|e| allocation("object transformation authority is unavailable", e))?;
    let _source = RootedValue::in_optional_context(Value::Object(source.clone()), runtime)?;
    let policy = HostAllocationReceipt::policy(runtime)
        .map_err(|e| allocation("object transformation policy is unavailable", e))?;
    let output = if policy == HostAllocationPolicy::Managed {
        let runtime = runtime.expect("managed object transformation context");
        let shape = preparation::Shape {
            properties: source.property_count(),
            dynamic: source.dynamic_property_count(),
            omission: preparation::Omission::None,
        };
        let receipt = preparation::admit(runtime, source, &shape, std::iter::empty(), &allocation)?;
        let builder =
            preparation::prepare(runtime, source, &shape, &receipt, &allocation, &semantic)?;
        receipt
            .checkpoint(runtime)
            .map_err(|e| allocation("object transformation was cancelled", e))?;
        let output = builder
            .finish()
            .map_err(|e| allocation("object transformation allocation failed", e.into()))?;
        receipt
            .checkpoint(runtime)
            .map_err(|e| allocation("object transformation was cancelled", e))?;
        output
    } else {
        source
            .clone()
            .into_unbounded_builder()
            .map_err(|_| {
                semantic("admitted object cannot enter an unbounded transformation".to_owned())
            })?
            .finish()
    };
    if let Some(owner) = owner {
        owner
            .check(runtime.expect("managed authority context"))
            .map_err(|e| allocation("object transformation authority changed", e))?;
    }
    let output = RootedValue::in_optional_context(Value::Object(output), runtime)?;
    let (value, roots) = output.into_value_and_lease();
    let Value::Object(output) = value else {
        unreachable!("object generation")
    };
    Ok((output, roots))
}
