//! Class-literal defaults and overrides in one admitted object generation.

use super::{effective_class, RuntimeClassState, SelectedClassProjection};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValues};
use crate::RuntimeError;
use runmat_types::ClassIdentity;
use runmat_value::Value;

/// Literal fields override inherited nonstatic defaults. Borrow the registry's
/// existing values and the caller's roots until the destination is published;
/// no temporary object or replacement property map is constructed.
pub fn construct_class_literal_in_context(
    runtime: &RuntimeContext,
    class: &ClassIdentity,
    names: &[String],
    values: RootedValues,
) -> Result<RootedValue, RuntimeError> {
    if names.len() != values.len() {
        return Err(crate::runtime_error::semantic_error(
            "ObjectConstruction",
            "object property names and values have different lengths",
        ));
    }
    let state = runtime.state().classes.borrow();
    let selected = runtime.selected_class_projection().map(|value| &**value);
    let depth = ancestry_depth(&state, selected, class)?;
    let ancestry = || {
        std::iter::successors(effective_class(&state, selected, class), |definition| {
            definition
                .parent
                .as_ref()
                .and_then(|parent| effective_class(&state, selected, parent))
        })
    };
    let empty = ancestry()
        .flat_map(|definition| definition.properties.values())
        .any(|property| !property.is_static && property.default_value.is_none())
        .then(|| empty_default(runtime))
        .transpose()?;
    // Class hierarchies are shallow metadata. Rewalking them in root-first
    // order avoids an intermediate ancestry vector and preserves override order.
    let defaults = (0..depth)
        .rev()
        .filter_map(|index| ancestry().nth(index))
        .flat_map(|definition| definition.properties.iter())
        .filter(|(_, property)| !property.is_static)
        .map(|(name, property)| {
            (
                name.0.as_str(),
                property
                    .default_value
                    .as_ref()
                    .or(empty.as_ref())
                    .expect("missing nonstatic defaults have an admitted empty value"),
            )
        });
    let selected_object = runtime
        .selected_object_class()
        .filter(|facts| facts.class() == class)
        .cloned();
    crate::execution::object::construct_registered_object_in_context(
        runtime,
        class.display_name(),
        defaults.chain(names.iter().map(String::as_str).zip(values.iter())),
        selected_object,
    )
}

fn empty_default(runtime: &RuntimeContext) -> Result<Value, RuntimeError> {
    use crate::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationPlan, HostAllocationReceipt,
    };
    let allocation = |source| {
        host_allocation_runtime_error(
            Some(runtime),
            "ObjectConstruction",
            "empty property default allocation failed",
            source,
        )
    };
    let shape = [0, 0];
    let dtype = runmat_value::NumericDType::F64;
    let retained = HostAllocationPlan::retained_tensor_bytes(dtype, &shape).map_err(allocation)?;
    let plan = HostAllocationPlan::new(0, retained).map_err(allocation)?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), plan).map_err(allocation)?;
    let tensor = if let Some(receipt) = receipt {
        let output = receipt
            .exact_output_tensor(runtime, dtype, &shape)
            .map_err(allocation)?;
        let tensor = output
            .into_tensor()
            .map_err(|source| allocation(source.into()))?;
        receipt.checkpoint(runtime).map_err(allocation)?;
        tensor
    } else {
        runmat_value::Tensor::zeros(shape.to_vec())
    };
    Ok(Value::Tensor(tensor))
}

fn ancestry_depth(
    state: &RuntimeClassState,
    selected: Option<&SelectedClassProjection>,
    class: &ClassIdentity,
) -> Result<usize, RuntimeError> {
    let limit = state
        .classes
        .len()
        .saturating_add(selected.map_or(0, |projection| projection.classes.len()));
    let mut current = effective_class(state, selected, class);
    let mut depth = 0;
    while let Some(definition) = current {
        if depth >= limit {
            return Err(crate::runtime_error::semantic_error(
                "ObjectConstruction",
                "class ancestry contains a cycle",
            ));
        }
        depth += 1;
        current = definition
            .parent
            .as_ref()
            .and_then(|parent| effective_class(state, selected, parent));
    }
    Ok(depth)
}
