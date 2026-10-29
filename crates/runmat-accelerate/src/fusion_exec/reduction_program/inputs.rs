//! Checked input-mapping storage for the typed reduction source. No hashes
//! grow during this projection: lookups borrow the original request and the
//! one selected-producer bit vector remains the canonical traversal owner.
use super::{FusionExecutionRequest, ReductionDataInput};
use crate::fusion::typed_producer::producer_indices;
use crate::fusion::{FusionGroup, FusionGroupPlan, FusionOp, FusionStackLayout};
use crate::graph::{ShapeInfo, ValueId};
use runmat_execution::resource::{ProviderHostMetadataAdmission, RetainedHostAllocationClaim};
use runmat_value::{exact_capacity_vec, exact_capacity_vec_from_slice, Value};
use std::collections::HashMap;
use std::sync::Arc;

pub(super) struct Inputs<'a> {
    pub producer: FusionGroupPlan,
    pub data: Vec<ReductionDataInput<'a>>,
    pub controls: Vec<Value>,
    pub root: ValueId,
    pub terminal: usize,
    pub producer_indices: Vec<usize>,
    // All owned source/mapping blocks above drop before this aggregate claim.
    pub(super) _mapping_claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

fn add_array<T>(bytes: &mut usize, count: usize) -> anyhow::Result<()> {
    *bytes = bytes
        .checked_add(std::alloc::Layout::array::<T>(count)?.size())
        .ok_or_else(|| anyhow::anyhow!("reduction mapping storage overflow"))?;
    Ok(())
}

fn mapping_bytes(
    request: &FusionExecutionRequest<'_>,
    control_count: usize,
) -> anyhow::Result<usize> {
    let plan = request.plan;
    let leaves = plan
        .inputs
        .len()
        .checked_add(plan.const_values.len())
        .ok_or_else(|| anyhow::anyhow!("reduction leaf count overflow"))?;
    let operations = plan.operations.len();
    let operands = plan
        .operations
        .iter()
        .try_fold(0usize, |count, op| count.checked_add(operands(op).0.len()))
        .ok_or_else(|| anyhow::anyhow!("reduction operand count overflow"))?;
    let mut bytes = 0;
    add_array::<ValueId>(&mut bytes, leaves)?; // sorted leaf IDs
    bytes = bytes
        .checked_add(usize::try_from(
            crate::fusion_exec::binding_storage_bytes(operations)
                .ok_or_else(|| anyhow::anyhow!("reduction selected binding size overflow"))?,
        )?)
        .ok_or_else(|| anyhow::anyhow!("reduction binding storage overflow"))?;
    add_array::<runmat_builtins::BuiltinBindingIdentity>(&mut bytes, operations)?;
    add_array::<Option<crate::fusion::semantic_bindings::SemanticBinding>>(&mut bytes, operations)?;
    add_array::<&Value>(&mut bytes, leaves)?;
    add_array::<(ValueId, usize)>(&mut bytes, operations)?; // producer index
    add_array::<bool>(&mut bytes, operations)?; // selected bits
    add_array::<(ValueId, usize)>(
        &mut bytes,
        operands
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("reduction traversal count overflow"))?,
    )?;
    add_array::<Value>(&mut bytes, control_count)?;
    add_array::<ValueId>(&mut bytes, plan.const_values.len())?;
    // The final filtered data projection may use a different Vec backing.
    add_array::<ReductionDataInput<'_>>(&mut bytes, leaves)?;
    add_array::<ReductionDataInput<'_>>(&mut bytes, leaves)?;
    add_array::<usize>(&mut bytes, operations)?;
    add_array::<FusionOp>(&mut bytes, operations)?;
    add_array::<ValueId>(&mut bytes, operands)?;
    add_array::<ValueId>(&mut bytes, leaves)?;
    add_array::<crate::graph::NodeId>(&mut bytes, plan.group.nodes.len())?;
    if let ShapeInfo::Tensor(shape) = &plan.group.shape {
        add_array::<Option<usize>>(&mut bytes, shape.len())?;
    }
    if let Some(layout) = &plan.group.stack_layout {
        add_array::<crate::fusion::FusionStackValueBinding>(&mut bytes, layout.bindings.len())?;
        add_array::<crate::fusion::FusionStackValueBinding>(
            &mut bytes,
            layout.compiler_variable_bindings.len(),
        )?;
    }
    for op in &plan.operations {
        if let FusionOp::Builtin { name, .. } = op {
            bytes = bytes
                .checked_add(name.len())
                .ok_or_else(|| anyhow::anyhow!("reduction copied binding name overflow"))?;
        }
    }
    // The eligibility check constructs these actual facts one at a time.
    // Summing all possible data leaves is a checked overlapping upper bound.
    for value in request.inputs.iter().chain(plan.const_values.values()) {
        let rank = match value {
            Value::Num(_) => Some(2),
            Value::Tensor(tensor) => Some(tensor.shape().len()),
            Value::GpuTensor(handle) => Some(handle.shape().len()),
            _ => None,
        };
        if let Some(rank) = rank {
            let payload =
                crate::fusion::semantics::preparation_resources::source_fact_payload_bytes(rank)?;
            bytes = bytes
                .checked_add(usize::try_from(payload)?)
                .ok_or_else(|| anyhow::anyhow!("reduction source fact storage overflow"))?;
        }
    }
    Ok(bytes)
}

fn copy_operation(operation: &FusionOp) -> anyhow::Result<FusionOp> {
    Ok(match operation {
        FusionOp::Primitive { op, inputs, output } => FusionOp::Primitive {
            op: *op,
            inputs: exact_capacity_vec_from_slice(inputs)?,
            output: *output,
        },
        FusionOp::Builtin {
            name,
            origin,
            inputs,
            output,
        } => FusionOp::Builtin {
            name: String::from_utf8(exact_capacity_vec_from_slice(name.as_bytes())?)?,
            origin: *origin,
            inputs: exact_capacity_vec_from_slice(inputs)?,
            output: *output,
        },
    })
}

pub(super) fn map<'a>(
    request: &'a FusionExecutionRequest<'_>,
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
) -> anyhow::Result<Option<Inputs<'a>>> {
    let plan = request.plan;
    if !plan.group.kind.is_reduction()
        || plan.inputs.len() != request.inputs.len()
        || plan.group.pattern.is_some()
    {
        return Ok(None);
    }
    let Some(output) = plan.output else {
        return Ok(None);
    };
    if plan
        .materialized_stores
        .iter()
        .any(|store| store.value_id != output)
    {
        return Ok(None);
    }
    let leaf = |id: ValueId| -> Option<(Option<usize>, &Value)> {
        if let Some(slot) = plan.inputs.iter().position(|&candidate| candidate == id) {
            Some((Some(slot), &request.inputs[slot]))
        } else {
            plan.const_values.get(&id).map(|value| (None, value))
        }
    };
    for (position, id) in plan.inputs.iter().enumerate() {
        if plan.inputs[..position].contains(id) {
            return Ok(None);
        }
    }
    let mut terminal = None;
    for (index, operation) in plan.operations.iter().enumerate() {
        let Some(result) = operands(operation).1 else {
            return Ok(None);
        };
        if leaf(result).is_some()
            || plan.operations[..index]
                .iter()
                .any(|op| operands(op).1 == Some(result))
        {
            return Ok(None);
        }
        if result == output {
            terminal = Some(index);
        }
    }
    let Some(terminal) = terminal else {
        return Ok(None);
    };
    let FusionOp::Builtin { inputs, .. } = &plan.operations[terminal] else {
        return Ok(None);
    };
    let Some((&root, control_ids)) = inputs.split_first() else {
        return Ok(None);
    };
    if plan.reduction_data.is_some_and(|id| id != root) {
        return Ok(None);
    }
    if control_ids.iter().any(|&id| {
        !leaf(id).is_some_and(|(_, value)| {
            matches!(
                value,
                Value::Num(_)
                    | Value::Int(_)
                    | Value::Tensor(_)
                    | Value::String(_)
                    | Value::CharArray(_)
                    | Value::StringArray(_)
            )
        })
    }) {
        return Ok(None);
    }

    let claim = crate::wgsl::host_buffers::claim_bytes(
        metadata,
        mapping_bytes(request, control_ids.len())?,
    )?;
    let capacity = plan
        .inputs
        .len()
        .checked_add(plan.const_values.len())
        .ok_or_else(|| anyhow::anyhow!("reduction leaf count overflow"))?;
    let mut leaf_ids = exact_capacity_vec(capacity)?;
    leaf_ids.extend(plan.inputs.iter().chain(plan.const_values.keys()).copied());
    leaf_ids.sort_unstable();
    leaf_ids.dedup();
    let selected = match producer_indices(plan, &leaf_ids, &[root]) {
        Ok(selected) => selected,
        Err(error)
            if error
                .downcast_ref::<runmat_value::ExactHostNumericAllocationError>()
                .is_some()
                || error
                    .downcast_ref::<std::collections::TryReserveError>()
                    .is_some() =>
        {
            return Err(error)
        }
        Err(_) => return Ok(None),
    };
    if selected.contains(&terminal)
        || selected.iter().any(|index| index >= terminal)
        || selected.len().checked_add(1) != Some(plan.operations.len())
    {
        return Ok(None);
    }
    let used = |id: ValueId| {
        (id == root && leaf(root).is_some())
            || selected.iter().any(|index| {
                operands(&plan.operations[index]).0.contains(&id) && leaf(id).is_some()
            })
    };
    let mut controls = exact_capacity_vec(control_ids.len())?;
    for &id in control_ids {
        controls.push(leaf(id).expect("checked control leaf").1.clone());
    }
    let mut data = exact_capacity_vec(capacity)?;
    for (slot, &id) in plan.inputs.iter().enumerate() {
        if used(id) {
            data.push(ReductionDataInput {
                value_id: id,
                request_slot: Some(slot),
                value: &request.inputs[slot],
            });
        }
    }
    let mut constant_ids = exact_capacity_vec(plan.const_values.len())?;
    constant_ids.extend(
        plan.const_values
            .keys()
            .filter(|&&id| used(id) && !plan.inputs.contains(&id))
            .copied(),
    );
    constant_ids.sort_unstable();
    for id in constant_ids {
        data.push(ReductionDataInput {
            value_id: id,
            request_slot: None,
            value: &plan.const_values[&id],
        });
    }
    if data.iter().any(|input| {
        let fact = runmat_runtime::value_fact::value_fact(input.value);
        !fact.numeric().is_some_and(|numeric| {
            numeric.domain == runmat_types::NumericDomain::Real
                && matches!(
                    numeric.class,
                    runmat_types::NumericClass::Single | runmat_types::NumericClass::Double
                )
        }) || !matches!(
            input.value,
            Value::Num(_) | Value::Tensor(_) | Value::GpuTensor(_)
        )
    }) {
        return Ok(None);
    }
    let mut indices = exact_capacity_vec(plan.operations.len())?;
    let mut operations = exact_capacity_vec(plan.operations.len())?;
    for index in selected.iter() {
        indices.push(index);
        operations.push(copy_operation(&plan.operations[index])?);
    }
    let mut ids = exact_capacity_vec(capacity)?;
    ids.extend(data.iter().map(|input| input.value_id));
    let shape = match &plan.group.shape {
        ShapeInfo::Tensor(dims) => ShapeInfo::Tensor(exact_capacity_vec_from_slice(dims)?),
        other => other.clone(),
    };
    let stack_layout = plan
        .group
        .stack_layout
        .as_ref()
        .map(|layout| -> anyhow::Result<_> {
            Ok(FusionStackLayout {
                required_stack_operands: layout.required_stack_operands,
                bindings: exact_capacity_vec_from_slice(&layout.bindings)?,
                compiler_variable_bindings: exact_capacity_vec_from_slice(
                    &layout.compiler_variable_bindings,
                )?,
            })
        })
        .transpose()?;
    let producer = FusionGroupPlan {
        index: plan.index,
        group: FusionGroup {
            id: plan.group.id,
            kind: plan.group.kind.clone(),
            nodes: exact_capacity_vec_from_slice(&plan.group.nodes)?,
            shape,
            span: plan.group.span.clone(),
            pattern: None,
            stack_layout,
        },
        operations,
        inputs: ids,
        stack_pattern: Vec::new(),
        constants: HashMap::new(),
        const_values: HashMap::new(),
        materialized_stores: Vec::new(),
        output: Some(root),
        kernel: plan.kernel.clone(),
        reduction_data: None,
        reduction_dim: None,
        reduction_flavor: None,
        reduction_axes: None,
        pattern: None,
    };
    Ok(Some(Inputs {
        producer,
        data,
        controls,
        root,
        terminal,
        producer_indices: indices,
        _mapping_claim: claim,
    }))
}

fn operands(operation: &FusionOp) -> (&[ValueId], Option<ValueId>) {
    match operation {
        FusionOp::Primitive { inputs, output, .. } | FusionOp::Builtin { inputs, output, .. } => {
            (inputs, *output)
        }
    }
}
