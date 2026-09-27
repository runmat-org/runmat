use super::{FusionExecutionRequest, ReductionDataInput};
use crate::fusion::typed_producer::producer_indices;
use crate::fusion::{FusionGroupPlan, FusionOp};
use crate::graph::ValueId;
use runmat_value::Value;
use std::collections::{HashMap, HashSet};

pub(super) struct Inputs<'a> {
    pub producer: FusionGroupPlan,
    pub data: Vec<ReductionDataInput<'a>>,
    pub controls: Vec<Value>,
    pub root: ValueId,
    pub terminal: usize,
}

pub(super) fn map<'a>(request: &'a FusionExecutionRequest<'_>) -> Option<Inputs<'a>> {
    let plan = request.plan;
    if !plan.group.kind.is_reduction() || plan.inputs.len() != request.inputs.len() {
        return None;
    }
    let output = plan.output?;
    if plan
        .materialized_stores
        .iter()
        .any(|store| store.value_id != output)
    {
        return None;
    }
    let mut leaves = HashMap::new();
    for (slot, (&id, value)) in plan.inputs.iter().zip(&request.inputs).enumerate() {
        if leaves.insert(id, (Some(slot), value)).is_some() {
            return None;
        }
    }
    for (&id, value) in &plan.const_values {
        // Never let a cached constant overwrite an actual invocation operand.
        if leaves.insert(id, (None, value)).is_some() {
            return None;
        }
    }
    let mut definitions = HashSet::new();
    let mut terminal = None;
    for (index, operation) in plan.operations.iter().enumerate() {
        let (_, result) = operands(operation);
        let result = result?;
        if !definitions.insert(result) || leaves.contains_key(&result) {
            return None;
        }
        if result == output {
            terminal = Some(index);
        }
    }
    let terminal = terminal?;
    let FusionOp::Builtin { inputs, .. } = &plan.operations[terminal] else {
        return None;
    };
    let (&root, control_ids) = inputs.split_first()?;
    if plan.reduction_data.is_some_and(|id| id != root) {
        return None;
    }
    let controls = control_ids
        .iter()
        .map(|id| {
            let (_, value) = leaves.get(id)?;
            // Keep computed and resident controls on ordinary dispatch before any
            // producer-domain admission can submit provider work.
            matches!(
                value,
                Value::Num(_)
                    | Value::Int(_)
                    | Value::Tensor(_)
                    | Value::String(_)
                    | Value::CharArray(_)
                    | Value::StringArray(_)
            )
            .then(|| (*value).clone())
        })
        .collect::<Option<Vec<_>>>()?;
    let expressions = leaves.keys().map(|&id| (id, String::new())).collect();
    let selected = producer_indices(plan, &expressions, &[root]).ok()?;
    if selected.contains(&terminal) || selected.iter().any(|&index| index >= terminal) {
        return None;
    }
    // This receipt cannot authorize dropping other evaluated statements or
    // intermediate publications, even when their results are not the root.
    if selected.len().checked_add(1)? != plan.operations.len() {
        return None;
    }

    let mut used = HashSet::new();
    if leaves.contains_key(&root) {
        used.insert(root);
    }
    for &index in &selected {
        let (inputs, _) = operands(&plan.operations[index]);
        used.extend(inputs.iter().filter(|id| leaves.contains_key(*id)).copied());
    }
    let mut data = Vec::new();
    // Preserve original request slot order; constant leaves use stable ID order.
    for (slot, &id) in plan.inputs.iter().enumerate() {
        if used.contains(&id) {
            data.push(ReductionDataInput {
                value_id: id,
                request_slot: Some(slot),
                value: &request.inputs[slot],
            });
        }
    }
    let mut constant_ids: Vec<_> = used
        .iter()
        .filter(|id| plan.const_values.contains_key(*id))
        .copied()
        .collect();
    constant_ids.sort_unstable();
    for id in constant_ids {
        data.push(ReductionDataInput {
            value_id: id,
            request_slot: None,
            value: &plan.const_values[&id],
        });
    }
    // No payload projection or shape-based dtype relabeling.
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
        return None;
    }
    let mut producer = plan.clone();
    producer.operations = plan
        .operations
        .iter()
        .enumerate()
        .filter(|(index, _)| selected.contains(index))
        .map(|(_, op)| op.clone())
        .collect();
    producer.inputs = data.iter().map(|input| input.value_id).collect();
    producer.constants.clear();
    producer.const_values.clear();
    producer.materialized_stores.clear();
    producer.output = Some(root);
    Some(Inputs {
        producer,
        data,
        controls,
        root,
        terminal,
    })
}

fn operands(operation: &FusionOp) -> (&[ValueId], Option<ValueId>) {
    match operation {
        FusionOp::Primitive { inputs, output, .. } | FusionOp::Builtin { inputs, output, .. } => {
            (inputs, *output)
        }
    }
}
