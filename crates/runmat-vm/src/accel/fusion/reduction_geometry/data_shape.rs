//! Resolve only the actual reduction operand's dependency graph, not unrelated inputs.
use super::cardinality;
use crate::interpreter::errors::mex;
use runmat_accelerate::fusion::FusionOp;
use runmat_accelerate::graph::{PrimitiveOp, ValueId};
use runmat_accelerate::FusionGroupPlan;
use runmat_runtime::{value_fact::value_fact, RuntimeError};
use runmat_types::{
    infer_binary, infer_unary, CallRequest, LiteralContext, OperatorKind, OutputSelection,
    ValueFact,
};
use runmat_value::Value;
use std::collections::{HashMap, HashSet};

fn unknown() -> RuntimeError {
    mex(
        "FusionReductionShapeUnresolved",
        "fusion: exact reduction data shape is not proven",
    )
}

fn dimensions(fact: &ValueFact) -> Result<Vec<usize>, RuntimeError> {
    fact.shape
        .known_dims()
        .and_then(|dims| dims.into_iter().collect())
        .ok_or_else(unknown)
}

fn primitive(op: PrimitiveOp, args: &[ValueFact]) -> Result<ValueFact, RuntimeError> {
    let operator = match op {
        PrimitiveOp::Add => OperatorKind::Add,
        PrimitiveOp::Sub => OperatorKind::Subtract,
        PrimitiveOp::ElemMul => OperatorKind::ElementwiseMultiply,
        PrimitiveOp::ElemDiv => OperatorKind::ElementwiseDivide,
        PrimitiveOp::ElemLeftDiv => OperatorKind::ElementwiseLeftDivide,
        PrimitiveOp::ElemPow => OperatorKind::ElementwisePower,
        PrimitiveOp::Neg => OperatorKind::UnaryMinus,
        PrimitiveOp::UPlus => OperatorKind::UnaryPlus,
        PrimitiveOp::Mul
            if args.iter().any(|fact| {
                dimensions(fact).ok().and_then(|dims| cardinality(&dims)) == Some(1)
            }) =>
        {
            OperatorKind::MatrixMultiply
        }
        // A scalar expression kernel cannot implement matrix products or transposes.
        _ => return Err(unknown()),
    };
    let inferred = match args {
        [arg] => infer_unary(operator, arg),
        [left, right] => infer_binary(operator, left, right),
        _ => return Err(unknown()),
    };
    if !inferred.diagnostics.is_empty() {
        return Err(unknown());
    }
    Ok(inferred.fact)
}

fn resolve_fact(
    id: ValueId,
    facts: &mut HashMap<ValueId, ValueFact>,
    producers: &HashMap<ValueId, &FusionOp>,
    visiting: &mut HashSet<ValueId>,
) -> Result<ValueFact, RuntimeError> {
    if let Some(fact) = facts.get(&id) {
        return Ok(fact.clone());
    }
    if !visiting.insert(id) {
        return Err(unknown());
    }
    let op = producers.get(&id).ok_or_else(unknown)?;
    let inputs = match op {
        FusionOp::Primitive { inputs, .. } | FusionOp::Builtin { inputs, .. } => inputs,
    };
    let arguments = inputs
        .iter()
        .map(|id| resolve_fact(*id, facts, producers, visiting))
        .collect::<Result<Vec<_>, _>>()?;
    let fact = match op {
        FusionOp::Primitive { op, .. } => primitive(*op, &arguments)?,
        FusionOp::Builtin { name, .. } => {
            let entry = runmat_builtins::builtin_catalog_entry_by_name(name).ok_or_else(unknown)?;
            let [binding] = entry.bindings else {
                return Err(unknown());
            };
            let declaration = runmat_runtime::builtins::common::spec::canonical_fusion_declaration(
                entry.binding_identity(binding),
            )
            .ok_or_else(unknown)?;
            if declaration.structural.is_some() {
                return Err(unknown());
            }
            let inference = runmat_builtins::infer_catalog_call(
                entry,
                &CallRequest {
                    arguments,
                    literals: LiteralContext::default(),
                    outputs: OutputSelection::default(),
                },
            );
            if !inference.diagnostics.is_empty() || inference.outputs.len() != 1 {
                return Err(unknown());
            }
            inference.outputs.into_iter().next().ok_or_else(unknown)?
        }
    };
    dimensions(&fact)?;
    visiting.remove(&id);
    facts.insert(id, fact.clone());
    Ok(fact)
}

pub(super) fn resolve(
    plan: &FusionGroupPlan,
    inputs: &[Value],
) -> Result<Vec<usize>, RuntimeError> {
    if plan.inputs.len() != inputs.len() {
        return Err(unknown());
    }
    let mut facts = HashMap::new();
    for (id, value) in plan.inputs.iter().zip(inputs) {
        // Kernel buffer admission is narrower than arbitrary Value shape inference.
        match value {
            Value::Num(_) | Value::Tensor(_) => {}
            Value::GpuTensor(handle)
                if handle.view().layout() == runmat_accelerate_api::GpuTensorLayout::Contiguous => {
            }
            _ => return Err(unknown()),
        }
        if facts.insert(*id, value_fact(value)).is_some() {
            return Err(unknown());
        }
    }
    for (id, value) in &plan.const_values {
        // Actual operands take precedence over any cached constant fact.
        facts.entry(*id).or_insert_with(|| value_fact(value));
    }
    let mut producers = HashMap::new();
    for op in &plan.operations {
        let output = match op {
            FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => output,
        };
        if let Some(output) = output {
            if producers.insert(*output, op).is_some() {
                return Err(unknown());
            }
        }
    }
    let data = plan.reduction_data.ok_or_else(unknown)?;
    let shape = dimensions(&resolve_fact(
        data,
        &mut facts,
        &producers,
        &mut HashSet::new(),
    )?)?;
    // The current reduction shader loads every physical input at the same flat
    // index. Shape inference does not grant it broadcasting or lazy-layout support.
    // Keep those expressions on the interpreter until actual stride-aware loads exist.
    for value in inputs {
        if !value_fact(value)
            .shape
            .is_proven_equivalent(&facts.get(&data).ok_or_else(unknown)?.shape)
        {
            return Err(mex(
                "FusionReductionBroadcastUnsupported",
                "fusion: reduction kernel requires equal-shape contiguous inputs",
            ));
        }
    }
    Ok(shape)
}
