use crate::accel::residency as accel_residency;
use crate::bytecode::ExecutionContext;
use crate::bytecode::Instr;
use crate::interpreter::engine as interp_engine;
use crate::interpreter::errors::mex;
use crate::runtime::workspace::refresh_workspace_state;
use runmat_accelerate::fusion::FusionStoreMaterialization;
use runmat_accelerate::fusion_exec::{
    execute_centered_gram, execute_elementwise, execute_explained_variance,
    execute_image_normalize, execute_matmul_epilogue, execute_power_step_normalize,
    FusionExecutionRequest,
};
use runmat_accelerate::InstrSpan;
use runmat_accelerate::{FusionKind, ValueOrigin, VarKind};
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;
use std::collections::HashMap;

#[inline]
pub fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Int(_) => "Int",
        Value::Num(_) => "Num",
        Value::Complex(_, _) => "Complex",
        Value::Bool(_) => "Bool",
        Value::LogicalArray(_) => "LogicalArray",
        Value::String(_) => "String",
        Value::StringArray(_) => "StringArray",
        Value::CharArray(_) => "CharArray",
        Value::Tensor(_) => "Tensor",
        Value::ComplexTensor(_) => "ComplexTensor",
        Value::Cell(_) => "Cell",
        Value::Struct(_) => "Struct",
        Value::GpuTensor(_) => "GpuTensor",
        Value::Object(_) => "Object",
        Value::HandleObject(_) => "HandleObject",
        Value::Listener(_) => "Listener",
        Value::FunctionHandle(_) => "FunctionHandle",
        Value::Closure(_) => "Closure",
        Value::ClassRef(_) => "ClassRef",
        Value::MException(_) => "MException",
        Value::OutputList(_) => "OutputList",
    }
}

#[inline]
pub fn summarize_value(i: usize, v: &Value) -> String {
    match v {
        Value::GpuTensor(h) => format!("in#{i}:GpuTensor shape={:?}", h.shape),
        Value::Tensor(t) => format!("in#{i}:Tensor shape={:?}", t.shape),
        Value::Num(n) => format!("in#{i}:Num({n:.6})"),
        Value::Int(n) => format!("in#{i}:Int({})", n.to_i64()),
        Value::Bool(b) => format!("in#{i}:Bool({})", if *b { 1 } else { 0 }),
        Value::String(s) => format!("in#{i}:String({})", s),
        _ => format!("in#{i}:{}", value_kind(v)),
    }
}

pub fn fusion_span_live_result_count(instructions: &[Instr], span: &InstrSpan) -> Option<usize> {
    if span.start > span.end || span.end >= instructions.len() {
        return None;
    }
    let mut current_depth = 0usize;
    for instr in &instructions[span.start..=span.end] {
        let effect = instr.stack_effect()?;
        if current_depth < effect.pops {
            current_depth = effect.pops;
        }
        current_depth = current_depth - effect.pops + effect.pushes;
    }
    Some(current_depth)
}

pub fn fusion_span_has_vm_barrier(instructions: &[Instr], span: &InstrSpan) -> bool {
    if span.start > span.end || span.end >= instructions.len() {
        return true;
    }
    for instr in &instructions[span.start..=span.end] {
        if matches!(
            instr,
            Instr::StoreIndex(_)
                | Instr::StoreSlice(_, _, _, _)
                | Instr::StoreSliceExpr { .. }
                | Instr::StoreIndexCell(_)
                | Instr::StoreMember(_)
                | Instr::StoreMemberOrInit(_)
                | Instr::StoreMemberDynamic
                | Instr::StoreMemberDynamicOrInit
        ) {
            return true;
        }
    }
    fusion_span_live_result_count(instructions, span) != Some(1)
}

pub struct StackSliceGuard<'a> {
    stack: *mut Vec<Value>,
    slice: Option<Vec<Value>>,
    _marker: std::marker::PhantomData<&'a mut Vec<Value>>,
}

impl<'a> StackSliceGuard<'a> {
    pub fn new(stack: &'a mut Vec<Value>, slice_start: usize) -> Self {
        let slice = stack.split_off(slice_start);
        Self {
            stack,
            slice: Some(slice),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn slice(&self) -> &[Value] {
        self.slice.as_ref().expect("stack slice missing").as_slice()
    }

    pub fn commit(mut self) {
        self.slice = None;
    }
}

impl Drop for StackSliceGuard<'_> {
    fn drop(&mut self) {
        if let Some(slice) = self.slice.take() {
            unsafe { (&mut *self.stack).extend(slice) }
        }
    }
}

pub fn gather_fusion_inputs<'a>(
    plan: &'a runmat_accelerate::FusionGroupPlan,
    graph: &runmat_accelerate::AccelGraph,
    stack: &'a mut Vec<Value>,
    vars: &mut Vec<Value>,
    context: &mut ExecutionContext,
) -> Result<
    (
        StackSliceGuard<'a>,
        FusionExecutionRequest<'a>,
        Vec<Option<Value>>,
    ),
    RuntimeError,
> {
    if plan.group.stack_layout.is_none() && !plan.stack_pattern.is_empty() {
        return Err(mex(
            "FusionMissingStackLayout",
            "fusion: missing compile-time stack layout metadata",
        ));
    }
    let required_stack_operands = plan
        .group
        .stack_layout
        .as_ref()
        .map(|layout| layout.required_stack_operands)
        .unwrap_or_else(|| plan.stack_pattern.len());
    let mut inputs: Vec<Option<Value>> = vec![None; plan.inputs.len()];

    for (idx, value) in &plan.constants {
        if let Some(slot) = inputs.get_mut(*idx) {
            if slot.is_none() {
                *slot = Some(value.clone());
            }
        }
    }

    for (idx, value_id) in plan.inputs.iter().enumerate() {
        let info = graph
            .value(*value_id)
            .ok_or_else(|| format!("fusion: missing value metadata for id {value_id}"))?;
        match &info.origin {
            ValueOrigin::Variable { kind, index } => {
                let value =
                    match kind {
                        VarKind::Global => vars
                            .get(*index)
                            .cloned()
                            .ok_or_else(|| format!("fusion: global var {index} out of range"))?,
                        VarKind::Local => {
                            if let Some(frame) = context.call_stack.last() {
                                let absolute = frame.locals_start + index;
                                context.locals.get(absolute).cloned().ok_or_else(|| {
                                    format!("fusion: local var {index} unavailable")
                                })?
                            } else {
                                vars.get(*index).cloned().ok_or_else(|| {
                                    format!("fusion: local var {index} unavailable")
                                })?
                            }
                        }
                    };
                debug_assert!(
                    inputs[idx].is_none(),
                    "fusion: duplicate input slot {} for plan {}",
                    idx,
                    plan.index
                );
                inputs[idx] = Some(value);
            }
            ValueOrigin::Constant | ValueOrigin::NodeOutput { .. } | ValueOrigin::Unknown => {}
        }
    }

    if log::log_enabled!(log::Level::Debug) && interp_engine::fusion_debug_enabled() {
        let stack_needed_preview = required_stack_operands;
        let stack_snapshot: Vec<&Value> = stack.iter().rev().take(stack_needed_preview).collect();
        let stack_kinds: Vec<&'static str> =
            stack_snapshot.iter().rev().map(|v| value_kind(v)).collect();
        let input_meta: Vec<String> = plan
            .inputs
            .iter()
            .enumerate()
            .map(|(i, value_id)| {
                if let Some(info) = graph.value(*value_id) {
                    format!("#{i}:id={} origin={:?}", value_id, info.origin)
                } else {
                    format!("#{i}:id={} origin=<missing>", value_id)
                }
            })
            .collect();
        log::debug!(
            "fusion group {} gather: stack_depth={} stack_needed={} stack_kinds={:?} pattern={:?} inputs={:?}",
            plan.index, stack.len(), stack_needed_preview, stack_kinds, &plan.stack_pattern, input_meta
        );
    }

    if stack.len() < required_stack_operands {
        if interp_engine::fusion_debug_enabled() {
            log::debug!(
                "fusion stack underflow: plan={} needed={} available={} pattern={:?}",
                plan.index,
                required_stack_operands,
                stack.len(),
                plan.stack_pattern
            );
        }
        return Err(mex(
            "FusionStackUnderflow",
            "fusion: stack underflow gathering inputs",
        ));
    }
    let available = required_stack_operands;
    let slice_start = stack.len() - available;
    let stack_guard = StackSliceGuard::new(stack, slice_start);
    let slice = stack_guard.slice().to_vec();
    let mut consumed_inputs: Vec<Option<Value>> = vec![None; plan.inputs.len()];
    let input_positions: HashMap<runmat_accelerate::graph::ValueId, usize> = plan
        .inputs
        .iter()
        .enumerate()
        .map(|(idx, value_id)| (*value_id, idx))
        .collect();

    let allow_stack_value = |val: &Value| {
        if plan.group.kind.is_reduction() {
            matches!(val, Value::GpuTensor(_) | Value::Tensor(_))
        } else {
            true
        }
    };

    if let Some(layout) = plan.group.stack_layout.as_ref() {
        for binding in &layout.bindings {
            let Some(input_idx) = input_positions.get(&binding.value_id).copied() else {
                continue;
            };
            let Some(val) = slice.get(binding.stack_offset).cloned() else {
                continue;
            };
            consumed_inputs[input_idx] = Some(val.clone());
            if inputs[input_idx].is_none() && allow_stack_value(&val) {
                inputs[input_idx] = Some(val);
            }
        }
    } else {
        for (offset, input_idx) in plan.stack_pattern.iter().enumerate() {
            let Some(val) = slice.get(offset).cloned() else {
                continue;
            };
            consumed_inputs[*input_idx] = Some(val.clone());
            if inputs[*input_idx].is_none() && allow_stack_value(&val) {
                inputs[*input_idx] = Some(val);
            }
        }
    }

    for (idx, slot) in inputs.iter_mut().enumerate() {
        if slot.is_some() {
            continue;
        }
        let vid = plan.inputs[idx];
        let info = graph.value(vid);
        if let Some(info) = info {
            match &info.origin {
                ValueOrigin::Variable { kind, index } => {
                    let value_opt = match kind {
                        VarKind::Global => vars.get(*index).cloned(),
                        VarKind::Local => {
                            if let Some(frame) = context.call_stack.last() {
                                let absolute = frame.locals_start + index;
                                context.locals.get(absolute).cloned()
                            } else {
                                vars.get(*index).cloned()
                            }
                        }
                    };
                    if let Some(value) = value_opt {
                        *slot = Some(value);
                        continue;
                    }
                }
                ValueOrigin::Constant => {
                    if let Some(value) = plan.const_values.get(&vid) {
                        *slot = Some(value.clone());
                        continue;
                    }
                }
                _ => {}
            }
        }
        if slot.is_none() {
            if let Some(binding) = graph.var_binding(vid) {
                let value_opt = match binding.kind {
                    VarKind::Global => vars.get(binding.index).cloned(),
                    VarKind::Local => {
                        if let Some(frame) = context.call_stack.last() {
                            let absolute = frame.locals_start + binding.index;
                            context.locals.get(absolute).cloned()
                        } else {
                            vars.get(binding.index).cloned()
                        }
                    }
                };
                if let Some(value) = value_opt {
                    *slot = Some(value);
                    continue;
                }
            }
        }
        if slot.is_none() {
            if let Some(info) = info {
                if let ValueOrigin::NodeOutput { node, .. } = info.origin {
                    if let Some(binding) = graph.node_binding(node) {
                        let value_opt = match binding.kind {
                            VarKind::Global => vars.get(binding.index).cloned(),
                            VarKind::Local => {
                                if let Some(frame) = context.call_stack.last() {
                                    let absolute = frame.locals_start + binding.index;
                                    context.locals.get(absolute).cloned()
                                } else {
                                    vars.get(binding.index).cloned()
                                }
                            }
                        };
                        if let Some(value) = value_opt {
                            *slot = Some(value);
                            continue;
                        }
                    }
                }
            }
        }
        if slot.is_none() {
            if let Some(value) = plan.const_values.get(&vid) {
                *slot = Some(value.clone());
            }
        }
    }

    let inputs: Vec<Value> = inputs
        .into_iter()
        .map(|opt| opt.ok_or_else(|| mex("FusionMissingInput", "fusion: missing input value")))
        .collect::<Result<_, _>>()?;

    if log::log_enabled!(log::Level::Debug) {
        let summaries: Vec<String> = inputs
            .iter()
            .enumerate()
            .map(|(i, v)| summarize_value(i, v))
            .collect();
        log::debug!("fusion inputs runtime: [{}]", summaries.join(", "));
    }

    Ok((
        stack_guard,
        FusionExecutionRequest { plan, inputs },
        consumed_inputs,
    ))
}

pub fn write_elementwise_materialized_stores(
    materialized_stores: Vec<(FusionStoreMaterialization, Value)>,
    vars: &mut Vec<Value>,
    context: &mut ExecutionContext,
) {
    for (store, value) in materialized_stores {
        match store.binding.kind {
            VarKind::Global => {
                let i = store.binding.index;
                if i < vars.len() && !accel_residency::same_gpu_handle(&vars[i], &value) {
                    accel_residency::clear_value(&vars[i]);
                }
                if i >= vars.len() {
                    vars.resize(i + 1, Value::Num(0.0));
                    refresh_workspace_state(vars);
                }
                vars[i] = value;
            }
            VarKind::Local => {
                if let Some(frame) = context.call_stack.last() {
                    let absolute = frame.locals_start + store.binding.index;
                    while context.locals.len() <= absolute {
                        context.locals.push(Value::Num(0.0));
                    }
                    if !accel_residency::same_gpu_handle(&context.locals[absolute], &value) {
                        accel_residency::clear_value(&context.locals[absolute]);
                    }
                    context.locals[absolute] = value;
                } else {
                    let i = store.binding.index;
                    if i < vars.len() && !accel_residency::same_gpu_handle(&vars[i], &value) {
                        accel_residency::clear_value(&vars[i]);
                    }
                    if i >= vars.len() {
                        vars.resize(i + 1, Value::Num(0.0));
                        refresh_workspace_state(vars);
                    }
                    vars[i] = value;
                }
            }
        }
    }
}

pub fn execute_fusion_elementwise(
    request: FusionExecutionRequest<'_>,
    stack_guard: StackSliceGuard<'_>,
    vars: &mut Vec<Value>,
    context: &mut ExecutionContext,
) -> Result<Value, RuntimeError> {
    match execute_elementwise(request) {
        Ok(result) => {
            write_elementwise_materialized_stores(result.materialized_stores, vars, context);
            stack_guard.commit();
            Ok(result.final_value)
        }
        Err(err) => Err(mex("FusionExecutionFailed", &err.to_string())),
    }
}

pub async fn execute_fusion_special_kind(
    kind: FusionKind,
    plan_inputs: &[runmat_accelerate::graph::ValueId],
    request: FusionExecutionRequest<'_>,
    stack_guard: StackSliceGuard<'_>,
) -> Result<Value, RuntimeError> {
    match kind {
        FusionKind::CenteredGram => match execute_centered_gram(request).await {
            Ok(result) => {
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => Err(mex("FusionExecutionFailed", &err.to_string())),
        },
        FusionKind::PowerStepNormalize => match execute_power_step_normalize(request).await {
            Ok(result) => {
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => Err(mex("FusionExecutionFailed", &err.to_string())),
        },
        FusionKind::ExplainedVariance => {
            log::debug!("explained variance plan inputs {:?}", plan_inputs);
            match execute_explained_variance(request).await {
                Ok(result) => {
                    stack_guard.commit();
                    Ok(result)
                }
                Err(err) => {
                    log::debug!("explained variance fusion fallback: {}", err);
                    Err(mex("FusionExecutionFailed", &err.to_string()))
                }
            }
        }
        FusionKind::MatmulEpilogue => match execute_matmul_epilogue(request).await {
            Ok(result) => {
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => Err(mex("FusionExecutionFailed", &err.to_string())),
        },
        FusionKind::ImageNormalize => match execute_image_normalize(request).await {
            Ok(result) => {
                stack_guard.commit();
                Ok(result)
            }
            Err(err) => Err(mex("FusionExecutionFailed", &err.to_string())),
        },
        _ => Err(mex("FusionUnsupportedKind", "fusion: unsupported fusion kind")),
    }
}
