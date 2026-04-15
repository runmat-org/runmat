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
    execute_reduction, FusionExecutionRequest,
};
use runmat_accelerate::InstrSpan;
use runmat_accelerate::{value_is_all_keyword, FusionKind, ShapeInfo, ValueOrigin, VarKind};
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
        _ => Err(mex(
            "FusionUnsupportedKind",
            "fusion: unsupported fusion kind",
        )),
    }
}

pub struct ReductionGeometry {
    pub axis: usize,
    pub reduce_len: usize,
    pub num_slices: usize,
}

pub fn resolve_reduction_geometry(
    plan: &runmat_accelerate::FusionGroupPlan,
    graph: &runmat_accelerate::AccelGraph,
    request: &FusionExecutionRequest<'_>,
    consumed_inputs: &[Option<Value>],
    vars: &Vec<Value>,
    context: &ExecutionContext,
) -> Result<ReductionGeometry, RuntimeError> {
    fn detect_reduce_all(
        plan: &runmat_accelerate::FusionGroupPlan,
        graph: &runmat_accelerate::AccelGraph,
    ) -> bool {
        let mut reduce_all = matches!(
            plan.reduction_axes,
            Some(runmat_accelerate::ReductionAxes::All)
        );
        let has_all = reduce_all
            || plan.constants.values().any(value_is_all_keyword)
            || plan.const_values.values().any(value_is_all_keyword);
        if has_all {
            return true;
        }
        for node_id in &plan.group.nodes {
            if let Some(node) = graph.node(*node_id) {
                if let runmat_accelerate::graph::AccelNodeLabel::Builtin { name } = &node.label {
                    if name.eq_ignore_ascii_case("mean") {
                        for input_vid in &node.inputs {
                            if let Some(info) = graph.value(*input_vid) {
                                if let Some(constant) = &info.constant {
                                    if value_is_all_keyword(constant) {
                                        reduce_all = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if reduce_all {
                break;
            }
        }
        reduce_all
    }

    fn resolve_reduction_axis(plan: &runmat_accelerate::FusionGroupPlan) -> (usize, bool) {
        let mut axis = 0usize;
        let mut axis_explicit = false;
        if let Some(runmat_accelerate::ReductionAxes::Explicit(dims)) = &plan.reduction_axes {
            if let Some(first) = dims.first().copied() {
                axis = first.saturating_sub(1);
                axis_explicit = true;
            }
        }
        if let Some(dim_vid) = plan.reduction_dim {
            if let Some(cv) = plan.const_values.get(&dim_vid) {
                axis = match cv {
                    Value::Num(n) if *n >= 1.0 => (*n as usize).saturating_sub(1),
                    Value::Int(i) => (i.to_f64() as usize).saturating_sub(1),
                    _ => axis,
                };
                axis_explicit = true;
            } else if let Some(input_idx) = plan.inputs.iter().position(|v| *v == dim_vid) {
                if let Some(cv) = plan.constants.get(&input_idx) {
                    axis = match cv {
                        Value::Num(n) if *n >= 1.0 => (*n as usize).saturating_sub(1),
                        Value::Int(i) => (i.to_f64() as usize).saturating_sub(1),
                        _ => axis,
                    };
                    axis_explicit = true;
                }
            }
        } else if let Some(dim_const) = plan.constants.get(&1) {
            axis = match dim_const {
                Value::Num(n) if *n >= 1.0 => (*n as usize).saturating_sub(1),
                Value::Int(i) => (i.to_f64() as usize).saturating_sub(1),
                _ => axis,
            };
            axis_explicit = true;
        }
        (axis, axis_explicit)
    }

    fn derive_rows_cols(
        plan: &runmat_accelerate::FusionGroupPlan,
        graph: &runmat_accelerate::AccelGraph,
        request: &FusionExecutionRequest<'_>,
        consumed_inputs: &[Option<Value>],
        vars: &Vec<Value>,
        context: &ExecutionContext,
    ) -> Option<(usize, usize)> {
        let shape_of = |value: &Value| -> Option<(usize, usize)> {
            match value {
                Value::GpuTensor(h) => Some((
                    h.shape.first().copied().unwrap_or(1).max(1),
                    h.shape.get(1).copied().unwrap_or(1).max(1),
                )),
                Value::Tensor(t) => Some((
                    t.shape.first().copied().unwrap_or(1).max(1),
                    t.shape.get(1).copied().unwrap_or(1).max(1),
                )),
                _ => None,
            }
        };

        if let Some(shape) = plan.reduction_data_shape(graph) {
            if shape.len() >= 2 {
                return Some((shape[0].max(1), shape[1].max(1)));
            }
            if shape.len() == 1 {
                return Some((shape[0].max(1), 1));
            }
        }

        for &vid in &plan.inputs {
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
                    if let Some(shape) = shape_of(&value) {
                        return Some(shape);
                    }
                }
            }
        }

        for v in consumed_inputs.iter().filter_map(|v| v.as_ref()) {
            if let Some(shape) = shape_of(v) {
                return Some(shape);
            }
        }

        if let Some(data_id) = plan.reduction_data {
            if let Some(input_index) = plan.inputs.iter().position(|vid| *vid == data_id) {
                if let Some(val) = consumed_inputs.get(input_index).and_then(|v| v.as_ref()) {
                    if let Some(shape) = shape_of(val) {
                        return Some(shape);
                    }
                }
                if let Some(val) = request.inputs.get(input_index) {
                    if let Some(shape) = shape_of(val) {
                        return Some(shape);
                    }
                }
            }
            if let Some(info) = graph.value(data_id) {
                if let ValueOrigin::Variable { kind, index } = &info.origin {
                    let val = match kind {
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
                    if let Some(v) = val {
                        if let Some(shape) = shape_of(&v) {
                            return Some(shape);
                        }
                    }
                }
                if let ShapeInfo::Tensor(dims) = &info.shape {
                    if !dims.is_empty() {
                        let r = dims.first().and_then(|d| *d).unwrap_or(1);
                        let c = dims.get(1).and_then(|d| *d).unwrap_or(1);
                        return Some((r.max(1), c.max(1)));
                    }
                }
            }
        }

        for v in &request.inputs {
            if let Some(shape) = shape_of(v) {
                return Some(shape);
            }
        }

        if let ShapeInfo::Tensor(dims) = &plan.group.shape {
            if !dims.is_empty() {
                let r = dims.first().and_then(|d| *d).unwrap_or(1);
                let c = dims.get(1).and_then(|d| *d).unwrap_or(1);
                return Some((r.max(1), c.max(1)));
            }
        }
        None
    }

    if log::log_enabled!(log::Level::Debug) {
        let meta: Vec<String> = plan
            .inputs
            .iter()
            .map(|vid| {
                if let Some(info) = graph.value(*vid) {
                    format!(
                        "vid={} origin={:?} shape={:?}",
                        vid, info.origin, info.shape
                    )
                } else {
                    format!("vid={} origin=<missing>", vid)
                }
            })
            .collect();
        log::debug!("reduction gather meta: [{}]", meta.join(", "));
    }

    let reduce_all = detect_reduce_all(plan, graph);
    let (mut axis, axis_explicit) = if reduce_all {
        (0usize, false)
    } else {
        resolve_reduction_axis(plan)
    };
    if reduce_all && interp_engine::fusion_debug_enabled() {
        log::debug!(
            "fusion reduction (all) meta: data_vid={:?} inputs={:?} stack_pattern={:?}",
            plan.reduction_data,
            plan.inputs,
            plan.stack_pattern
        );
    }

    let (r, c) =
        derive_rows_cols(plan, graph, request, consumed_inputs, vars, context).unwrap_or((1, 1));
    let (reduce_len, num_slices) = if reduce_all {
        let total_from_runtime = consumed_inputs
            .iter()
            .filter_map(|v| v.as_ref())
            .chain(request.inputs.iter())
            .find_map(|value| match value {
                Value::GpuTensor(handle) => Some(if handle.shape.is_empty() {
                    1
                } else {
                    handle
                        .shape
                        .iter()
                        .copied()
                        .map(|d| d.max(1))
                        .product::<usize>()
                }),
                Value::Tensor(tensor) => Some(if tensor.shape.is_empty() {
                    1
                } else {
                    tensor
                        .shape
                        .iter()
                        .copied()
                        .map(|d| d.max(1))
                        .product::<usize>()
                }),
                _ => None,
            });
        let total = plan
            .reduction_data_shape(graph)
            .map(|shape| shape.into_iter().map(|d| d.max(1)).product::<usize>())
            .or(total_from_runtime)
            .or_else(|| plan.element_count())
            .filter(|v| *v > 0)
            .ok_or_else(|| {
                mex(
                    "FusionReductionExtentUnknown",
                    "fusion: reduction all extent unknown",
                )
            })?;
        if interp_engine::fusion_debug_enabled() {
            log::debug!(
                "fusion reduction (all): total_elems={} fallback_rows={} fallback_cols={}",
                total,
                r,
                c
            );
        }
        (total, 1usize)
    } else {
        if !axis_explicit {
            axis = if r == 1 && c > 1 {
                1
            } else if r > 1 {
                0
            } else {
                axis
            };
        }
        if interp_engine::fusion_debug_enabled() {
            if r == 1 && c == 1 {
                log::debug!(
                    "fusion reduction: unresolved shape (defaulted to 1x1); axis={}, constants={:?}",
                    axis,
                    plan.constants
                );
            } else {
                log::debug!(
                    "fusion reduction: resolved shape rows={} cols={} axis={} constants={:?}",
                    r,
                    c,
                    axis,
                    plan.constants
                );
            }
        }
        if axis == 0 {
            (r, c)
        } else {
            (c, r)
        }
    };

    if interp_engine::fusion_debug_enabled() {
        log::debug!(
            "fusion reduction: axis={} reduce_len={} num_slices={} constants={:?}",
            axis,
            reduce_len,
            num_slices,
            plan.constants
        );
    }

    let looks_wrong = reduce_len == 1 && num_slices == 1 && {
        let mut big = false;
        let mut check_val = |v: &Value| match v {
            Value::GpuTensor(h) => {
                let prod = h.shape.iter().copied().product::<usize>();
                if prod > 1 {
                    big = true;
                }
            }
            Value::Tensor(t) => {
                let prod = t.shape.iter().copied().product::<usize>();
                if prod > 1 {
                    big = true;
                }
            }
            _ => {}
        };
        for v in consumed_inputs.iter().filter_map(|v| v.as_ref()) {
            check_val(v);
        }
        for v in &request.inputs {
            check_val(v);
        }
        big
    };
    if looks_wrong {
        log::debug!("fusion reduction: skipping fusion due to unresolved shape; falling back to provider path");
        return Err(mex(
            "FusionReductionShapeUnresolved",
            "fusion: reduction shape unresolved",
        ));
    }
    if std::env::var("RUNMAT_DISABLE_FUSED_REDUCTION")
        .ok()
        .as_deref()
        == Some("1")
    {
        return Err(mex(
            "FusionReductionDisabled",
            "fusion: fused reductions disabled",
        ));
    }

    Ok(ReductionGeometry {
        axis,
        reduce_len,
        num_slices,
    })
}

pub fn execute_fusion_reduction(
    plan: &runmat_accelerate::FusionGroupPlan,
    graph: &runmat_accelerate::AccelGraph,
    request: FusionExecutionRequest<'_>,
    consumed_inputs: &[Option<Value>],
    stack_guard: StackSliceGuard<'_>,
    vars: &Vec<Value>,
    context: &ExecutionContext,
) -> Result<Value, RuntimeError> {
    let geom = resolve_reduction_geometry(plan, graph, &request, consumed_inputs, vars, context)?;
    match execute_reduction(request, geom.reduce_len, geom.num_slices, 256u32) {
        Ok(result) => {
            stack_guard.commit();
            Ok(result)
        }
        Err(err) => Err(mex("FusionExecutionFailed", &err.to_string())),
    }
}

pub async fn try_execute_fusion_group(
    plan: &runmat_accelerate::FusionGroupPlan,
    graph: &runmat_accelerate::AccelGraph,
    stack: &mut Vec<Value>,
    vars: &mut Vec<Value>,
    context: &mut ExecutionContext,
) -> Result<Value, RuntimeError> {
    let (stack_guard, request, consumed_inputs) =
        gather_fusion_inputs(plan, graph, stack, vars, context)?;
    log::debug!(
        "dispatch fusion kind {:?}, supported {}",
        plan.group.kind,
        plan.kernel.supported
    );
    if plan.group.kind.is_elementwise() {
        execute_fusion_elementwise(request, stack_guard, vars, context)
    } else if plan.group.kind.is_reduction() {
        execute_fusion_reduction(
            plan,
            graph,
            request,
            &consumed_inputs,
            stack_guard,
            vars,
            context,
        )
    } else {
        execute_fusion_special_kind(plan.group.kind.clone(), &plan.inputs, request, stack_guard)
            .await
    }
}
