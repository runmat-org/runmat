use crate::bytecode::program::ExecutionContext;
use crate::bytecode::Instr;
use crate::interpreter::errors::mex;
use crate::interpreter::vars::InterpreterVars;
use runmat_accelerate::fusion::FusionStoreMaterialization;
use runmat_accelerate::fusion_exec::{
    elementwise_floating_admission, execute_elementwise_placed, execute_explained_variance_placed,
    execute_matmul_epilogue_placed, execute_power_step_normalize_placed,
    execute_reduction_with_shape_placed_async, execute_reduction_with_shape_placed_async_vm,
    CpuReadbackFailure, ElementwiseFloatingAdmission, FusionExecutionOutcome,
    FusionExecutionRequest, FusionProviderCompletion,
};
use runmat_accelerate::placement::{
    record_event, PlacementCorrelationId, PlacementEventKind, PlacementVariant,
};
use runmat_accelerate::InstrSpan;
use runmat_accelerate::{FusionKind, VarKind};
use runmat_runtime::builtins::common::shape::is_scalar_shape;
use runmat_runtime::RuntimeError;
use runmat_time::{duration_ns_saturating, Instant};
use runmat_value::Value;
use std::collections::HashSet;

#[path = "fusion/input_transport.rs"]
pub(crate) mod input_transport;

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
        Value::Symbolic(_) => "Symbolic",
        Value::SymbolicArray(_) => "SymbolicArray",
        Value::CharArray(_) => "CharArray",
        Value::Tensor(_) => "Tensor",
        Value::SparseTensor(_) => "SparseTensor",
        Value::ComplexTensor(_) => "ComplexTensor",
        Value::Cell(_) => "Cell",
        Value::Struct(_) => "Struct",
        Value::StructArray(_) => "StructArray",
        Value::GpuTensor(_) => "GpuTensor",
        Value::ObjectArray(_) => "ObjectArray",
        Value::Object(_) => "Object",
        Value::HandleObject(_) => "HandleObject",
        Value::Listener(_) => "Listener",
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_) => "FunctionHandle",
        Value::BoundFunctionHandle(_) | Value::ScopedFunctionHandle(_) => "FunctionHandle",
        Value::Closure(_) => "Closure",
        Value::ClassRef(_) => "ClassRef",
        Value::MException(_) => "MException",
        Value::OutputList(_) => "OutputList",
        Value::Future(_) => "Future",
        Value::Task(_) => "Task",
        Value::Pool(_) => "Pool",
        Value::Job(_) => "Job",
        Value::Distributed(_) => "Distributed",
        Value::Composite(_) => "Composite",
        Value::Foreign(_) => "Foreign",
    }
}

#[inline]
pub fn summarize_value(i: usize, v: &Value) -> String {
    match v {
        Value::GpuTensor(h) => format!("in#{i}:GpuTensor shape={:?}", h.shape()),
        Value::Tensor(t) => format!("in#{i}:Tensor shape={:?}", t.shape()),
        Value::Num(n) => format!("in#{i}:Num({n:.6})"),
        Value::Int(n) => format!("in#{i}:Int({n:?})"),
        Value::Bool(b) => format!("in#{i}:Bool({})", if *b { 1 } else { 0 }),
        Value::String(s) => format!("in#{i}:String({})", s.display()),
        _ => format!("in#{i}:{}", value_kind(v)),
    }
}

#[inline]
fn is_scalarish_runtime_value(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => true,
        Value::Tensor(tensor) => is_scalar_shape(tensor.shape()),
        Value::ComplexTensor(tensor) => is_scalar_shape(tensor.shape()),
        Value::LogicalArray(array) => is_scalar_shape(&array.shape),
        Value::GpuTensor(handle) => is_scalar_shape(handle.shape()),
        Value::CharArray(array) => array.data().len() == 1,
        _ => false,
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
            Instr::StochasticEvolution { .. }
                | Instr::StoreIndex(_)
                | Instr::StoreIndexDelete(_)
                | Instr::StoreSlice(_, _, _, _)
                | Instr::StoreSliceDelete(_, _, _, _)
                | Instr::StoreIndexCell { .. }
                | Instr::StoreIndexCellDelete { .. }
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

#[path = "fusion/final_only_stores.rs"]
mod final_only_stores;
pub use final_only_stores::final_only_span_requires_ordinary_stores;
pub(crate) use final_only_stores::final_only_span_requires_ordinary_stores_with_private;

pub struct StackSliceGuard<'a> {
    stack: *mut crate::interpreter::stack::InterpreterStack,
    slice: Option<runmat_value::OwnedOutputListValues>,
    _marker: std::marker::PhantomData<&'a mut crate::interpreter::stack::InterpreterStack>,
}

impl<'a> StackSliceGuard<'a> {
    pub fn new(
        stack: &'a mut crate::interpreter::stack::InterpreterStack,
        slice_start: usize,
    ) -> Result<Self, RuntimeError> {
        let slice = stack
            .try_extract_output(slice_start)?
            .try_into_owned()
            .map_err(|_| {
                mex(
                    "SharedOutputStorage",
                    "fusion operands unexpectedly have shared storage",
                )
            })?;
        Ok(Self {
            stack,
            slice: Some(slice),
            _marker: std::marker::PhantomData,
        })
    }

    pub fn slice(&self) -> &[Value] {
        self.slice.as_ref().map_or(&[], |slice| slice.as_slice())
    }

    /// Validate every destination before changing the selected operand slice.
    /// The readback owner already checked each live view and source owner.
    pub fn replace_host_operands(
        &mut self,
        replacements: Vec<(usize, Value)>,
    ) -> Result<(), RuntimeError> {
        let Some(slice) = self.slice.as_mut().map(|slice| slice.as_mut_slice()) else {
            return Err(mex(
                "FusionHostReadback",
                "fusion host stack slice was consumed",
            ));
        };
        let mut offsets = HashSet::new();
        if replacements.iter().any(|(offset, value)| {
            !offsets.insert(*offset)
                || !matches!(slice.get(*offset), Some(Value::GpuTensor(_)))
                || runmat_runtime::dispatcher::value_contains_gpu(value)
        }) || slice.iter().enumerate().any(|(offset, value)| {
            runmat_runtime::dispatcher::value_contains_gpu(value) && !offsets.contains(&offset)
        }) {
            return Err(mex(
                "FusionHostReadback",
                "selected host readback cannot publish its complete operand slice",
            ));
        }
        for (offset, value) in replacements {
            slice[offset] = value;
        }
        Ok(())
    }

    pub fn commit(mut self) {
        self.slice = None;
    }
}

impl Drop for StackSliceGuard<'_> {
    fn drop(&mut self) {
        if let Some(slice) = self.slice.take() {
            unsafe { (&mut *self.stack).restore_extracted(slice) }
        }
    }
}

pub(crate) struct GatheredFusionInputs<'a> {
    pub guard: StackSliceGuard<'a>,
    pub request: FusionExecutionRequest<'a>,
}

pub(crate) fn gather_fusion_inputs<'a>(
    plan: &'a runmat_accelerate::FusionGroupPlan,
    transport: &input_transport::VmFusionInputTransport,
    stack: &'a mut crate::interpreter::stack::InterpreterStack,
    vars: &InterpreterVars,
    context: &mut ExecutionContext,
    placement: Option<PlacementCorrelationId>,
) -> Result<Option<GatheredFusionInputs<'a>>, RuntimeError> {
    let prepare_started = Instant::now();
    let required = transport.required_stack_operands();
    if !transport.matches(plan) || stack.len() < required {
        return Ok(None);
    }
    // Read only the sealed input sources before taking ownership of operands.
    // An unavailable source declines without changing the VM stack.
    let slice_start = stack.len() - required;
    let Some(inputs) = transport.gather(&stack.as_slice()[slice_start..], vars, context) else {
        return Ok(None);
    };
    let stack_guard = StackSliceGuard::new(stack, slice_start)?;

    if log::log_enabled!(log::Level::Debug) {
        let summaries: Vec<String> = inputs
            .iter()
            .enumerate()
            .map(|(i, v)| summarize_value(i, v))
            .collect();
        log::debug!("fusion inputs runtime: [{}]", summaries.join(", "));
    }

    Ok(Some(GatheredFusionInputs {
        guard: stack_guard,
        request: FusionExecutionRequest {
            plan,
            inputs,
            placement,
            runtime: Some(context.runtime.clone()),
        },
    }))
    .inspect(|_| {
        if let Some(correlation) = placement {
            record_event(
                correlation,
                PlacementEventKind::Prepare,
                Some(PlacementVariant::ProviderFusion),
                Some("gather_runtime_inputs"),
                Some(duration_ns_saturating(prepare_started.elapsed())),
                None,
                &[],
            );
        }
    })
}

/// Direct writeback utility retained for allocation-lifetime tests. VM
/// publication uses the ordinary StoreVar/StoreLocal owner in the runner.
pub fn write_elementwise_materialized_stores(
    materialized_stores: Vec<(FusionStoreMaterialization, Value)>,
    vars: &mut InterpreterVars,
    context: &mut ExecutionContext,
) -> Result<(), RuntimeError> {
    for (store, value) in materialized_stores {
        match store.binding.kind {
            VarKind::Global => {
                let i = store.binding.index;
                crate::runtime::workspace::commit_workspace_slot_value(vars, i, None, value)?;
            }
            VarKind::Local => {
                if let Some(frame) = context.call_stack.last() {
                    let absolute = frame.locals_start + store.binding.index;
                    while context.locals.len() <= absolute {
                        context.locals.push(Value::Num(0.0));
                    }
                    context.locals[absolute] = value;
                } else {
                    let i = store.binding.index;
                    crate::runtime::workspace::commit_workspace_slot_value(vars, i, None, value)?;
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn fusion_execution_error(
    error: runmat_accelerate_api::ProviderExecutionError,
) -> RuntimeError {
    let fallback = mex("FusionExecutionFailed", &error.to_string());
    runmat_runtime::runtime_error::with_provider_execution_source(fallback, error)
}

/// Only an explicit unsupported, pre-effect provider result permits replay.
/// Hard failures and failures after a provider or workspace effect retain
/// their original error and must follow normal exception handling.
#[derive(Debug)]
pub enum FusionAttemptError {
    Deopt(RuntimeError),
    Terminal(RuntimeError),
}

// Resource admission and continuation control failures are never an explicit
// pre-effect provider Unsupported receipt. Preserve them as terminal failures,
// including when the selected continuation never constructs its child.
impl From<RuntimeError> for FusionAttemptError {
    fn from(error: RuntimeError) -> Self {
        Self::Terminal(error)
    }
}

pub struct FusionVmResult {
    pub final_value: Value,
    pub ordered_stores: Vec<(FusionStoreMaterialization, Value)>,
    pub completion: Option<FusionProviderCompletion>,
}

fn without_intermediate_stores(
    outcome: FusionExecutionOutcome<Value>,
) -> FusionExecutionOutcome<FusionVmResult> {
    match outcome {
        FusionExecutionOutcome::Executed(final_value) => {
            FusionExecutionOutcome::Executed(FusionVmResult {
                final_value,
                ordered_stores: Vec::new(),
                completion: None,
            })
        }
        FusionExecutionOutcome::SharedRuntime(observation) => {
            FusionExecutionOutcome::SharedRuntime(observation)
        }
    }
}

fn classify_provider_error(
    error: runmat_accelerate_api::ProviderExecutionError,
) -> FusionAttemptError {
    let retryable = error.is_unsupported();
    let error = fusion_execution_error(error);
    if retryable {
        FusionAttemptError::Deopt(error)
    } else {
        FusionAttemptError::Terminal(error)
    }
}

fn finish_provider_execution(
    outcome: runmat_accelerate_api::ProviderExecutionResult<FusionExecutionOutcome<Value>>,
    stack_guard: StackSliceGuard<'_>,
) -> Result<FusionExecutionOutcome<Value>, FusionAttemptError> {
    match outcome {
        Ok(FusionExecutionOutcome::Executed(result)) => {
            stack_guard.commit();
            Ok(FusionExecutionOutcome::Executed(result))
        }
        Ok(FusionExecutionOutcome::SharedRuntime(observation)) => {
            Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
        Err(error) => {
            let classified = classify_provider_error(error);
            if matches!(&classified, FusionAttemptError::Terminal(_)) {
                stack_guard.commit();
            }
            Err(classified)
        }
    }
}

#[cfg(test)]
mod error_tests;

pub async fn execute_fusion_elementwise(
    request: FusionExecutionRequest<'_>,
    stack_guard: StackSliceGuard<'_>,
) -> Result<FusionExecutionOutcome<FusionVmResult>, FusionAttemptError> {
    execute_fusion_elementwise_vm(request, stack_guard, None, None, None).await
}

async fn execute_fusion_elementwise_vm(
    request: FusionExecutionRequest<'_>,
    mut stack_guard: StackSliceGuard<'_>,
    right: Option<runmat_runtime::context::VmFusionCompletionRight>,
    decline: Option<runmat_runtime::context::VmFusionDeclineRight>,
    input_uses: Option<&runmat_accelerate::fusion_exec::VmHostInputUses>,
) -> Result<FusionExecutionOutcome<FusionVmResult>, FusionAttemptError> {
    let admission = decline
        .as_ref()
        .map(|_| elementwise_floating_admission(&request))
        .transpose();
    let execution = match admission {
        Err(error) => Err(error),
        Ok(Some(ElementwiseFloatingAdmission::NativeIntegerIneligible)) => {
            runmat_accelerate::fusion_exec::execute_elementwise_placed_vm_decline(
                request,
                decline.expect("integer admission requires the inspected VM decline right"),
            )
            .await
        }
        Ok(_) => {
            if let Some(right) = right {
                if let Some(uses) = input_uses {
                    runmat_accelerate::fusion_exec::execute_elementwise_placed_vm_inputs(
                        request, right, uses,
                    )
                    .await
                } else {
                    runmat_accelerate::fusion_exec::execute_elementwise_placed_vm(request, right)
                        .await
                }
            } else if let Some(decline) = decline {
                runmat_accelerate::fusion_exec::execute_elementwise_placed_vm_decline(
                    request, decline,
                )
                .await
            } else {
                execute_elementwise_placed(request)
                    .await
                    .map(|outcome| match outcome {
                        FusionExecutionOutcome::Executed(result) => {
                            FusionExecutionOutcome::Executed(
                                runmat_accelerate::fusion_exec::ElementwiseVmExecutionResult {
                                    result,
                                    completion: None,
                                },
                            )
                        }
                        FusionExecutionOutcome::SharedRuntime(observation) => {
                            FusionExecutionOutcome::SharedRuntime(observation)
                        }
                    })
            }
        }
    };
    match execution {
        Ok(FusionExecutionOutcome::Executed(result)) => {
            stack_guard.commit();
            Ok(FusionExecutionOutcome::Executed(FusionVmResult {
                final_value: result.result.final_value,
                ordered_stores: result.result.materialized_stores,
                completion: result.completion,
            }))
        }
        Ok(FusionExecutionOutcome::SharedRuntime(mut observation)) => {
            match observation
                .prepare_vm_host_inputs(stack_guard.slice())
                .await
            {
                Ok(Some(replacements)) => stack_guard
                    .replace_host_operands(replacements)
                    .map_err(FusionAttemptError::Terminal)?,
                Ok(None) => {}
                Err(CpuReadbackFailure::BeforeEffect(error)) => {
                    return Err(FusionAttemptError::Deopt(error))
                }
                Err(CpuReadbackFailure::AfterEffect(error)) => {
                    return Err(FusionAttemptError::Terminal(error))
                }
            }
            Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
        Err(err) => {
            let classified = classify_provider_error(err);
            if matches!(&classified, FusionAttemptError::Terminal(_)) {
                // Ordinary failing arithmetic consumes its operands before
                // entering the catch handler. Deopt alone needs them back.
                stack_guard.commit();
            }
            Err(classified)
        }
    }
}

pub async fn execute_fusion_special_kind(
    kind: FusionKind,
    plan_inputs: &[runmat_accelerate::graph::ValueId],
    request: FusionExecutionRequest<'_>,
    stack_guard: StackSliceGuard<'_>,
    decline: Option<runmat_runtime::context::VmFusionDeclineRight>,
) -> Result<FusionExecutionOutcome<Value>, FusionAttemptError> {
    match kind {
        FusionKind::CenteredGram => {
            // Structural recognition does not prove equivalent classes,
            // accumulation, operand order or shape. Leave all operands and
            // stores untouched for the original bytecode, including GPU ones.
            Ok(FusionExecutionOutcome::implementation_unavailable())
        }
        FusionKind::PowerStepNormalize => finish_provider_execution(
            execute_power_step_normalize_placed(request).await,
            stack_guard,
        ),
        FusionKind::ExplainedVariance => {
            log::debug!("explained variance plan inputs {:?}", plan_inputs);
            let outcome = execute_explained_variance_placed(request).await;
            if let Err(error) = &outcome {
                log::debug!("explained variance fusion execution: {}", error);
            }
            finish_provider_execution(outcome, stack_guard)
        }
        FusionKind::MatmulEpilogue => {
            finish_provider_execution(execute_matmul_epilogue_placed(request).await, stack_guard)
        }
        FusionKind::ImageNormalize => finish_provider_execution(
            runmat_accelerate::fusion_exec::execute_image_normalize_vm(request, decline.as_ref())
                .await,
            stack_guard,
        ),
        _ => Err(FusionAttemptError::Terminal(mex(
            "FusionUnsupportedKind",
            "fusion: unsupported fusion kind",
        ))),
    }
}

#[path = "fusion/reduction_geometry.rs"]
mod reduction_geometry;
pub use reduction_geometry::{resolve_reduction_geometry, ReductionGeometry};

pub async fn execute_fusion_reduction(
    plan: &runmat_accelerate::FusionGroupPlan,
    request: FusionExecutionRequest<'_>,
    stack_guard: StackSliceGuard<'_>,
    right: Option<runmat_runtime::context::VmFusionCompletionRight>,
) -> Result<FusionExecutionOutcome<FusionVmResult>, FusionAttemptError> {
    execute_fusion_reduction_vm(plan, request, stack_guard, right, None).await
}

async fn execute_fusion_reduction_vm(
    plan: &runmat_accelerate::FusionGroupPlan,
    request: FusionExecutionRequest<'_>,
    mut stack_guard: StackSliceGuard<'_>,
    right: Option<runmat_runtime::context::VmFusionCompletionRight>,
    decline: Option<runmat_runtime::context::VmFusionDeclineRight>,
) -> Result<FusionExecutionOutcome<FusionVmResult>, FusionAttemptError> {
    let geom = match resolve_reduction_geometry(plan, &request) {
        Ok(geom) => geom,
        Err(error) => {
            // Geometry admission runs before provider work. Unsupported or
            // unresolved fused geometry must leave the original runtime call
            // responsible for its value, options, and owning diagnostics.
            return Err(FusionAttemptError::Deopt(error));
        }
    };
    // Specialize only this invocation. The generator must use the same axis as
    // geometry; never overwrite a cached dimension value used by another op.
    let resolved_plan = match reduction_geometry::bind_axis(plan, &geom) {
        Ok(resolved) => resolved,
        Err(error) => {
            return Err(FusionAttemptError::Terminal(error));
        }
    };
    let request = FusionExecutionRequest {
        plan: &resolved_plan,
        ..request
    };
    if right.is_some() || decline.is_some() {
        let execution = if let Some(right) = right {
            execute_reduction_with_shape_placed_async_vm(
                request,
                geom.reduce_len,
                geom.num_slices,
                &geom.output_shape,
                256u32,
                right,
            )
            .await
        } else {
            runmat_accelerate::fusion_exec::execute_reduction_with_shape_placed_async_vm_decline(
                request,
                geom.reduce_len,
                geom.num_slices,
                &geom.output_shape,
                256u32,
                decline.expect("checked VM decline right"),
            )
            .await
        };
        match execution {
            Ok(FusionExecutionOutcome::Executed(result)) => {
                stack_guard.commit();
                Ok(FusionExecutionOutcome::Executed(FusionVmResult {
                    final_value: result.value,
                    ordered_stores: Vec::new(),
                    completion: result.completion,
                }))
            }
            Ok(FusionExecutionOutcome::SharedRuntime(mut observation)) => {
                match observation.prepare_vm_host_slice(stack_guard.slice()).await {
                    Ok(Some(replacements)) => stack_guard
                        .replace_host_operands(replacements)
                        .map_err(FusionAttemptError::Terminal)?,
                    Ok(None) => {}
                    Err(CpuReadbackFailure::BeforeEffect(error)) => {
                        return Err(FusionAttemptError::Deopt(error));
                    }
                    Err(CpuReadbackFailure::AfterEffect(error)) => {
                        return Err(FusionAttemptError::Terminal(error));
                    }
                }
                Ok(FusionExecutionOutcome::SharedRuntime(observation))
            }
            Err(error) => {
                let classified = classify_provider_error(error);
                if matches!(&classified, FusionAttemptError::Terminal(_)) {
                    stack_guard.commit();
                }
                Err(classified)
            }
        }
    } else {
        finish_provider_execution(
            execute_reduction_with_shape_placed_async(
                request,
                geom.reduce_len,
                geom.num_slices,
                &geom.output_shape,
                256u32,
            )
            .await,
            stack_guard,
        )
        .map(without_intermediate_stores)
    }
}

pub async fn try_execute_fusion_group(
    plan: &runmat_accelerate::FusionGroupPlan,
    graph: &runmat_accelerate::AccelGraph,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    vars: &InterpreterVars,
    context: &mut ExecutionContext,
    placement: Option<PlacementCorrelationId>,
    right: Option<runmat_runtime::context::VmFusionCompletionRight>,
) -> Result<FusionExecutionOutcome<FusionVmResult>, FusionAttemptError> {
    // This compatibility entry point has no instruction transcript. It can
    // transport evaluated operands/literals, but never skip a semantic load.
    let transport = input_transport::VmFusionInputTransport::project(plan, graph, &[], &[]);
    try_execute_fusion_group_with_inputs(
        plan,
        graph,
        stack,
        vars,
        context,
        placement,
        VmFusionInvocation {
            right,
            decline: None,
            input_uses: None,
            transport: transport.as_ref(),
        },
    )
    .await
}

pub(crate) struct VmFusionInvocation<'a> {
    pub right: Option<runmat_runtime::context::VmFusionCompletionRight>,
    pub decline: Option<runmat_runtime::context::VmFusionDeclineRight>,
    pub input_uses: Option<&'a runmat_accelerate::fusion_exec::VmHostInputUses>,
    pub transport: Option<&'a input_transport::VmFusionInputTransport>,
}

pub(crate) async fn try_execute_fusion_group_with_inputs(
    plan: &runmat_accelerate::FusionGroupPlan,
    _graph: &runmat_accelerate::AccelGraph,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    vars: &InterpreterVars,
    context: &mut ExecutionContext,
    placement: Option<PlacementCorrelationId>,
    invocation: VmFusionInvocation<'_>,
) -> Result<FusionExecutionOutcome<FusionVmResult>, FusionAttemptError> {
    let VmFusionInvocation {
        right,
        decline,
        input_uses,
        transport,
    } = invocation;
    let Some(transport) = transport else {
        return Err(FusionAttemptError::Deopt(mex(
            "FusionInputTransport",
            "fusion input transport has not been installed",
        )));
    };
    let GatheredFusionInputs {
        guard: stack_guard,
        request,
    } = gather_fusion_inputs(plan, transport, stack, vars, context, placement)
        .map_err(FusionAttemptError::Terminal)?
        .ok_or_else(|| {
            FusionAttemptError::Deopt(mex(
                "FusionInputTransport",
                "fusion input transport is unavailable before execution",
            ))
        })?;
    if plan.group.kind.is_elementwise()
        && !request.inputs.is_empty()
        && request.inputs.iter().all(is_scalarish_runtime_value)
    {
        return Err(FusionAttemptError::Deopt(mex(
            "FusionScalarBypass",
            "fusion: bypass scalar-only elementwise group",
        )));
    }
    log::debug!(
        "dispatch fusion kind {:?}, supported {}",
        plan.group.kind,
        plan.kernel.supported
    );
    if plan.group.kind.is_elementwise() {
        execute_fusion_elementwise_vm(request, stack_guard, right, decline, input_uses).await
    } else if plan.group.kind.is_reduction() {
        execute_fusion_reduction_vm(plan, request, stack_guard, right, decline).await
    } else {
        execute_fusion_special_kind(
            plan.group.kind.clone(),
            &plan.inputs,
            request,
            stack_guard,
            decline,
        )
        .await
        .map(without_intermediate_stores)
    }
}

#[cfg(all(test, feature = "native-accel"))]
mod tests {
    use super::reduction_geometry::{
        fused_reduction_output_shape, validate_fused_reduction_domain,
    };
    use super::{write_elementwise_materialized_stores, StackSliceGuard};
    use crate::bytecode::program::ExecutionContext;
    use crate::interpreter::vars::InterpreterVars;
    use runmat_accelerate::fusion::FusionStoreMaterialization;
    use runmat_accelerate::fusion_residency;
    use runmat_accelerate::graph::VarBinding;
    use runmat_accelerate::VarKind;
    use runmat_value::Value;

    #[test]
    fn fused_reduction_domain_defers_multi_axis_but_admits_single_nd_axis() {
        use runmat_accelerate::ReductionAxes;

        assert!(validate_fused_reduction_domain(
            Some(&ReductionAxes::Explicit(vec![2, 3])),
            Some(&[2, 4, 5]),
            false,
        )
        .is_err());
        assert!(validate_fused_reduction_domain(
            Some(&ReductionAxes::Explicit(vec![2])),
            Some(&[2, 4, 5]),
            false,
        )
        .is_ok());
        assert!(
            validate_fused_reduction_domain(Some(&ReductionAxes::All), Some(&[2, 4, 5]), true,)
                .is_ok()
        );
        assert!(validate_fused_reduction_domain(
            Some(&ReductionAxes::Explicit(vec![2])),
            Some(&[2, 4]),
            false,
        )
        .is_ok());
    }

    #[test]
    fn fused_reduction_output_shape_retains_matlab_singleton_axes() {
        assert_eq!(
            fused_reduction_output_shape(Some(vec![3, 5]), 3, 5, 0, false).unwrap(),
            vec![1, 5]
        );
        assert_eq!(
            fused_reduction_output_shape(Some(vec![3, 5]), 3, 5, 1, false).unwrap(),
            vec![3, 1]
        );
        assert_eq!(
            fused_reduction_output_shape(Some(vec![2, 4, 5]), 2, 4, 0, true).unwrap(),
            vec![1, 1]
        );
    }

    #[test]
    fn fusion_writeback_preserves_shared_gpu_handles() {
        use futures::executor::block_on;
        use once_cell::sync::Lazy;
        use runmat_accelerate::simple_provider::InProcessProvider;
        use runmat_accelerate_api::{AccelProvider, HostTensorView, ThreadProviderGuard};

        static PROVIDER: Lazy<InProcessProvider> = Lazy::new(InProcessProvider::new);
        runmat_gc::gc_test_context(|| {
            let _provider = ThreadProviderGuard::set(Some(&*PROVIDER));
            runmat_accelerate::ensure_residency_hooks();
            let upload = |value| {
                PROVIDER
                    .upload(&HostTensorView {
                        data: &[value],
                        shape: &[1, 1],
                    })
                    .unwrap()
            };
            let shared = upload(7.0);
            let old_only = upload(9.0);
            fusion_residency::mark(&shared);
            fusion_residency::mark(&old_only);
            let stack = Vec::new();
            let mut vars = InterpreterVars::from_owned(vec![Value::Cell(
                runmat_value::CellArray::new_with_shape(
                    vec![
                        Value::GpuTensor(shared.clone()),
                        Value::GpuTensor(old_only.clone()),
                    ],
                    vec![1, 2],
                )
                .unwrap(),
            )]);
            let frame_roots = crate::runtime::gc::InterpretContext::new(&stack, &vars).unwrap();
            drop(frame_roots);
            let old_weak = old_only.downgrade();
            let shared_weak = shared.downgrade();
            let mut context = ExecutionContext::default();
            write_elementwise_materialized_stores(
                vec![(
                    FusionStoreMaterialization {
                        value_id: 1,
                        binding: VarBinding {
                            kind: VarKind::Global,
                            index: 0,
                        },
                    },
                    Value::GpuTensor(shared.clone()),
                )],
                &mut vars,
                &mut context,
            )
            .unwrap();
            // Both external handles are genuine owners; writeback cannot free either.
            assert!(block_on(PROVIDER.download(&shared)).is_ok());
            assert!(block_on(PROVIDER.download(&old_only)).is_ok());
            assert!(fusion_residency::is_resident(&old_only));
            assert_eq!(
                block_on(PROVIDER.download(&shared)).unwrap().data,
                vec![7.0]
            );
            drop(old_only);
            runmat_gc::gc_collect_major().unwrap();
            assert!(old_weak.acquire().is_err());
            assert!(fusion_residency::is_resident(&shared));
            drop(shared);
            assert!(
                shared_weak.acquire().is_ok(),
                "published variable owns the output"
            );
            drop(vars);
            assert!(shared_weak.acquire().is_err());
        });
    }

    #[test]
    fn host_readback_slice_publication_is_atomic_and_keeps_source_handles() {
        use futures::executor::block_on;
        use once_cell::sync::Lazy;
        use runmat_accelerate::simple_provider::InProcessProvider;
        use runmat_accelerate_api::{AccelProvider, HostTensorView, ThreadProviderGuard};

        static PROVIDER: Lazy<InProcessProvider> = Lazy::new(InProcessProvider::new);
        runmat_gc::gc_test_context(|| {
            let _provider = ThreadProviderGuard::set(Some(&*PROVIDER));
            let upload = |value| {
                PROVIDER
                    .upload(&HostTensorView {
                        data: &[value],
                        shape: &[1, 1],
                    })
                    .unwrap()
            };
            let left = upload(7.0);
            let right = upload(9.0);
            let public = [
                Value::GpuTensor(left.clone()),
                Value::GpuTensor(right.clone()),
            ];
            let mut stack = crate::interpreter::stack::InterpreterStack::from_unbounded(vec![
                Value::Num(3.0),
                Value::GpuTensor(left.clone()),
                Value::GpuTensor(right.clone()),
            ]);
            {
                let mut guard = StackSliceGuard::new(&mut stack, 1)
                    .expect("the unbounded test stack has a unique operand slice");
                assert!(guard
                    .replace_host_operands(vec![(0, Value::Num(7.0))])
                    .is_err());
                assert_eq!(guard.slice().len(), 2);
                assert!(guard
                    .slice()
                    .iter()
                    .all(|value| matches!(value, Value::GpuTensor(_))));
                assert!(guard
                    .replace_host_operands(vec![(0, Value::Num(7.0)), (0, Value::Num(9.0))])
                    .is_err());
                assert!(guard
                    .slice()
                    .iter()
                    .all(|value| matches!(value, Value::GpuTensor(_))));
                assert!(guard
                    .replace_host_operands(vec![
                        (0, Value::Num(7.0)),
                        (1, Value::GpuTensor(right.clone()))
                    ])
                    .is_err());
                assert!(guard
                    .slice()
                    .iter()
                    .all(|value| matches!(value, Value::GpuTensor(_))));
                guard
                    .replace_host_operands(vec![(0, Value::Num(7.0)), (1, Value::Num(9.0))])
                    .unwrap();
                assert_eq!(guard.slice(), &[Value::Num(7.0), Value::Num(9.0)]);
            }
            assert_eq!(
                stack.as_slice(),
                &[Value::Num(3.0), Value::Num(7.0), Value::Num(9.0)]
            );
            assert!(
                matches!(&public[0], Value::GpuTensor(handle) if handle.view_identity() == left.view_identity())
            );
            assert!(
                matches!(&public[1], Value::GpuTensor(handle) if handle.view_identity() == right.view_identity())
            );
            assert_eq!(block_on(PROVIDER.download(&left)).unwrap().data, vec![7.0]);
            assert_eq!(block_on(PROVIDER.download(&right)).unwrap().data, vec![9.0]);
        });
    }

    #[test]
    fn fusion_stack_transaction_restores_before_commit_and_never_replays_after_commit() {
        let mut failed = crate::interpreter::stack::InterpreterStack::from_unbounded(vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
        ]);
        {
            let guard = StackSliceGuard::new(&mut failed, 1)
                .expect("the unbounded test stack has a unique operand slice");
            assert_eq!(guard.slice(), &[Value::Num(2.0), Value::Num(3.0)]);
            // A provider/placement failure drops the guard without publishing
            // outputs; the original VM operands are restored exactly once.
        }
        assert_eq!(
            failed.as_slice(),
            &[Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)]
        );

        let mut completed = crate::interpreter::stack::InterpreterStack::from_unbounded(vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
        ]);
        {
            let guard = StackSliceGuard::new(&mut completed, 1)
                .expect("the unbounded test stack has a unique operand slice");
            guard.commit();
        }
        assert_eq!(completed.as_slice(), &[Value::Num(1.0)]);
    }
}
