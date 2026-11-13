use super::super::{
    admit_provider_availability, host_f64_bytes, mark_fusion_output, scalar_upload_dtype,
    select_fusion_provider_vm_decline, AdmittedBindings, FusionCpuObservation,
    FusionExecutionOutcome, FusionExecutionRequest, FusionPlacementObserver, FusionStageTimer,
    PreparedInput,
};
use super::reduction_arithmetic;
use crate::precision::ensure_provider_supports_dtype;
use anyhow::anyhow;
use runmat_accelerate_api::{
    provider, GpuTensorHandle, HostTensorView, ProviderExecutionError,
    ProviderExecutionResult as ExecutionResult, ProviderPrecision, ReductionFlavor,
};
use runmat_runtime::builtins::common::shape::normalize_scalar_shape;
use runmat_time::Instant;
use runmat_types::FusionReductionOperation;

mod contract;
use runmat_value::Value;

/// A failing provider hook may already have made an effect. Its Unsupported
/// cause cannot authorize replay of the skipped bytecode.
fn after_reduction_provider_effect(error: anyhow::Error) -> ProviderExecutionError {
    ProviderExecutionError::from(error).after_effect()
}

/// Legacy provider ABI has only Sum and Mean. Its operation is certified by
/// the selected reduction declaration, never by a graph name heuristic or an
/// absent planner flavor default. Extrema and unknown terminal bindings stay
/// on ordinary dispatch when no typed recipe was selected.
pub(super) fn selected_legacy_flavor(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
) -> Option<ReductionFlavor> {
    let output = request.plan.output?;
    let mut terminals = request
        .plan
        .operations
        .iter()
        .enumerate()
        .filter(|(_, op)| match op {
            crate::fusion::FusionOp::Primitive { output: result, .. }
            | crate::fusion::FusionOp::Builtin { output: result, .. } => *result == Some(output),
        });
    let (terminal, operation) = terminals.next()?;
    if terminals.next().is_some() {
        return None;
    }
    if !matches!(operation, crate::fusion::FusionOp::Builtin { .. }) {
        return None;
    }
    match bindings.reduction(terminal)?.operation {
        FusionReductionOperation::Sum => Some(ReductionFlavor::Sum),
        FusionReductionOperation::Mean => Some(ReductionFlavor::Mean),
        FusionReductionOperation::Minimum | FusionReductionOperation::Maximum => None,
    }
}

pub(super) struct LegacyReductionProofs<'a, 'request> {
    pub(super) mathematical: &'a super::super::ResolvedReductionProgram<'request>,
    pub(super) producer: Option<&'a reduction_arithmetic::PreparedReductionArithmetic>,
    pub(super) owner: &'a dyn runmat_accelerate_api::AccelProvider,
}

pub(super) fn execute_reduction_with_shape_core(
    request: &FusionExecutionRequest<'_>,
    proofs: LegacyReductionProofs<'_, '_>,
    reduce_len: usize,
    num_slices: usize,
    output_shape: &[usize],
    workgroup_size: u32,
    vm_decline: Option<&runmat_runtime::context::VmFusionDeclineRight>,
) -> ExecutionResult<FusionExecutionOutcome<Value>> {
    let LegacyReductionProofs {
        mathematical: program,
        producer: arithmetic,
        owner,
    } = proofs;
    let Some(bindings) = AdmittedBindings::new(request) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    let Some(flavor) = selected_legacy_flavor(request, &bindings) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    super::validation::request(request, reduce_len, num_slices, output_shape)?;
    let len = reduce_len * num_slices;
    let scalar_shape = {
        let constant_shape = request.plan.constant_shape(len);
        normalize_scalar_shape(&vec![1; constant_shape.len()])
    };
    let provider = match admit_provider_availability(request, provider())? {
        FusionExecutionOutcome::Executed(provider) => provider,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation));
        }
    };
    if !runmat_accelerate_api::same_provider_instance(owner, provider) {
        return Err(anyhow!("selected reduction provider changed before legacy admission").into());
    }
    let Some(proof) = contract::admit(
        request,
        program,
        flavor,
        provider.precision(),
        reduce_len,
        num_slices,
        output_shape,
    ) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    let flavor = proof.flavor();
    if let Some(arithmetic) = arithmetic {
        arithmetic.validate(
            request.plan,
            &request.inputs,
            provider.precision(),
            reduce_len,
            num_slices,
        )?;
    }
    let shader_started = Instant::now();
    let generated_shader = request
        .plan
        .prepare_reduction_shader(
            provider.precision(),
            flavor,
            arithmetic.and_then(reduction_arithmetic::PreparedReductionArithmetic::emission),
        )
        .ok_or_else(|| anyhow!("fusion: reduction WGSL generation failed"))?;
    FusionPlacementObserver::new(request.placement).shader_source_generation(shader_started);
    let selection = select_fusion_provider_vm_decline(
        request,
        &bindings,
        provider,
        &[output_shape.to_vec()],
        Some(generated_shader.requirements()),
        vm_decline,
    )?;
    let selection = match selection {
        FusionExecutionOutcome::Executed(selection) => selection,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
    };
    let observer = selection.observer;
    let mut timer = FusionStageTimer::new("reduction", request.plan.index, len);
    let mut prepared = Vec::with_capacity(request.inputs.len());
    let mut temp_scalars: Vec<Vec<f64>> = Vec::new();
    let scalar_dtype = scalar_upload_dtype(provider);
    for value in &request.inputs {
        match value {
            Value::GpuTensor(handle) => prepared.push(PreparedInput {
                handle: handle.clone(),
            }),
            Value::Tensor(t) => {
                if let Err(msg) = ensure_provider_supports_dtype(provider, t.numeric_dtype()) {
                    return Err(anyhow!(
                        "fusion: tensor input requires unsupported precision ({msg})"
                    )
                    .into());
                }
                let data = t.materialize_f64();
                let view = HostTensorView {
                    data: &data,
                    shape: t.shape(),
                };
                let started = Instant::now();
                let handle = provider
                    .upload(&view)
                    .map_err(after_reduction_provider_effect)?;
                observer.upload(started, host_f64_bytes(data.len()));
                prepared.push(PreparedInput { handle });
            }
            Value::Num(n) => {
                if let Err(msg) = ensure_provider_supports_dtype(provider, scalar_dtype) {
                    return Err(anyhow!(
                        "fusion: scalar input requires unsupported precision ({msg})"
                    )
                    .into());
                }
                let scalar = match provider.precision() {
                    ProviderPrecision::F32 => (*n as f32) as f64,
                    ProviderPrecision::F64 => *n,
                };
                temp_scalars.push(vec![scalar]);
                let data = temp_scalars.last().unwrap();
                let view = HostTensorView {
                    data,
                    shape: &scalar_shape,
                };
                let started = Instant::now();
                let handle = provider
                    .upload(&view)
                    .map_err(after_reduction_provider_effect)?;
                observer.upload(started, host_f64_bytes(data.len()));
                prepared.push(PreparedInput { handle });
            }
            Value::Int(_) => unreachable!("integer inputs rejected before fusion upload"),
            _ => return Err(anyhow!("fusion: unsupported value type").into()),
        }
    }
    timer.mark("prepare_inputs");

    let handles: Vec<GpuTensorHandle> = prepared.iter().map(|p| p.handle.clone()).collect();
    timer.mark("generate_wgsl");
    if std::env::var("RUNMAT_DEBUG_DUMP_FUSED_WGSL").is_ok() {
        println!(
            "---- fused reduction WGSL ----\n{}\n------------------------------",
            generated_shader.source()
        );
    }

    let mut wg = if workgroup_size == 0 {
        provider.default_reduction_workgroup_size()
    } else {
        workgroup_size
    };
    if let Ok(raw) = std::env::var("RUNMAT_FUSED_WG") {
        if let Ok(val) = raw.trim().parse::<u32>() {
            if val > 0 {
                let capped = val.min(provider.default_reduction_workgroup_size());
                wg = capped.max(1);
            }
        }
    }
    let started = Instant::now();
    let mut output = provider
        .fused_reduction(
            generated_shader.source(),
            generated_shader.requirements(),
            &handles,
            output_shape,
            reduce_len,
            num_slices,
            wg,
            flavor,
        )
        .map_err(after_reduction_provider_effect)?;
    observer.queue(started);
    observer.kernel_unmeasured();
    timer.mark("dispatch");
    mark_fusion_output(&mut output, &request.inputs)?;

    drop(handles);
    drop(prepared);
    timer.mark("cleanup");
    timer.finish();

    selection.submitted();
    Ok(FusionExecutionOutcome::Executed(Value::GpuTensor(output)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduction_provider_hook_failure_cannot_replay_typed_unsupported() {
        let typed = runmat_accelerate_api::unsupported_provider_operation(
            "injected unsupported fused reduction",
        );
        assert!(runmat_accelerate_api::ProviderExecutionError::from(
            runmat_accelerate_api::unsupported_provider_operation("pre-effect unsupported")
        )
        .is_unsupported());
        assert!(!after_reduction_provider_effect(typed).is_unsupported());
    }
}
