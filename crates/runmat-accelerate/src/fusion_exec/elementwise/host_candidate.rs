//! Host implementation admission does not borrow a GPU shader proof. This
//! sibling still uses the same candidate, resource and VM completion owners.
use super::*;

pub(super) fn unfused_provider_required(request: &FusionExecutionRequest<'_>) -> bool {
    request
        .inputs
        .iter()
        .chain(unbound_constant_values(request.plan))
        .any(|value| matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle)))
}

pub(super) fn without_shader(
    request: &FusionExecutionRequest<'_>,
    output_ids: &[ValueId],
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    right: Option<&runmat_runtime::context::VmFusionCompletionRight>,
    uses: Option<&crate::fusion_exec::VmHostInputUses>,
    continuation: &mut Option<ElementwiseHostContinuation>,
) -> ExecutionResult<FusionExecutionOutcome<(HashMap<ValueId, Value>, FusionPlacementSelection)>> {
    let unavailable = || FusionExecutionOutcome::SharedRuntime(FusionCpuObservation::new(None));
    // A fused shader was unavailable before any physical effect. Explicit
    // resident input requires its provider, so a whole-group CPU candidate is
    // illegal. Decline only this optimization and let the VM execute the
    // original operations in order; do not prepare host readbacks or ask the
    // local selector to manufacture a no-route failure for the fused group.
    if unfused_provider_required(request) {
        return Ok(unavailable());
    }
    let Some((right, uses)) = right.zip(uses) else {
        return Ok(unavailable());
    };
    let Some(operations) = bindings.host_operations(request.runtime.as_ref()) else {
        return Ok(unavailable());
    };
    let Some(admission) = host_readback::admit_host(request, bindings, provider, uses) else {
        return Ok(unavailable());
    };
    let outputs = output_ids
        .iter()
        .map(|&id| {
            let shape = admission
                .program
                .fact(id)
                .ok()?
                .shape
                .known_dims()?
                .into_iter()
                .collect::<Option<Vec<_>>>()?;
            let precision = match admission.program.floating_class(id).ok()? {
                runmat_types::NumericClass::Single => ProviderPrecision::F32,
                runmat_types::NumericClass::Double => ProviderPrecision::F64,
                _ => return None,
            };
            Some(dense_output_representation(precision, &shape))
        })
        .collect::<Option<Vec<_>>>();
    let Some(outputs) = outputs else {
        return Ok(unavailable());
    };
    let Ok(work) = admission.program.host_scalar_work_units(request.plan) else {
        return Ok(unavailable());
    };
    let identity = runmat_execution::Digest::sha256_serialized(&(
        "elementwise-host-source-without-shader-v1",
        admission.program.cache_identity()?,
        work,
    ))
    .map_err(anyhow::Error::from)?;
    let selected = super::super::select_fusion_provider_with_implementation(
        request,
        bindings,
        provider,
        FusionProviderContract {
            elementwise_abi: None,
            input_indices: None,
            outputs,
            semantic_identity: Some(identity),
            scalar_work_units: Some(work),
            host_scalar_work_units: Some(work),
            matmul: None,
            preparation_program: None,
            shader_requirements: None,
        },
        Some(super::super::VmFusionSelectionRight::Completion(right)),
        super::super::host_readback::CpuHostPlacement::Readbacks(&admission.resources),
        crate::placement::LocalProviderAdmission::ImplementationUnavailable,
    )?;
    match selected {
        FusionExecutionOutcome::SharedRuntime(observation) => {
            *continuation = Some(ElementwiseHostContinuation {
                operations: Some(operations),
                readback: Some(admission.plan),
            });
            Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
        FusionExecutionOutcome::Executed(_) => {
            Err(anyhow!("unavailable shader was selected for execution").into())
        }
    }
}
