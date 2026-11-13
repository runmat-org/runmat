//! Host implementation admission does not borrow a GPU shader proof. This
//! sibling uses the same candidate/resource owners and the actual VM or Native
//! completion authority; neither source borrows the other's installation right.
use super::*;
use runmat_runtime::numeric_region::service::{
    NativeNumericRegionCpuCandidates, NativeNumericRegionProgramOwner, NativeNumericRegionRight,
};

#[derive(Clone, Copy)]
pub(super) enum Source<'a> {
    Vm {
        right: &'a runmat_runtime::context::VmFusionCompletionRight,
        uses: &'a crate::fusion_exec::VmHostInputUses,
    },
    Native {
        cpu: Option<&'a NativeNumericRegionCpuCandidates>,
        right: &'a NativeNumericRegionRight,
        program: &'a NativeNumericRegionProgramOwner,
    },
}

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
    source: Option<Source<'_>>,
    preparation_receipt: Option<runmat_runtime::context::host_allocation::HostAllocationReceipt>,
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
    let Some(source) = source else {
        return Ok(unavailable());
    };
    let operations = match source {
        Source::Vm { .. } => match bindings.host_operations(request.runtime.as_ref()) {
            Some(operations) => Some(operations),
            None => return Ok(unavailable()),
        },
        Source::Native { .. } => None,
    };
    let admission = match source {
        Source::Vm { uses, .. } => {
            match host_readback::admit_host(request, bindings, provider, uses) {
                Some(admission) => Some(admission),
                None => return Ok(unavailable()),
            }
        }
        Source::Native { program, right, .. } => bindings
            .has_host_operations(request.runtime.as_ref())
            .then(|| host_readback::admit_native(request, bindings, provider, program, right))
            .flatten(),
    };
    // A vectorized Native candidate owns its own execution allocations. It
    // does not require a canonical-source-host readback admission, but shape
    // and work still come from the same original host facts, never a shader.
    let fallback_program = admission
        .is_none()
        .then(|| bindings.host_facts(request, provider.precision()))
        .flatten();
    let Some(program) = admission
        .as_ref()
        .map(|admission| &admission.program)
        .or(fallback_program.as_ref())
    else {
        return Ok(unavailable());
    };
    let outputs = output_ids
        .iter()
        .map(|&id| {
            let shape = program
                .fact(id)
                .ok()?
                .shape
                .known_dims()?
                .into_iter()
                .collect::<Option<Vec<_>>>()?;
            let precision = match program.floating_class(id).ok()? {
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
    let Ok(work) = program.host_scalar_work_units(request.plan) else {
        return Ok(unavailable());
    };
    let identity = runmat_execution::Digest::sha256_serialized(&(
        "elementwise-host-source-without-shader-v1",
        program.cache_identity()?,
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
        match source {
            Source::Vm { right, .. } => {
                Some(super::super::VmFusionSelectionRight::Completion(right))
            }
            Source::Native { .. } => None,
        },
        match source {
            Source::Vm { .. } => super::super::host_readback::CpuHostPlacement::Readbacks(
                &admission
                    .as_ref()
                    .expect("VM host admission checked above")
                    .resources,
            ),
            Source::Native { cpu, right, .. } => {
                super::super::host_readback::CpuHostPlacement::Native {
                    cpu,
                    right,
                    source_host: admission.as_ref().map(|admission| &admission.resources),
                }
            }
        },
        crate::placement::LocalProviderAdmission::ImplementationUnavailable,
    )?;
    // Only the retained source-host admission can escape with preparation
    // funding. The alternate fact graph is no longer needed after selection.
    drop(fallback_program);
    match selected {
        FusionExecutionOutcome::SharedRuntime(mut observation) => {
            if matches!(source, Source::Native { .. }) {
                let runtime = request
                    .runtime
                    .as_ref()
                    .ok_or_else(|| anyhow!("native fusion requires its owning runtime"))?;
                observation.attach_native_host_admission(
                    runtime,
                    admission,
                    preparation_receipt,
                )?;
                return Ok(FusionExecutionOutcome::SharedRuntime(observation));
            }
            let admission = admission.expect("VM host admission checked above");
            *continuation = Some(ElementwiseHostContinuation {
                operations,
                readback: (!admission.plan.sources.is_empty()).then_some(admission.plan),
                outputs: (!admission.outputs.is_empty()).then_some(admission.outputs),
            });
            Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
        FusionExecutionOutcome::Executed(_) => {
            Err(anyhow!("unavailable shader was selected for execution").into())
        }
    }
}
