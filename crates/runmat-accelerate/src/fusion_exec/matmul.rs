//! Matrix-product fusion publishes only an admitted, ordered epilogue.
use super::outputs::{reject_outputs, FusionOutputAdmission};
use super::*;

mod plan;
mod representation;
pub(super) mod workload;

pub async fn execute_matmul_epilogue(
    request: FusionExecutionRequest<'_>,
) -> ExecutionResult<Value> {
    match execute_matmul_epilogue_placed(request).await? {
        FusionExecutionOutcome::Executed(value) => Ok(value),
        FusionExecutionOutcome::SharedRuntime(_) => {
            Err(anyhow!("direct fusion request selected shared runtime").into())
        }
    }
}

pub async fn execute_matmul_epilogue_placed(
    request: FusionExecutionRequest<'_>,
) -> ExecutionResult<FusionExecutionOutcome<Value>> {
    let Some(bindings) = AdmittedBindings::new(&request) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    crate::ensure_residency_hooks();
    let provider = match admit_provider_availability(&request, provider())? {
        FusionExecutionOutcome::Executed(provider) => provider,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation));
        }
    };
    // Admission reads typed operations and represented operands only. No
    // upload, metadata mutation, allocation or placement selection precedes it.
    // Provider precision/ownership metadata is read to validate representation.
    let Some(admitted) = plan::AdmittedMatmul::new(&request, provider, &bindings) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    let contract = admitted
        .selection(&request, provider)
        .ok_or_else(|| anyhow!("matrix workload overflow"))?;
    let output_shape = if admitted.diagonal {
        vec![admitted.shape[0].min(admitted.shape[1]), 1]
    } else {
        admitted.shape.to_vec()
    };
    let selection = match select_fusion_provider_with_contract(
        &request,
        &bindings,
        provider,
        FusionProviderContract {
            elementwise_abi: None,
            input_indices: None,
            outputs: vec![dense_output_representation(
                provider.precision(),
                &output_shape,
            )],
            semantic_identity: None,
            scalar_work_units: None,
            host_scalar_work_units: None,
            matmul: Some(contract),
            preparation_program: None,
            shader_requirements: None,
        },
    )? {
        FusionExecutionOutcome::Executed(selection) => selection,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
    };
    let mut allocations = FusionOutputAdmission::new(&request.inputs);
    let mut handles = HashMap::new();
    for id in admitted.data_inputs() {
        if handles.contains_key(&id) {
            continue;
        }
        let value = admitted.value(&request, id).expect("admitted input");
        let handle = match value {
            Value::GpuTensor(handle) => handle.clone(),
            Value::Tensor(tensor) => {
                let started = Instant::now();
                let data = tensor.materialize_f64();
                let handle = allocations.upload(provider, &data, tensor.shape())?;
                selection
                    .observer
                    .upload(started, host_f64_bytes(data.len()));
                handle
            }
            _ => unreachable!("matrix admission accepts only represented data tensors"),
        };
        handles.insert(id, handle);
    }
    let epilogue = admitted.descriptor(&handles);
    let uploaded_inputs = admitted
        .data_inputs()
        .any(|id| matches!(admitted.value(&request, id), Some(Value::Tensor(_))));
    let output = {
        let _metadata: Vec<_> = handles
            .values()
            .map(runmat_accelerate_api::ResidencyRestoreGuard::new)
            .collect();
        selection.observer.queue_unmeasured();
        let output = provider
            .matmul_epilogue(
                &handles[&admitted.left],
                &handles[&admitted.right],
                &epilogue,
            )
            .await
            .map_err(|error| {
                if uploaded_inputs {
                    error.with_unproven_effects()
                } else {
                    error
                }
            })?;
        selection.observer.kernel_unmeasured();
        output
    };
    // Provider-produced results cannot alias a borrowed operand or a temporary
    // upload owned by this request. Metadata restoration is not an alias proof.
    if handles
        .values()
        .any(|input| input.shares_allocation_with(&output))
    {
        return Err(reject_outputs(
            vec![output],
            "matmul provider returned an operand alias",
        ));
    }
    let mut output = allocations.admit(output, provider, &admitted.shape, provider.precision())?;
    if admitted.diagonal {
        let diagonal = {
            let _metadata = runmat_accelerate_api::ResidencyRestoreGuard::new(&output);
            selection.observer.queue_unmeasured();
            let diagonal = provider.diag_extract(&output, 0)?;
            selection.observer.kernel_unmeasured();
            diagonal
        };
        if output.shares_allocation_with(&diagonal)
            || handles
                .values()
                .any(|input| input.shares_allocation_with(&diagonal))
        {
            return Err(reject_outputs(
                vec![diagonal],
                "diagonal provider returned an operand alias",
            ));
        }
        output = allocations.admit(
            diagonal,
            provider,
            &[admitted.shape[0].min(admitted.shape[1]), 1],
            provider.precision(),
        )?;
    }
    mark_fusion_output(&mut output, &request.inputs)?;
    selection.submitted();
    Ok(FusionExecutionOutcome::Executed(Value::GpuTensor(output)))
}
