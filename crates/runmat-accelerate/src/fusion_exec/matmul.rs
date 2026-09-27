//! Matrix-product fusion publishes only an admitted, ordered epilogue.
use super::outputs::FusionAllocations;
use super::*;

mod plan;
mod representation;

pub async fn execute_matmul_epilogue(request: FusionExecutionRequest<'_>) -> Result<Value> {
    match execute_matmul_epilogue_placed(request).await? {
        FusionExecutionOutcome::Executed(value) => Ok(value),
        FusionExecutionOutcome::SharedRuntime(_) => {
            Err(anyhow!("direct fusion request selected shared runtime"))
        }
    }
}

pub async fn execute_matmul_epilogue_placed(
    request: FusionExecutionRequest<'_>,
) -> Result<FusionExecutionOutcome<Value>> {
    crate::ensure_residency_hooks();
    let provider = provider().ok_or_else(|| anyhow!("no acceleration provider registered"))?;
    // Admission reads typed operations and represented operands only. No
    // upload, metadata mutation, allocation or provider query precedes it.
    let Some(admitted) = plan::AdmittedMatmul::new(&request, provider) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    let selection = match select_fusion_provider(&request, provider, &[admitted.shape.to_vec()])? {
        FusionExecutionOutcome::Executed(selection) => selection,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
    };
    let mut allocations = FusionAllocations::new(&request.inputs);
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
                let handle = allocations.upload(provider, &data, &tensor.shape)?;
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
    let output = {
        let _metadata = allocations.protect_metadata();
        selection.observer.queue_unmeasured();
        let output = provider
            .matmul_epilogue(
                &handles[&admitted.left],
                &handles[&admitted.right],
                &epilogue,
            )
            .await?;
        selection.observer.kernel_unmeasured();
        output
    };
    let mut output = allocations.admit(output, provider, &admitted.shape, provider.precision())?;
    mark_fusion_output(&mut output, &request.inputs);
    allocations.publish(&[&output]);
    selection.submitted();
    Ok(FusionExecutionOutcome::Executed(Value::GpuTensor(output)))
}
