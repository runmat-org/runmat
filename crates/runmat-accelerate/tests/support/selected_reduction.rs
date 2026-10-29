//! Same-semantic actual-provider qualification: wider accumulator capability
//! is not interchangeable with signal/native-precision accumulation.
use runmat_accelerate::fusion_exec::{
    execute_reduction_with_shape_placed_async, FusionExecutionOutcome, FusionExecutionRequest,
};
use runmat_accelerate_api::{ProviderDoubleSumPreflight, ProviderExecutionResult};
use runmat_value::Value;

pub async fn execute(
    request: FusionExecutionRequest<'_>,
    reduce_len: usize,
    slices: usize,
    shape: &[usize],
    workgroup_size: u32,
) -> ProviderExecutionResult<Option<Value>> {
    let source = request
        .inputs
        .iter()
        .find_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
        .expect("actual resident reduction source required");
    let owner = runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle(source)
        .expect("source's actual provider owner");
    // These fixtures select the default canonical Sum/Mean declarations, both
    // of which require Double accumulation. Probe that actual owner's existing
    // ABI, not an adapter/backend name or a generic SharedRuntime diagnostic.
    let capability = owner.preflight_double_sum(source);
    let outcome = execute_reduction_with_shape_placed_async(
        request,
        reduce_len,
        slices,
        shape,
        workgroup_size,
    )
    .await?;
    match (capability, outcome) {
        (ProviderDoubleSumPreflight::Unsupported, FusionExecutionOutcome::SharedRuntime(_)) => {
            eprintln!("QUALIFICATION: negative canonical Double-accumulator admission");
            Ok(None)
        }
        (ProviderDoubleSumPreflight::Unknown, FusionExecutionOutcome::Executed(value)) => {
            eprintln!(
                "QUALIFICATION: actual capable canonical Double-accumulator numerical positive"
            );
            Ok(Some(value))
        }
        (ProviderDoubleSumPreflight::Unsupported, FusionExecutionOutcome::Executed(_)) => {
            panic!("canonical Double accumulation executed on unsupported provider")
        }
        (ProviderDoubleSumPreflight::Unknown, FusionExecutionOutcome::SharedRuntime(_)) => {
            panic!("capable canonical reduction must execute, not accept generic SharedRuntime")
        }
    }
}
