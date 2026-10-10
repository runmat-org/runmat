use super::*;

pub(super) async fn reduction_max(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    args: ReductionArgs,
    demand: super::extrema_provider::ExtremaDemand,
) -> BuiltinResult<super::extrema_provider::ExtremaResult> {
    match value {
        Value::GpuTensor(handle) => {
            if let Some(eval) = reduction_max_gpu(runtime, handle.clone(), &args, demand).await? {
                return Ok(eval);
            }
            // Fall back to host if GPU path is unavailable.
            let value = super::extrema_provider::ExtremaKind::Max
                .gather(&handle)
                .await?;
            reduction_max_host(value, &args).map(|eval| {
                super::extrema_provider::ExtremaResult::from_pair(demand, eval.values, eval.indices)
            })
        }
        other => reduction_max_host(other, &args).map(|eval| {
            super::extrema_provider::ExtremaResult::from_pair(demand, eval.values, eval.indices)
        }),
    }
}

mod provider;
use provider::reduction_max_gpu;

mod host;
use host::reduction_max_host;
pub(super) use host::{materialize_for_max, InputData};

mod selection;
pub(super) use selection::{should_replace_complex, should_replace_real};

pub(super) mod geometry;
