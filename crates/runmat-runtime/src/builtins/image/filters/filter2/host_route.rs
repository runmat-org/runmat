//! Exact host-result route for filter2 forms that cannot use the provider hook.
//! The shared context service admits every resident readback and owns timing.

use super::*;
use crate::context::host_presentation::{
    combine_host_scratch, select_exact_host_sink_with_readbacks, ExactHostSink,
};
use runmat_execution::{Digest, ResourceBoundUnavailable, ResourceByteBound};

fn count(shape: &[usize]) -> Option<u64> {
    shape.iter().try_fold(1_u64, |total, &extent| {
        total.checked_mul(u64::try_from(extent).ok()?)
    })
}

fn f64_materialization_bound(value: &Value) -> ResourceByteBound {
    let bytes = filter2_value_shape(value)
        .and_then(|shape| count(&shape))
        .and_then(|count| count.checked_mul(8));
    bytes
        .map(ResourceByteBound::Known)
        .unwrap_or(ResourceByteBound::Unavailable(
            ResourceBoundUnavailable::NoScratchBound,
        ))
}

pub(super) async fn run(
    runtime: Option<&crate::context::RuntimeContext>,
    kernel: Value,
    image: Value,
    options: &ImfilterOptions,
) -> BuiltinResult<Value> {
    let kernel_shape = filter2_value_shape(&kernel).ok_or_else(|| {
        filter2_error_with_detail(&FILTER2_ERROR_INVALID_INPUT, "H must be numeric or logical")
    })?;
    let image_shape = filter2_value_shape(&image).ok_or_else(|| {
        filter2_error_with_detail(&FILTER2_ERROR_INVALID_INPUT, "X must be numeric or logical")
    })?;
    let sources = [&image, &kernel]
        .into_iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    for source in &sources {
        source.allocation_lease().validate_live().map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                filter2_error_with_detail(
                    &FILTER2_ERROR_INVALID_INPUT,
                    "input allocation is unavailable",
                ),
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        })?;
    }
    let receipt = if let Some(source) = sources.first() {
        let (scratch, output) = placement::conditional_host_bounds(
            FILTER2_BUILTIN,
            &image_shape,
            &kernel_shape,
            options,
            value_is_single(&image),
            value_is_single(&kernel),
            false,
        );
        // The evaluator materializes A as f64 and a logical H is converted
        // to a floating Tensor. Include both even when a numeric view borrows.
        let scratch = combine_host_scratch(scratch, f64_materialization_bound(&image))?;
        let scratch = combine_host_scratch(scratch, f64_materialization_bound(&kernel))?;
        let domain_facts = Digest::sha256(
            serde_json::to_vec(&(
                image_shape,
                kernel_shape,
                options.mode,
                options.shape,
                options.padding,
                options.constant_value.to_bits(),
                value_is_single(&image),
                value_is_single(&kernel),
            ))
            .expect("typed filter2 host facts serialize"),
        );
        select_exact_host_sink_with_readbacks(
            runtime,
            ExactHostSink {
                operation: FILTER2_BUILTIN,
                variant: "filter2.resident_host_result.v1",
                source,
                domain_facts,
                output_bytes: output,
                host_compute_scratch: scratch,
            },
            &sources,
        )?
    } else {
        None
    };
    let result = async {
        let kernel = gather_filter2_value(kernel).await?;
        let image = gather_filter2_value(image).await?;
        filter2_host(kernel, image, options)
    }
    .await;
    if let Some(receipt) = receipt {
        receipt.finish(result.is_ok());
    }
    result
}
