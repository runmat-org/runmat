//! Async reverse of an actually executed resident network output.
//! Snapshot only the immutable record pointer before awaiting a provider; a
//! `RefCell<Tape>` borrow must never span a future suspension.

use super::super::*;
use std::collections::HashMap;

#[cfg(test)]
mod tests;
mod graph;

pub(super) async fn backward(
    runtime: Option<&crate::context::RuntimeContext>,
    loss_node: usize,
    requested_nodes: &[usize],
) -> BuiltinResult<HashMap<usize, runmat_accelerate_api::GpuTensorHandle>> {
    admit_cancellation(runtime)?;
    let graph = graph::admit(loss_node, requested_nodes)?;
    let output = &graph.loss;
    if output
        .shape()
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        != Some(1)
    {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: loss must be scalar",
        ));
    }
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(output)
        .ok_or_else(|| deep_learning_error("dlgradient", "dlgradient: resident owner retired"))?;
    let seed = provider
        .ones_like(output)
        .map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                deep_learning_error("dlgradient", "dlgradient: resident seed allocation failed"),
                error.with_unproven_effects(),
            )
        })?
        .ok_or_else(|| {
            let mut error = deep_learning_error(
                "dlgradient",
                "dlgradient: resident seed constructor unavailable",
            );
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })?;
    use crate::builtins::common::gpu_helpers::{
        reject_gpu_output, unary_gpu_output_matches_shape, GpuOutputAliasPolicy,
        UnaryGpuOutputContract,
    };
    let seed = if unary_gpu_output_matches_shape(
        &seed,
        output,
        provider,
        output.shape(),
        UnaryGpuOutputContract {
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(output),
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    ) {
        seed
    } else {
        return Err(reject_gpu_output(
            seed,
            deep_learning_error("dlgradient", "dlgradient: invalid resident seed descriptor"),
        ));
    };
    let mut gradients = HashMap::new();
    gradients.insert(loss_node, seed);
    for (id, execution, plan) in graph.executions {
        let Some(seed) = gradients.get(&id).cloned() else {
            continue;
        };
        admit_cancellation(runtime)?;
        for (node, mut gradient) in
            super::super::network::reverse_resident_execution(&execution, &plan, seed, runtime)
                .await?
        {
            crate::builtins::common::gpu_helpers::propagate_output_provenance(
                &mut gradient,
                std::iter::once(&execution.trace.output),
            )?;
            if let Some(previous) = gradients.get(&node) {
                if previous.shape() != gradient.shape()
                    || runmat_accelerate_api::handle_precision(previous)
                        != runmat_accelerate_api::handle_precision(&gradient)
                {
                    return Err(deep_learning_error(
                        "dlgradient",
                        "dlgradient: resident gradient contribution geometry or class mismatch",
                    ));
                }
                let provider =
                    crate::builtins::common::gpu_helpers::exact_provider_for_binary_inputs(
                        previous, &gradient,
                    )
                    .map_err(|_| {
                        deep_learning_error(
                            "dlgradient",
                            "dlgradient: resident gradient owner mismatch",
                        )
                    })?;
                let combined = provider
                    .elem_add(previous, &gradient)
                    .await
                    .map_err(|error| {
                        crate::runtime_error::with_provider_execution_source(
                            deep_learning_error(
                                "dlgradient",
                                "dlgradient: resident gradient accumulation failed",
                            ),
                            error,
                        )
                    })?;
                admit_cancellation(runtime)?;
                let combined = super::super::super::model::gpu::validate_binary_forward_result(
                    combined,
                    previous,
                    &gradient,
                    provider,
                    gradient.shape(),
                    runmat_accelerate_api::handle_precision(&gradient).ok_or_else(|| {
                        deep_learning_error(
                            "dlgradient",
                            "dlgradient: resident gradient lost floating class",
                        )
                    })?,
                    "dlgradient",
                )?;
                gradients.insert(node, combined);
            } else {
                gradients.insert(node, gradient);
            }
        }
    }
    admit_cancellation(runtime)?;
    Ok(gradients)
}

pub(in crate::builtins::deep_learning::autodiff) fn admit_cancellation(
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    if runtime.map_or_else(crate::interrupt::is_cancelled, |runtime| {
        runtime.is_cancelled()
    }) {
        let namespace = runtime
            .map(crate::context::RuntimeContext::error_namespace)
            .unwrap_or_else(|| "RunMat".to_string());
        return Err(crate::runtime_error::build_runtime_error(
            "dlgradient: differentiation was cancelled",
        )
        .with_identifier(format!("{namespace}:Cancelled"))
        .build());
    }
    Ok(())
}
