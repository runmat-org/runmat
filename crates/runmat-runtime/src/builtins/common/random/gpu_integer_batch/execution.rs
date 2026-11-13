//! A selected native batch continues through separately admitted, physically
//! acknowledged retry batches. Only completion publishes its numeric output.

use super::super::{gpu, gpu_effect::after_stream_commit};
use super::progress::validate_progress;
use super::selection::{batch_calibration, batch_query, SelectedInitialNativeBatch};
use super::{error, provider_error};
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::gpu_helpers::query_metadata;
use crate::context::host_allocation::{
    provider_registration_host_quote, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::provider_operation::{
    select_exact_provider_reused_output_operation, ExactProviderOperation,
};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::observation::ProviderObservationFailure;
use runmat_accelerate_api::random::{
    GpuBoundedIntegerDomain, GpuIntegerBatchRequest, GpuIntegerPending, GpuIntegerRequest,
    GpuRandomReservation, PROVIDER_GPU_INTEGER_BATCH_OPERATION,
};
use runmat_accelerate_api::{GpuTensorHandle, NumericElementType, ProviderQueryMetadataFootprint};
use runmat_execution::{Digest, ResourceByteBound};

pub(crate) async fn execute_selected(
    source: &GpuTensorHandle,
    domain: GpuBoundedIntegerDomain,
    shape: &[usize],
    class: NumericElementType,
    first: SelectedInitialNativeBatch,
) -> BuiltinResult<GpuTensorHandle> {
    let provider = gpu_helpers::exact_provider_for_handle(source)
        .ok_or_else(|| error("randi: selected provider owner retired"))?;
    source.allocation_lease().validate_live().map_err(|cause| {
        crate::runtime_error::with_provider_execution_source(
            error(format!("randi: prototype lease: {cause}")),
            runmat_accelerate_api::ProviderExecutionError::from_source(cause)
                .with_unproven_effects(),
        )
    })?;
    let count = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            .ok_or_else(|| error("randi: output extent overflow"))?
    };
    let snapshot = first.snapshot;
    let reservation = GpuRandomReservation {
        algorithm: snapshot.algorithm,
        seed: snapshot.seed,
        first: snapshot.position,
        count: u64::try_from(count).map_err(|_| error("randi: child range exceeds u64"))?,
        normal_transform: snapshot.normal_transform,
    };
    let integer = GpuIntegerRequest {
        shape,
        class,
        domain,
        reservation,
    };
    if integer.rejection_domain_32().is_none() {
        return Err(error("randi: selected native integer domain changed"));
    }
    let runtime = crate::context::legacy::active();
    let mut first = Some(first);
    let mut pending: Option<GpuIntegerPending> = None;
    let mut first_attempt = 0_u64;
    let mut cooperation = runtime
        .as_ref()
        .map(crate::context::RuntimeContext::cpu_cooperation);
    let result: BuiltinResult<GpuTensorHandle> = async {
        loop {
            if let Some(budget) = cooperation.as_mut() {
                budget.checkpoint(1).await.map_err(|cause| {
                    let failure = build_runtime_error(format!("randi: {cause}"))
                        .with_builtin("randi")
                        .with_source(cause)
                        .build();
                    if pending.is_some() {
                        after_stream_commit("randi", failure)
                    } else {
                        failure
                    }
                })?;
            }
            let attempts_per_item = 8;
            let batch = GpuIntegerBatchRequest {
                integer,
                first_attempt,
                attempts_per_item,
                pending: pending.as_ref(),
            };
            let proposed = provider.propose_gpu_integer_batch(batch).map_err(|cause| {
                let failure = provider_error(cause);
                if pending.is_some() {
                    after_stream_commit("randi", failure)
                } else {
                    failure
                }
            })?;
            let Some(physical) = proposed else {
                return Err(after_stream_commit(
                    "randi",
                    error("randi: provider lost its selected batch route"),
                ));
            };
            if first
                .as_ref()
                .is_some_and(|first| first.physical != physical)
            {
                return Err(error("randi: selected first-batch physical quote changed"));
            }
            let input_ranks = [shape.len(), shape.len()];
            let output_ranks = [shape.len(), shape.len()];
            let footprint = ProviderQueryMetadataFootprint {
                operation: PROVIDER_GPU_INTEGER_BATCH_OPERATION,
                input_ranks: if pending.is_some() { &input_ranks } else { &[] },
                output_ranks: &output_ranks,
                preparation_shape_ranks: &[],
                preparation_representation_slots: 0,
            };
            let query_bytes =
                query_metadata::requested_bytes("randi", &footprint).map_err(|cause| {
                    if pending.is_some() {
                        after_stream_commit("randi", cause)
                    } else {
                        cause
                    }
                })?;
            let _query_receipt =
                query_metadata::admit(runtime.as_ref(), query_bytes).map_err(|cause| {
                    let cause = query_metadata::admission_error("randi", runtime.as_ref(), cause);
                    if pending.is_some() {
                        after_stream_commit("randi", cause)
                    } else {
                        cause
                    }
                })?;
            let query = batch_query(
                shape,
                class,
                domain,
                reservation,
                first_attempt,
                pending.is_some(),
            )?;
            if !footprint.matches(&query) {
                return Err(error("randi: batch query metadata plan changed"));
            }
            let register = if pending.is_none() {
                provider_registration_host_quote(provider, shape.len(), 2, 1)
                    .map_err(|cause| error(format!("randi: batch metadata quote: {cause}")))?
            } else {
                None
            };
            let finite_host = runtime
                .as_ref()
                .and_then(|runtime| runtime.service_ports().parallel())
                .and_then(|port| port.placement_resources().memory_available_bytes)
                .is_some();
            if pending.is_none() && finite_host && register.is_none() {
                return Err(error(
                    "randi: finite batch lacks a provider registration quote",
                ));
            }
            let host_bytes = register
                .map_or(Some(0), |quote| {
                    quote.preparation_bytes.checked_add(quote.retained_bytes)
                })
                .map(|bytes| bytes.max(query_bytes))
                .ok_or_else(|| error("randi: batch host metadata overflow"))?;
            let readback_retained = HostAllocationPlan::retained_metadata_bytes(
                physical.readback_control_host_bytes,
            )
            .map_err(|cause| error(format!("randi: count readback control quote: {cause}")))?;
            let provider_host_bytes = readback_retained
                .checked_add(physical.pending_view_workspace_bytes)
                .ok_or_else(|| error("randi: provider host workspace overflow"))?;
            let facts = Digest::sha256_serialized(&(
                &query,
                source.view_identity(),
                pending.as_ref().map(|held| {
                    (
                        held.output.view_identity(),
                        held.accepted.view_identity(),
                        held.remaining,
                    )
                }),
                physical,
            ))
            .expect("typed batch facts serialize");
            let selected = if first.is_none() {
                Some(select_exact_provider_reused_output_operation(
                    runtime.as_ref(),
                    ExactProviderOperation {
                        provider,
                        query: &query,
                        semantic_operation: "randi",
                        variant: "random.gpu_child.integer.batch.v1",
                        domain_facts: facts,
                        calibration_facts: Some(batch_calibration(
                            &query,
                            reservation,
                            domain,
                            pending.as_ref().map(|held| held.remaining),
                            physical,
                        )),
                        host_scratch: ResourceByteBound::Known(host_bytes),
                        provider_host_scratch: ResourceByteBound::Known(provider_host_bytes),
                        host_cpu_millicores: 1_000,
                        already_live_bytes: pending.as_ref().map_or(0, |held| {
                            held.output
                                .physical()
                                .requested_bytes()
                                .saturating_add(held.accepted.physical().requested_bytes())
                        }),
                        readbacks: &[],
                        produced_readback: None,
                        internal_input_stages: None,
                        output_queue_slots: Some(physical.queue_submissions),
                        conditional_host_fallback: None,
                        conditional_host_scratch: ResourceByteBound::Known(0),
                        conditional_provider_upload: None,
                        provider_host_mediated: false,
                    },
                ))
            } else {
                None
            };
            let mut selected = if let Some(first) = first.take() {
                first.selected
            } else {
                selected
                    .expect("later batch requires a new selection")
                    .map_err(|cause| after_stream_commit("randi", cause))?
                    .ok_or_else(|| {
                        after_stream_commit(
                            "randi",
                            error("randi: selected batch continuation unavailable"),
                        )
                    })?
            };
            drop(query);
            drop(_query_receipt);
            let receipt = if let Some(runtime) = runtime.as_ref() {
                let (preparation, retained) = register.map_or((0, 0), |quote| {
                    (quote.preparation_bytes, quote.retained_bytes)
                });
                let plan = preparation
                    .checked_add(physical.pending_view_workspace_bytes)
                    .ok_or_else(|| error("randi: pending view workspace overflow"))
                    .and_then(|preparation| {
                        retained
                            .checked_add(readback_retained)
                            .ok_or_else(|| error("randi: batch retained control extent overflow"))
                            .map(|retained| (preparation, retained))
                    })
                    .and_then(|(preparation, retained)| {
                        HostAllocationPlan::new(preparation, retained)
                            .map_err(|cause| error(format!("randi: batch host plan: {cause}")))
                    })
                    .and_then(|plan| {
                        HostAllocationReceipt::admit(Some(runtime), plan)
                            .map_err(|cause| error(format!("randi: batch host admission: {cause}")))
                    });
                match plan {
                    Ok(receipt) => receipt,
                    Err(cause) => {
                        selected.finish(Err(ProviderObservationFailure::ResourceLimit));
                        let failure = error(format!("randi: batch metadata admission: {cause}"));
                        return Err(if pending.is_some() {
                            after_stream_commit("randi", failure)
                        } else {
                            failure
                        });
                    }
                }
            } else {
                None
            };
            let permit = if let (Some(runtime), Some(receipt), Some(_register)) =
                (runtime.as_ref(), receipt.as_ref(), register)
            {
                let retained = match receipt.retained_view_permit(runtime) {
                    Ok(retained) => retained,
                    Err(cause) => {
                        selected.finish(Err(ProviderObservationFailure::ResourceLimit));
                        let failure = error(format!("randi: batch metadata permit: {cause}"));
                        return Err(if pending.is_some() {
                            after_stream_commit("randi", failure)
                        } else {
                            failure
                        });
                    }
                };
                match provider.prepare_output_host_metadata(Box::new(retained), 2) {
                    Ok(permit) => Some(permit),
                    Err(cause) => {
                        selected.finish(Err(ProviderObservationFailure::ProviderFailed));
                        let failure = provider_error(cause);
                        return Err(if pending.is_some() {
                            after_stream_commit("randi", failure)
                        } else {
                            failure
                        });
                    }
                }
            } else {
                None
            };
            let readback_claim = if let (Some(runtime), Some(receipt)) =
                (runtime.as_ref(), receipt.as_ref())
            {
                match receipt.claim_retained_metadata(runtime, physical.readback_control_host_bytes)
                {
                    Ok(claim) => Some(claim),
                    Err(cause) => {
                        selected.finish(Err(ProviderObservationFailure::ResourceLimit));
                        let failure = error(format!("randi: count readback admission: {cause}"));
                        return Err(if pending.is_some() {
                            after_stream_commit("randi", failure)
                        } else {
                            failure
                        });
                    }
                }
            } else {
                None
            };
            if pending.is_none() {
                if let Err(cause) = gpu::reserve_if_snapshot(snapshot, count, "randi") {
                    selected.finish(Err(ProviderObservationFailure::ResourceLimit));
                    return Err(cause);
                }
            }
            let progress = runmat_accelerate_api::observation::observe_provider_future(
                Some(selected.recorder()),
                provider.execute_gpu_integer_batch(batch, permit.as_deref(), readback_claim),
            )
            .await;
            let progress = match progress {
                Ok(progress) => progress,
                Err(cause) => {
                    selected.finish(Err(ProviderObservationFailure::ProviderFailed));
                    return Err(after_stream_commit("randi", provider_error(cause)));
                }
            };
            let (mut buffers, complete) = match validate_progress(
                provider,
                source,
                batch,
                progress,
                physical.output_buffer_bytes,
                physical.accepted_buffer_bytes,
            ) {
                Ok(result) => result,
                Err(cause) => {
                    selected.finish(Err(ProviderObservationFailure::PublicationFailed));
                    return Err(after_stream_commit("randi", cause));
                }
            };
            let publication_valid = match batch.pending {
                Some(previous) => selected.validate_reused_direct_provider_outputs(
                    &[&buffers.output, &buffers.accepted],
                    &[&previous.output, &previous.accepted],
                ),
                None => {
                    selected.validate_direct_provider_outputs(&[&buffers.output, &buffers.accepted])
                }
            };
            if !publication_valid {
                selected.finish(Err(ProviderObservationFailure::PublicationFailed));
                return Err(after_stream_commit(
                    "randi",
                    gpu_helpers::reject_gpu_outputs(
                        [buffers.output, buffers.accepted],
                        error("randi: batch result violates selected physical bound"),
                    ),
                ));
            }
            if complete {
                if let Err(cause) =
                    gpu_helpers::propagate_output_provenance(&mut buffers.output, [source])
                {
                    selected.finish(Err(ProviderObservationFailure::PublicationFailed));
                    return Err(after_stream_commit(
                        "randi",
                        gpu_helpers::reject_gpu_outputs([buffers.output, buffers.accepted], cause),
                    ));
                }
                drop(pending.take());
                if let Err(cause) = buffers.accepted.release() {
                    selected.finish(Err(ProviderObservationFailure::PublicationFailed));
                    return Err(after_stream_commit(
                        "randi",
                        gpu_helpers::reject_gpu_output(
                            buffers.output,
                            error(format!("randi: completion mask retirement failed: {cause}")),
                        ),
                    ));
                }
                selected.finish(Ok(true));
                return Ok(buffers.output);
            }
            selected.finish(Ok(true));
            pending = Some(buffers);
            first_attempt = first_attempt
                .checked_add(u64::from(attempts_per_item))
                .ok_or_else(|| {
                    after_stream_commit("randi", error("randi: integer attempt lane exhausted"))
                })?;
        }
    }
    .await;
    match result {
        Ok(output) => Ok(output),
        Err(cause) => {
            // A pending batch is proof that the child range and at least one
            // provider submission already happened. Even an error while
            // preparing the next query is terminal, not an Unsupported
            // capability decline eligible for replay on a different route.
            let cause = if pending.is_some() {
                after_stream_commit("randi", cause)
            } else {
                cause
            };
            match pending.take() {
                Some(held) => Err(gpu_helpers::reject_gpu_outputs(
                    [held.output, held.accepted],
                    cause,
                )),
                None => Err(cause),
            }
        }
    }
}
