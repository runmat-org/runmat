//! Selected finite readback for admitted real numerical host continuations.
//! Every provider stage and native host payload is funded before its first
//! external effect, including earlier sources still live in the invocation.
use super::{source_matches, CpuReadbackFailure, CpuReadbackPlan, PreparedVmHostInputs};
use runmat_accelerate_api::{
    GpuTensorStorage, NumericElementType, ProviderFeasibilityQuery, ProviderQueryMetadataFootprint,
};
use runmat_runtime::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::{build_runtime_error, runtime_error, GpuGatherRetry, RuntimeError};
use runmat_value::{ExactHostTensorAllocation, NumericDType, Value};
use std::mem::size_of;

mod adoption;
use adoption::adopt_settled_download;

fn invalid(message: &'static str) -> RuntimeError {
    let mut error = build_runtime_error(message)
        .with_identifier("RunMat:Placement:InvalidHostReadback")
        .build();
    error.gpu_gather_retry = GpuGatherRetry::Never;
    error
}

fn admission_error(
    runtime: &RuntimeContext,
    cause: HostAllocationError,
    prior_effect: bool,
) -> CpuReadbackFailure {
    phase_error(
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:Placement:HostReadback",
            "selected numerical readback admission failed",
            cause,
        ),
        prior_effect,
    )
}

fn phase_error(error: RuntimeError, prior_effect: bool) -> CpuReadbackFailure {
    if prior_effect {
        CpuReadbackFailure::AfterEffect(error)
    } else {
        CpuReadbackFailure::BeforeEffect(error)
    }
}

struct QuoteError(&'static str);

fn callback_bound(source: &super::Source) -> Result<u64, QuoteError> {
    match source
        .owner
        .download_numeric_completion_host_bound(&source.handle)
    {
        runmat_execution::ResourceByteBound::Known(bytes) => Ok(bytes),
        runmat_execution::ResourceByteBound::Unavailable(_) => {
            Err(QuoteError("selected provider has no finite callback bound"))
        }
    }
}

pub(super) struct ReadbackQuote {
    pub plan: HostAllocationPlan,
    /// Decoded mirrors and their entry/mirror vectors remain live in the VM
    /// body after the last provider staging allocation has retired.
    pub body_live_bytes: u64,
}

pub(super) fn quote(sources: &[super::Source]) -> Option<ReadbackQuote> {
    quote_checked(sources).ok()
}

fn quote_checked(sources: &[super::Source]) -> Result<ReadbackQuote, QuoteError> {
    let mut retained_values = 0_u64;
    let mut phase_peak = 0_u64;
    let mut callback_claims = 0_u64;
    let mut entry_count = 0_usize;
    for source in sources {
        if source.handle.descriptor().storage != Some(GpuTensorStorage::Real)
            || runmat_accelerate_api::handle_is_logical(&source.handle)
        {
            return Err(QuoteError(
                "selected finite readback requires native real floating storage",
            ));
        }
        let shape = source.handle.shape();
        let dtype = match source.handle.physical().element_type() {
            NumericElementType::F32 => NumericDType::F32,
            NumericElementType::F64 => NumericDType::F64,
            _ => {
                return Err(QuoteError(
                    "selected host source is not native real floating",
                ))
            }
        };
        let rank = shape.len();
        let footprint = ProviderQueryMetadataFootprint {
            operation: "transfer.download",
            input_ranks: &[rank],
            output_ranks: &[rank],
            preparation_shape_ranks: &[rank],
            preparation_representation_slots: 0,
        };
        let query = footprint
            .requested_bytes()
            .ok_or(QuoteError("selected host query extent overflows"))?;
        let shape_bytes = rank
            .checked_mul(size_of::<usize>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(QuoteError("selected host shape extent overflows"))?;
        if source
            .payload_bytes
            .checked_add(shape_bytes)
            .is_none_or(|minimum| source.host_workspace < minimum)
        {
            return Err(QuoteError(
                "selected provider host workspace omits returned native backing",
            ));
        }
        let expected_capacity = shape
            .iter()
            .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or(QuoteError("selected host value extent overflows"))?;
        let adopted_raw =
            ExactHostTensorAllocation::adopted_requested_bytes(dtype, expected_capacity, rank)
                .ok()
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or(QuoteError("selected host value extent overflows"))?;
        let overhead = adopted_raw
            .checked_sub(source.payload_bytes)
            .and_then(|bytes| bytes.checked_sub(shape_bytes))
            .ok_or(QuoteError("selected host wrapper extent overflows"))?;
        let controls = HostAllocationPlan::retained_exact_output_bytes(
            0,
            ExactHostTensorAllocation::retained_claim_count(),
        )
        .map_err(|_| QuoteError("selected host claim extent overflows"))?;
        // The provider's host-workspace bound covers the returned Vec backing
        // at its actual capacity, even when it exceeds the logical length.
        // It may also include transient transpose/work buffers, so using the
        // entire bound for each retained mirror is conservative admission,
        // not a statement that those transient bytes remain physically live.
        // Fixed adoption metadata and both claim controls are additional.
        let full_value = source
            .host_workspace
            .checked_add(overhead)
            .and_then(|bytes| bytes.checked_add(controls))
            .ok_or(QuoteError("selected host retained extent overflows"))?;
        // Earlier decoded values remain live while the next native provider
        // workspace and its query/stage are allocated. Only one download is
        // active at a time; each finished value joins the retained mirrors.
        let active = retained_values
            .checked_add(source.host_workspace)
            .and_then(|bytes| bytes.checked_add(overhead))
            .and_then(|bytes| bytes.checked_add(controls))
            .and_then(|bytes| bytes.checked_add(query.checked_mul(2)?))
            .and_then(|bytes| bytes.checked_add(shape_bytes.checked_mul(2)?))
            .ok_or(QuoteError("selected host readback peak overflows"))?;
        retained_values = retained_values
            .checked_add(full_value)
            .ok_or(QuoteError("selected host value peak overflows"))?;
        phase_peak = phase_peak.max(active).max(retained_values);
        callback_claims = callback_claims
            .checked_add(
                HostAllocationPlan::retained_metadata_bytes(callback_bound(source)?)
                    .map_err(|_| QuoteError("selected host callback claim overflows"))?,
            )
            .ok_or(QuoteError("selected host callback total overflows"))?;
        entry_count = entry_count
            .checked_add(source.offsets.len())
            .ok_or(QuoteError("selected host operand count overflows"))?;
    }
    let entry_slots = entry_count
        .checked_mul(size_of::<(usize, Value)>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(QuoteError("selected host entry extent overflows"))?;
    let mirror_slots = sources
        .len()
        .checked_mul(size_of::<(super::Source, Value)>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(QuoteError("selected host mirror extent overflows"))?;
    let scratch = phase_peak
        .checked_add(entry_slots)
        .and_then(|bytes| bytes.checked_add(mirror_slots))
        .ok_or(QuoteError("selected host storage peak overflows"))?;
    let body_live_bytes = retained_values
        .checked_add(entry_slots)
        .and_then(|bytes| bytes.checked_add(mirror_slots))
        .ok_or(QuoteError("selected host body storage overflows"))?;
    Ok(ReadbackQuote {
        plan: HostAllocationPlan::new(scratch, callback_claims)
            .map_err(|_| QuoteError("selected host readback plan overflows"))?,
        body_live_bytes,
    })
}

impl CpuReadbackPlan {
    pub(crate) async fn prepare_inputs_managed(
        self,
        stack: &[Value],
        runtime: &RuntimeContext,
    ) -> Result<PreparedVmHostInputs, CpuReadbackFailure> {
        if self.sources.is_empty()
            || stack.len() != self.stack_len
            || !self.covers_resident_slice(stack)
        {
            return Err(CpuReadbackFailure::BeforeEffect(invalid(
                "selected host sources no longer match their admitted operands",
            )));
        }
        if self
            .sources
            .iter()
            .any(|source| !matches!(&source.capabilities, std::borrow::Cow::Borrowed(_)))
        {
            return Err(CpuReadbackFailure::BeforeEffect(invalid(
                "selected host provider cannot borrow its capability snapshot",
            )));
        }
        let entry_count = self
            .sources
            .iter()
            .try_fold(0_usize, |count, source| {
                count.checked_add(source.offsets.len())
            })
            .ok_or_else(|| {
                CpuReadbackFailure::BeforeEffect(invalid("selected host operand count overflows"))
            })?;
        let source_count = self.sources.len();
        let quote = quote_checked(&self.sources)
            .map_err(|cause| CpuReadbackFailure::BeforeEffect(invalid(cause.0)))?;
        let receipt = HostAllocationReceipt::admit(Some(runtime), quote.plan)
            .map_err(|cause| admission_error(runtime, cause, false))?
            .ok_or_else(|| {
                CpuReadbackFailure::BeforeEffect(invalid(
                    "selected host readback lost its finite allocation owner",
                ))
            })?;
        if quote_checked(&self.sources).ok().is_none_or(|current| {
            current.plan != quote.plan || current.body_live_bytes != quote.body_live_bytes
        }) || !self
            .sources
            .iter()
            .all(|source| source_matches(source, stack))
        {
            return Err(CpuReadbackFailure::BeforeEffect(invalid(
                "selected host readback resource bounds changed",
            )));
        }
        let resident_offsets = self.resident_offsets;
        let mut entry = runmat_value::exact_capacity_vec(entry_count).map_err(|_| {
            CpuReadbackFailure::BeforeEffect(invalid("selected host entry allocation failed"))
        })?;
        let mut mirrors = runmat_value::exact_capacity_vec(source_count).map_err(|_| {
            CpuReadbackFailure::BeforeEffect(invalid("selected host mirror allocation failed"))
        })?;
        let mut prior_effect = false;
        for source in self.sources {
            if !source_matches(&source, stack) {
                return Err(phase_error(
                    invalid("selected host source changed before staging"),
                    prior_effect,
                ));
            }
            receipt
                .checkpoint(runtime)
                .map_err(|cause| admission_error(runtime, cause, prior_effect))?;
            let callback = callback_bound(&source)
                .map_err(|cause| phase_error(invalid(cause.0), prior_effect))?;
            let claim = receipt
                .claim_retained_metadata(runtime, callback)
                .map_err(|cause| admission_error(runtime, cause, prior_effect))?;
            let query = ProviderFeasibilityQuery::exact_numeric_download(&source.handle)
                .ok_or_else(|| phase_error(invalid("selected host query changed"), prior_effect))?;
            let rank = source.handle.shape().len();
            let footprint = ProviderQueryMetadataFootprint {
                operation: "transfer.download",
                input_ranks: &[rank],
                output_ranks: &[rank],
                preparation_shape_ranks: &[rank],
                preparation_representation_slots: 0,
            };
            if !footprint.matches(&query) || !query.matches_numeric_download(&source.handle) {
                return Err(phase_error(
                    invalid("selected host query no longer matches its source"),
                    prior_effect,
                ));
            }
            let stage = source
                .owner
                .prepare_numeric_download_stage(&source.handle, &query, claim)
                .map_err(|cause| {
                    phase_error(
                        runtime_error::with_provider_execution_source(
                            invalid("selected host provider staging failed"),
                            runmat_accelerate_api::ProviderExecutionError::from(cause),
                        ),
                        prior_effect,
                    )
                })?;
            if !stage.matches_source(&source.handle) {
                return Err(phase_error(
                    invalid("selected host provider staged a different source"),
                    prior_effect,
                ));
            }
            receipt
                .checkpoint(runtime)
                .map_err(|cause| admission_error(runtime, cause, prior_effect))?;
            prior_effect = true;
            let host = runmat_runtime::builtins::common::gpu_helpers::download_numeric_preserving_residency_staged_async(
                source.owner,
                &source.handle,
                stage,
            )
            .await
            .map_err(CpuReadbackFailure::AfterEffect)?;
            receipt.checkpoint(runtime).map_err(|cause| {
                CpuReadbackFailure::AfterEffect(host_allocation_runtime_error(
                    Some(runtime),
                    "RunMat:Placement:HostReadback",
                    "selected host readback authority changed",
                    cause,
                ))
            })?;
            if !source_matches(&source, stack) {
                return Err(CpuReadbackFailure::AfterEffect(invalid(
                    "selected host source changed during readback",
                )));
            }
            let value =
                adopt_settled_download(host, &source, runtime, &receipt).map_err(|error| *error)?;
            entry.extend(source.offsets.iter().map(|&offset| (offset, value.clone())));
            mirrors.push((source, value));
        }
        if stack.iter().enumerate().any(|(offset, value)| {
            runmat_runtime::dispatcher::value_contains_gpu(value)
                && resident_offsets.binary_search(&offset).is_err()
        }) || !mirrors
            .iter()
            .all(|(source, _)| source_matches(source, stack))
        {
            return Err(CpuReadbackFailure::AfterEffect(invalid(
                "selected host sources changed before operand publication",
            )));
        }
        receipt.checkpoint(runtime).map_err(|cause| {
            CpuReadbackFailure::AfterEffect(host_allocation_runtime_error(
                Some(runtime),
                "RunMat:Placement:HostReadback",
                "selected host readback authority changed before publication",
                cause,
            ))
        })?;
        Ok(PreparedVmHostInputs {
            entry,
            mirrors,
            _managed_readback: Some(receipt),
        })
    }
}
