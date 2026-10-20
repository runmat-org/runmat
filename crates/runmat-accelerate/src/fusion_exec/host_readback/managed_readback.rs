//! Selected finite readback for admitted real numerical host continuations.
//! Every provider stage and native host payload is funded before its first
//! external effect, including earlier sources still live in the invocation.
use super::{source_matches, CpuReadbackFailure, CpuReadbackPlan, PreparedVmHostInputs};
use runmat_runtime::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::{build_runtime_error, GpuGatherRetry, RuntimeError};
use runmat_value::Value;
use std::mem::size_of;

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
        let source_quote =
            runmat_runtime::builtins::common::gpu_helpers::quote_exact_native_download(
                source.owner,
                &source.handle,
            )
            .map_err(|_| QuoteError("selected host source geometry is unavailable"))?;
        if source_quote.host_workspace != source.host_workspace
            || source_quote.payload_bytes != source.payload_bytes
        {
            return Err(QuoteError("selected host source geometry changed"));
        }
        // The provider's host-workspace bound covers the returned Vec backing
        // at its actual capacity, even when it exceeds the logical length.
        // It may also include transient transpose/work buffers, so using the
        // entire bound for each retained mirror is conservative admission,
        // not a statement that those transient bytes remain physically live.
        // Fixed adoption metadata and both claim controls are additional.
        let full_value = source_quote.retained_value_upper;
        // Earlier decoded values remain live while the next native provider
        // workspace and its query/stage are allocated. Only one download is
        // active at a time; each finished value joins the retained mirrors.
        let active = retained_values
            .checked_add(source_quote.active_upper)
            .ok_or(QuoteError("selected host readback peak overflows"))?;
        retained_values = retained_values
            .checked_add(full_value)
            .ok_or(QuoteError("selected host value peak overflows"))?;
        phase_peak = phase_peak.max(active).max(retained_values);
        callback_claims = callback_claims
            .checked_add(
                HostAllocationPlan::retained_metadata_bytes(source_quote.callback_bytes)
                    .map_err(|_| QuoteError("selected host callback claim overflows"))?,
            )
            .and_then(|bytes| bytes.checked_add(source_quote.precharged_output_bytes))
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
            let source_quote =
                runmat_runtime::builtins::common::gpu_helpers::quote_exact_native_download(
                    source.owner,
                    &source.handle,
                )
                .map_err(|error| phase_error(error, prior_effect))?;
            if source_quote.host_workspace != source.host_workspace
                || source_quote.payload_bytes != source.payload_bytes
            {
                return Err(phase_error(
                    invalid("selected host source geometry changed"),
                    prior_effect,
                ));
            }
            let value = runmat_runtime::builtins::common::gpu_helpers::download_exact_native(
                runtime,
                source.owner,
                &source.handle,
                &receipt,
                source_quote,
            )
            .await
            .map_err(|failure| phase_error(
                failure.error,
                prior_effect || failure.phase == runmat_runtime::builtins::common::gpu_helpers::ExactDownloadPhase::AfterEffect,
            ))?;
            prior_effect = true;
            if !source_matches(&source, stack) {
                return Err(CpuReadbackFailure::AfterEffect(invalid(
                    "selected host source changed during readback",
                )));
            }
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
