//! Exactly owned rare End result; ordinary sequence stages stay inline.
use super::super::super::state::{EmbeddedOperationIdentity, HostState};
use super::{begin_sequence_stage, SequenceStage};
use crate::{NativeExecutorError, NativeExecutorResult};
use runmat_runtime::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::RootedValue;
use runmat_runtime::object::protocol::PreparedSubscriptReceiver;
use runmat_runtime::sequence::SequenceDestinationBuilder;
use std::future::Future;

struct SequenceEnd {
    builder: SequenceDestinationBuilder,
    receiver: PreparedSubscriptReceiver,
    value: RootedValue,
}

impl SequenceEnd {
    fn into_payload(self: Box<Self>) -> Self {
        *self
    }
}

/// Only the rare End result needs independent backing across completion.
/// The actual Box retires before the external receipt on every drop path.
pub(in crate::execute) struct SequenceEndOwner {
    completion: Box<SequenceEnd>,
    metadata: Option<HostAllocationReceipt>,
}

impl SequenceEndOwner {
    pub(in crate::execute) fn into_parts(
        self,
    ) -> (
        SequenceDestinationBuilder,
        PreparedSubscriptReceiver,
        RootedValue,
    ) {
        let Self {
            completion,
            metadata,
        } = self;
        let completion = completion.into_payload();
        // Consuming the Box receiver returned after physical deallocation.
        drop(metadata);
        (completion.builder, completion.receiver, completion.value)
    }
}

/// Exact End backing admission, acquired after the completion-reuse gate but
/// before the future factory may take the original builder or run a callback.
pub(in crate::execute) struct SequenceEndAdmission {
    metadata: Option<HostAllocationReceipt>,
}

impl SequenceEndAdmission {
    fn prepare(runtime: &RuntimeContext) -> NativeExecutorResult<Self> {
        let allocation_error = |source| end_allocation_error(runtime, source);
        let bytes = u64::try_from(std::mem::size_of::<SequenceEnd>())
            .map_err(|_| allocation_error(HostAllocationError::Overflow))?;
        let metadata = HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(bytes, 0).map_err(allocation_error)?,
        )
        .map_err(allocation_error)?;
        Ok(Self { metadata })
    }

    pub(in crate::execute) fn finish(
        self,
        runtime: &RuntimeContext,
        builder: SequenceDestinationBuilder,
        receiver: PreparedSubscriptReceiver,
        value: RootedValue,
    ) -> NativeExecutorResult<SequenceStage> {
        check_metadata(self.metadata.as_ref(), runtime)?;
        let owner = SequenceEndOwner {
            completion: Box::new(SequenceEnd {
                builder,
                receiver,
                value,
            }),
            metadata: self.metadata,
        };
        check_metadata(owner.metadata.as_ref(), runtime)?;
        Ok(SequenceStage::End(owner))
    }
}

fn check_metadata(
    metadata: Option<&HostAllocationReceipt>,
    runtime: &RuntimeContext,
) -> NativeExecutorResult<()> {
    match metadata {
        Some(metadata) => metadata.checkpoint(runtime),
        None => HostAllocationReceipt::policy(Some(runtime)).and_then(|policy| {
            if policy == HostAllocationPolicy::Unbounded {
                Ok(())
            } else {
                Err(HostAllocationError::StaleAuthority)
            }
        }),
    }
    .map_err(|source| end_allocation_error(runtime, source))
}

fn end_allocation_error(
    runtime: &RuntimeContext,
    source: HostAllocationError,
) -> NativeExecutorError {
    runmat_runtime::context::host_allocation::host_allocation_runtime_error(
        Some(runtime),
        "RunMat:Native:SequencePreparation",
        "sequence End completion admission failed",
        source,
    )
    .into()
}

pub(in crate::execute) fn begin_sequence_end_stage<F>(
    state: &mut HostState,
    embedded: Option<EmbeddedOperationIdentity>,
    make_future: impl FnOnce(SequenceEndAdmission) -> F,
) -> NativeExecutorResult<SequenceStage>
where
    F: Future<Output = NativeExecutorResult<SequenceStage>> + 'static,
{
    let admission = SequenceEndAdmission::prepare(&state.runtime)?;
    begin_sequence_stage(state, embedded, move || make_future(admission))
}
