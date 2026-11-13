//! Known-layout shared reader records, funded before allocation.
use runmat_execution::resource::ExactShared;
use runmat_runtime::context::host_allocation::{HostAllocationError, HostControlAllocationOwner};
use runmat_value::{ExactHostAllocationClaim, ExactHostNumericAllocationError};

use super::{OutputResult, RawOutputError, RawOutputReader};

struct RetainedReader {
    reader: RawOutputReader,
    _claim: Option<ExactHostAllocationClaim>,
}

#[derive(Clone)]
pub(crate) struct RawOutputReadHandle(ExactShared<RetainedReader>);

impl RawOutputReadHandle {
    pub(crate) fn new(
        reader: &RawOutputReader,
        owner: Option<&HostControlAllocationOwner>,
    ) -> OutputResult<Self> {
        let claim = owner
            .map(|owner| owner.claim(ExactShared::<RetainedReader>::requested_bytes()))
            .transpose()?;
        let reader = reader.retained_copy(owner)?;
        ExactShared::try_new(RetainedReader {
            reader,
            _claim: claim,
        })
        .map(Self)
        .map_err(|_| {
            RawOutputError::HostAllocation(HostAllocationError::Numeric(
                ExactHostNumericAllocationError::AllocationFailed,
            ))
        })
    }
}

impl std::ops::Deref for RawOutputReadHandle {
    type Target = RawOutputReader;
    fn deref(&self) -> &Self::Target {
        &self.0.get().reader
    }
}
