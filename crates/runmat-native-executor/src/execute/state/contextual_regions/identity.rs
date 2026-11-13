//! One immutable, exactly admitted embedded-operation path. Clones across
//! suspended calls and derived indexes share its physical backing.

use runmat_execution::resource::ExactShared;
use runmat_runtime::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_value::{
    exact_capacity_vec_with, ExactHostAllocationClaim, ExactHostNumericAllocationError,
};
use std::alloc::Layout;
use std::hash::{Hash, Hasher};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::{continuation_admission_error, EmbeddedOperationSegment, OperationScope};

#[derive(Debug)]
struct IdentityPayload {
    path: Vec<EmbeddedOperationSegment>,
    _claim: Option<ExactHostAllocationClaim>,
}

#[derive(Clone, Debug)]
pub(crate) struct EmbeddedOperationIdentity(ExactShared<IdentityPayload>);

impl EmbeddedOperationIdentity {
    pub(super) fn prepare(
        runtime: &RuntimeContext,
        scopes: &[OperationScope],
    ) -> NativeExecutorResult<Self> {
        if scopes.is_empty() {
            return Err(NativeExecutorError::Host(
                "empty embedded-operation path has no identity".into(),
            ));
        }
        let backing = Layout::array::<EmbeddedOperationSegment>(scopes.len())
            .map_err(|_| NativeExecutorError::Host("embedded-operation path overflow".into()))?
            .size();
        let physical = backing
            .checked_add(ExactShared::<IdentityPayload>::requested_bytes())
            .ok_or_else(|| {
                NativeExecutorError::Host("embedded-operation layout overflow".into())
            })?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(physical).map_err(|_| {
                NativeExecutorError::Host("embedded-operation layout overflow".into())
            })?,
            1,
        )
        .map_err(|cause| continuation_admission_error(runtime, cause))?;
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| continuation_admission_error(runtime, cause))?;
        let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
            .map_err(|cause| continuation_admission_error(runtime, cause))?;
        let construct = |claim| {
            let path = exact_capacity_vec_with(scopes.len(), |index| scopes[index].segment)?;
            ExactShared::try_new(IdentityPayload {
                path,
                _claim: claim,
            })
            .map(Self)
            .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
        };
        match receipt {
            Some(receipt) => receipt
                .with_output_allocation(runtime, |permit| {
                    let claim = permit.reserve_exact_claim(physical)?;
                    construct(Some(claim))
                })
                .map_err(|cause| continuation_admission_error(runtime, cause)),
            None => construct(None).map_err(|cause| {
                continuation_admission_error(runtime, HostAllocationError::Numeric(cause))
            }),
        }
    }

    pub(super) fn path(&self) -> &[EmbeddedOperationSegment] {
        &self.0.get().path
    }

    pub fn is_descendant_of(&self, ancestor: Option<&Self>) -> bool {
        match ancestor {
            None => !self.path().is_empty(),
            Some(ancestor) => {
                self.path().len() > ancestor.path().len()
                    && self.path().starts_with(ancestor.path())
            }
        }
    }
}

impl PartialEq for EmbeddedOperationIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.path() == other.path()
    }
}

impl Eq for EmbeddedOperationIdentity {}

impl Hash for EmbeddedOperationIdentity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.path().hash(state);
    }
}
