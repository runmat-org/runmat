use runmat_execution::resource::ExactShared;
use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError,
};

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostOutputPermit,
};
use crate::context::RuntimeContext;
use crate::workspace::retained::WorkspaceName;

use super::BindingTreeValue;

pub(super) type Link = Option<ExactShared<Node>>;

/// Claimed physical header plus shared binding/root owners. Claim is last so
/// child/name/value ownership retires before its exact node budget refunds.
#[derive(Debug)]
pub(super) struct Node {
    pub(super) name: WorkspaceName,
    pub(super) binding: BindingTreeValue,
    pub(super) left: Link,
    pub(super) right: Link,
    pub(super) height: usize,
    pub(super) len: usize,
    _claim: Option<ExactHostAllocationClaim>,
}

impl Node {
    pub(super) fn requested_bytes() -> usize {
        ExactShared::<Self>::requested_bytes()
    }

    pub(super) fn link_height(link: &Link) -> usize {
        link.as_ref().map_or(0, |node| node.get().height)
    }

    pub(super) fn link_len(link: &Link) -> usize {
        link.as_ref().map_or(0, |node| node.get().len)
    }

    pub(super) fn make(
        name: WorkspaceName,
        binding: BindingTreeValue,
        left: Link,
        right: Link,
        context: Option<&RuntimeContext>,
    ) -> Result<ExactShared<Self>, HostAllocationError> {
        let height = Self::link_height(&left)
            .max(Self::link_height(&right))
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        let len = Self::link_len(&left)
            .checked_add(Self::link_len(&right))
            .and_then(|children| children.checked_add(1))
            .ok_or(HostAllocationError::Overflow)?;
        let physical = Self::requested_bytes();
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(physical).map_err(|_| HostAllocationError::Overflow)?,
            1,
        )?;
        let plan = HostAllocationPlan::new(0, retained)?;
        match HostAllocationReceipt::admit(context, plan)? {
            Some(receipt) => {
                let context = context.expect("managed receipt has an operation context");
                receipt.with_output_allocation(context, |permit| {
                    let claim = permit.reserve_exact_claim(physical)?;
                    ExactShared::try_new(Self {
                        name,
                        binding,
                        left,
                        right,
                        height,
                        len,
                        _claim: Some(claim),
                    })
                    .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
                })
            }
            None => ExactShared::try_new(Self {
                name,
                binding,
                left,
                right,
                height,
                len,
                _claim: None,
            })
            .map_err(|_| {
                HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed)
            }),
        }
    }

    /// Bulk import draws each exact node claim from its already admitted
    /// generation permit. The caller owns the sorting scratch and retains the
    /// permit until every constructed node has transferred its claim.
    pub(super) fn make_with_permit(
        name: WorkspaceName,
        binding: BindingTreeValue,
        left: Link,
        right: Link,
        permit: &HostOutputPermit,
    ) -> Result<ExactShared<Self>, HostAllocationError> {
        let height = Self::link_height(&left)
            .max(Self::link_height(&right))
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        let len = Self::link_len(&left)
            .checked_add(Self::link_len(&right))
            .and_then(|children| children.checked_add(1))
            .ok_or(HostAllocationError::Overflow)?;
        let claim = permit.reserve_exact_claim(Self::requested_bytes())?;
        ExactShared::try_new(Self {
            name,
            binding,
            left,
            right,
            height,
            len,
            _claim: Some(claim),
        })
        .map_err(|_| {
            HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed)
        })
    }
}
