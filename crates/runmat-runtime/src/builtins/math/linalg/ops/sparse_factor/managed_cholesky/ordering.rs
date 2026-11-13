use super::{ClaimedVec, ManagedSparseCholeskyError};
use crate::builtins::math::linalg::ops::sparse_factor::managed_storage::Work;
use crate::context::host_allocation::{
    AlignedHostWorkspace, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::execution::CpuCooperationPolicy;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::{linalg::amd, SymbolicSparseColMatRef};
use runmat_value::SparseTensor;

enum Scratch {
    Managed(AlignedHostWorkspace),
    Unbounded(MemBuffer),
}
impl Scratch {
    fn bytes(&mut self) -> &mut [std::mem::MaybeUninit<u8>] {
        match self {
            Self::Managed(v) => v.as_mut_uninit(),
            Self::Unbounded(v) => &mut *v,
        }
    }
}

pub(super) async fn amd_order(
    sparse: &SparseTensor,
    work: &mut Work<'_>,
) -> Result<ClaimedVec<usize>, ManagedSparseCholeskyError> {
    let n = sparse.cols;
    let mut pointers = ClaimedVec::new();
    pointers.ensure(sparse.col_ptrs.len(), work).await?;
    for &value in sparse.col_ptrs.iter() {
        pointers.push_reserved(value);
        work.tick().await?;
    }
    let mut rows = ClaimedVec::new();
    rows.ensure(sparse.row_indices.len(), work).await?;
    for &value in sparse.row_indices.iter() {
        rows.push_reserved(value);
        work.tick().await?;
    }
    let mut perm = ClaimedVec::<usize>::zeroed(n, work).await?;
    let mut inverse = ClaimedVec::<usize>::zeroed(n, work).await?;
    let requirement = amd::order_maybe_unsorted_scratch::<usize>(n, rows.len());
    let layout = requirement
        .layout()
        .map_err(|_| HostAllocationError::Overflow)?;
    let runtime = work.required_runtime()?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(AlignedHostWorkspace::quoted_bytes(layout)?, 0)?,
    )?;
    work.fence.checkpoint(runtime)?;
    let mut scratch = match receipt.as_ref() {
        Some(receipt) => Scratch::Managed(AlignedHostWorkspace::new(layout, receipt, runtime)?),
        None => Scratch::Unbounded(MemBuffer::new(requirement)),
    };
    work.flush().await?;
    let job = move || {
        let matrix =
            SymbolicSparseColMatRef::new_checked(n, n, pointers.as_slice(), None, rows.as_slice());
        amd::order_maybe_unsorted(
            perm.as_mut_slice(),
            inverse.as_mut_slice(),
            matrix,
            Default::default(),
            MemStack::new(scratch.bytes()),
        )
        .map(|_| perm)
    };
    let result = if runtime.service_ports().blocking_cpu().is_none()
        && runtime.execution().cpu_cooperation_policy() == CpuCooperationPolicy::NeverYield
    {
        job()
    } else {
        work.required_budget()?
            .run_owned_blocking_in_context(
                runtime,
                runtime
                    .service_ports()
                    .blocking_cpu()
                    .map(|service| service.as_ref()),
                receipt
                    .as_ref()
                    .map(HostAllocationReceipt::retain_scratch_for_cpu_work),
                job,
            )
            .await?
    };
    if let Some(receipt) = receipt {
        receipt.checkpoint(runtime)?;
    }
    work.flush().await?;
    result.map_err(ManagedSparseCholeskyError::Library)
}
