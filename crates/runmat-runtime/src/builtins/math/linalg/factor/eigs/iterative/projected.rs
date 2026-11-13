//! Pinned faer 0.24.4 low-level EVD: all arrays and MemStack are supplied.
//! evd_imp uses temp_mat_uninit for H/Z/Householder/Schur/X; unlike the
//! high-level eig wrapper, this path owns no hidden Mat allocation.
use super::storage::Buffer;
use crate::builtins::math::linalg::ops::dense_evd::{kernel, EvdFailure};
use crate::{
    context::{
        host_allocation::{
            AlignedHostWorkspace, HostAllocationError, HostAllocationReceipt, HostTypedWorkspace,
        },
        RuntimeContext,
    },
    execution::CpuCooperationPolicy,
    BuiltinResult,
};
use faer::dyn_stack::{MemBuffer, StackReq};
use num_complex::Complex64;

enum Scratch {
    Managed(AlignedHostWorkspace),
    Unbounded(MemBuffer),
}
pub(super) use kernel::Kind;
pub(super) struct Projected {
    pub matrix: Buffer<Complex64>,
    pub values: Buffer<Complex64>,
    pub vectors: Buffer<Complex64>,
    scratch: Scratch,
    order: usize,
    capacity: usize,
    kind: Kind,
}
pub(super) fn requirement(order: usize, kind: Kind) -> StackReq {
    kernel::requirement(order, kind, false)
}
pub(super) fn bytes(order: usize, kind: Kind) -> Result<u64, HostAllocationError> {
    let square = order
        .checked_mul(order)
        .ok_or(HostAllocationError::Overflow)?;
    let arrays = HostTypedWorkspace::<Complex64>::quoted_bytes(square)?
        .checked_mul(2)
        .and_then(|b| b.checked_add(HostTypedWorkspace::<Complex64>::quoted_bytes(order).ok()?))
        .ok_or(HostAllocationError::Overflow)?;
    arrays
        .checked_add(AlignedHostWorkspace::quoted_bytes(
            requirement(order, kind)
                .layout()
                .map_err(|_| HostAllocationError::Overflow)?,
        )?)
        .ok_or(HostAllocationError::Overflow)
}
impl Projected {
    pub async fn new(
        order: usize,
        kind: Kind,
        runtime: Option<&RuntimeContext>,
        receipt: Option<&HostAllocationReceipt>,
    ) -> BuiltinResult<Self> {
        let req = requirement(order, kind);
        let scratch = if let (Some(runtime), Some(receipt)) = (runtime, receipt) {
            Scratch::Managed(
                AlignedHostWorkspace::new(
                    req.layout().map_err(|_| {
                        super::admission(Some(runtime), HostAllocationError::Overflow)
                    })?,
                    receipt,
                    runtime,
                )
                .map_err(|e| super::admission(Some(runtime), e))?,
            )
        } else {
            Scratch::Unbounded(MemBuffer::new(req))
        };
        Ok(Self {
            matrix: Buffer::new(order * order, runtime, receipt).await?,
            values: Buffer::new(order, runtime, receipt).await?,
            vectors: Buffer::new(order * order, runtime, receipt).await?,
            scratch,
            order,
            capacity: order,
            kind,
        })
    }
    fn kernel(&mut self) -> Result<(), EvdFailure> {
        let scratch = match &mut self.scratch {
            Scratch::Managed(v) => v.as_mut_uninit(),
            Scratch::Unbounded(v) => &mut *v,
        };
        kernel::compute(
            self.order,
            self.kind,
            &self.matrix.get()[..self.order * self.order],
            &mut self.values.get_mut()[..self.order],
            &mut self.vectors.get_mut()[..self.order * self.order],
            None,
            scratch,
        )
    }
    pub async fn compute(
        mut self,
        active: usize,
        runtime: Option<&RuntimeContext>,
        receipt: Option<&HostAllocationReceipt>,
    ) -> BuiltinResult<Self> {
        super::work::Work::new(runtime, receipt)?.check()?;
        if active == 0 || active > self.capacity {
            return Err(super::invalid(
                "projected active order is outside admitted capacity",
            ));
        }
        let needed = requirement(active, self.kind);
        let admitted = requirement(self.capacity, self.kind);
        if needed.size_bytes() > admitted.size_bytes()
            || needed.align_bytes() > admitted.align_bytes()
        {
            return Err(super::invalid(
                "active projected EVD exceeds its quoted scratch layout",
            ));
        }
        self.order = active;
        let work = move || {
            let result = self.kernel();
            (self, result)
        };
        let (projected, result) = if let Some(runtime) = runtime {
            if runtime.service_ports().blocking_cpu().is_none()
                && runtime.execution().cpu_cooperation_policy() == CpuCooperationPolicy::NeverYield
            {
                work()
            } else {
                runtime
                    .cpu_cooperation()
                    .run_owned_blocking_in_context(
                        runtime,
                        runtime.service_ports().blocking_cpu().map(|s| s.as_ref()),
                        receipt.map(HostAllocationReceipt::retain_scratch_for_cpu_work),
                        work,
                    )
                    .await
                    .map_err(|e| e.into_runtime_error_in_context(Some(runtime)))?
            }
        } else {
            work()
        };
        super::work::Work::new(runtime, receipt)?.check()?;
        result.map_err(|e| {
            crate::build_runtime_error(format!("eigs: projected EVD failed: {:?}", e.0))
                .with_builtin("eigs")
                .with_source(e)
                .with_identifier("RunMat:eigs:ProjectedNoConvergence")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        })?;
        Ok(projected)
    }
}
