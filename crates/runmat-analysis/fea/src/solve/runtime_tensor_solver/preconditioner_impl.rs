use futures::executor::block_on;
use runmat_accelerate_api::GpuTensorHandle;

use crate::solve::preconditioner::SpdPreconditionerKind;

use super::RuntimeTensorWorkspace;

pub(super) struct PreconditionerDeviceContext<'a> {
    pub(super) provider: &'a dyn runmat_accelerate_api::AccelProvider,
    pub(super) inverse_diagonal: &'a GpuTensorHandle,
}

pub(super) fn apply_preconditioner_device(
    context: &PreconditionerDeviceContext<'_>,
    preconditioner: SpdPreconditionerKind,
    residual: &GpuTensorHandle,
    _workspace: &mut RuntimeTensorWorkspace,
) -> Option<GpuTensorHandle> {
    if preconditioner != SpdPreconditionerKind::Jacobi {
        return None;
    }
    block_on(
        context
            .provider
            .elem_mul(residual, context.inverse_diagonal),
    )
    .ok()
}
