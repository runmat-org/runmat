use futures::executor::block_on;
use runmat_accelerate_api::GpuTensorHandle;

use crate::{assembly::AssemblySummary, solve::preconditioner::SpdPreconditionerKind};

use super::RuntimeTensorWorkspace;

pub(super) struct PreconditionerDeviceContext<'a> {
    pub(super) provider: &'a dyn runmat_accelerate_api::AccelProvider,
    pub(super) inv_diag: &'a GpuTensorHandle,
    pub(super) ilu_l_subdiag: &'a GpuTensorHandle,
    pub(super) ilu_upper_superdiag: &'a GpuTensorHandle,
    pub(super) ilu_inv_u_diag: &'a GpuTensorHandle,
    pub(super) constrained_mask: &'a GpuTensorHandle,
    pub(super) unconstrained_mask: &'a GpuTensorHandle,
    pub(super) prev_indices: &'a [u32],
    pub(super) next_indices: &'a [u32],
    pub(super) shape: &'a [usize],
    pub(super) zero_like: &'a GpuTensorHandle,
}

pub(super) fn apply_preconditioner_device(
    ctx: &PreconditionerDeviceContext<'_>,
    preconditioner_kind: SpdPreconditionerKind,
    r: &GpuTensorHandle,
    workspace: &mut RuntimeTensorWorkspace,
) -> Option<GpuTensorHandle> {
    match preconditioner_kind {
        SpdPreconditionerKind::Jacobi => {
            let values = block_on(ctx.provider.elem_mul(r, ctx.inv_diag)).ok()?;
            let z_target = ensure_workspace_slot(
                ctx.provider,
                &mut workspace.precond_z,
                ctx.zero_like,
                &workspace.full_indices,
                &values,
            )?;
            let _ = ctx.provider.free(&values);
            block_on(ctx.provider.elem_add(&z_target, ctx.zero_like)).ok()
        }
        SpdPreconditionerKind::Ilu0 => apply_ilu0_approx_device(ctx, r, workspace),
    }
}

fn apply_ilu0_approx_device(
    ctx: &PreconditionerDeviceContext<'_>,
    r: &GpuTensorHandle,
    workspace: &mut RuntimeTensorWorkspace,
) -> Option<GpuTensorHandle> {
    const SWEEPS: usize = 3;

    let y = ensure_workspace_slot(
        ctx.provider,
        &mut workspace.precond_y,
        ctx.zero_like,
        &workspace.full_indices,
        r,
    )?;
    for _ in 0..SWEEPS {
        let y_prev_shift = ctx
            .provider
            .gather_linear(&y, ctx.prev_indices, ctx.shape)
            .ok()?;
        let l_term = block_on(ctx.provider.elem_mul(ctx.ilu_l_subdiag, &y_prev_shift)).ok()?;
        let tmp = block_on(ctx.provider.elem_sub(r, &l_term)).ok()?;
        let unconstrained = block_on(ctx.provider.elem_mul(ctx.unconstrained_mask, &tmp)).ok()?;
        let constrained = block_on(ctx.provider.elem_mul(ctx.constrained_mask, r)).ok()?;
        let y_next = block_on(ctx.provider.elem_add(&unconstrained, &constrained)).ok()?;
        let _ = ctx
            .provider
            .scatter_linear(&y, &workspace.full_indices, &y_next);

        let _ = ctx.provider.free(&constrained);
        let _ = ctx.provider.free(&unconstrained);
        let _ = ctx.provider.free(&tmp);
        let _ = ctx.provider.free(&l_term);
        let _ = ctx.provider.free(&y_prev_shift);
        let _ = ctx.provider.free(&y_next);
    }

    let z = ensure_workspace_slot(
        ctx.provider,
        &mut workspace.precond_z,
        ctx.zero_like,
        &workspace.full_indices,
        &y,
    )?;
    for _ in 0..SWEEPS {
        let z_next_shift = ctx
            .provider
            .gather_linear(&z, ctx.next_indices, ctx.shape)
            .ok()?;
        let u_term = block_on(
            ctx.provider
                .elem_mul(ctx.ilu_upper_superdiag, &z_next_shift),
        )
        .ok()?;
        let tmp = block_on(ctx.provider.elem_sub(&y, &u_term)).ok()?;
        let scaled = block_on(ctx.provider.elem_mul(&tmp, ctx.ilu_inv_u_diag)).ok()?;
        let unconstrained =
            block_on(ctx.provider.elem_mul(ctx.unconstrained_mask, &scaled)).ok()?;
        let constrained = block_on(ctx.provider.elem_mul(ctx.constrained_mask, &y)).ok()?;
        let z_new = block_on(ctx.provider.elem_add(&unconstrained, &constrained)).ok()?;
        let _ = ctx
            .provider
            .scatter_linear(&z, &workspace.full_indices, &z_new);

        let _ = ctx.provider.free(&constrained);
        let _ = ctx.provider.free(&unconstrained);
        let _ = ctx.provider.free(&scaled);
        let _ = ctx.provider.free(&tmp);
        let _ = ctx.provider.free(&u_term);
        let _ = ctx.provider.free(&z_next_shift);
        let _ = ctx.provider.free(&z_new);
    }

    block_on(ctx.provider.elem_add(&z, ctx.zero_like)).ok()
}

fn ensure_workspace_slot(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    slot: &mut Option<GpuTensorHandle>,
    zero_like: &GpuTensorHandle,
    full_indices: &[u32],
    values: &GpuTensorHandle,
) -> Option<GpuTensorHandle> {
    if slot.is_none() {
        *slot = Some(block_on(provider.elem_add(zero_like, zero_like)).ok()?);
    }
    let handle = slot.as_ref()?.clone();
    provider
        .scatter_linear(&handle, full_indices, values)
        .ok()?;
    Some(handle)
}

pub(super) fn build_ilu0_factors(summary: &AssemblySummary) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = summary.dof_count;
    let constrained = &summary.operator.constrained;

    let mut lower = vec![0.0; n.saturating_sub(1)];
    let mut upper = vec![0.0; n.saturating_sub(1)];
    for i in 0..n.saturating_sub(1) {
        if constrained[i] || constrained[i + 1] {
            continue;
        }
        let coupling = summary.operator.stiffness_upper[i];
        lower[i] = -coupling;
        upper[i] = -coupling;
    }

    let mut u_diag = vec![1.0; n];
    if n > 0 {
        u_diag[0] = if constrained[0] {
            1.0
        } else {
            summary.operator.stiffness_diag[0].max(1.0e-12)
        };
    }

    let mut l_subdiag = vec![0.0; n];
    let mut upper_superdiag = vec![0.0; n];
    for i in 1..n {
        if constrained[i] {
            u_diag[i] = 1.0;
            continue;
        }
        let prev_u = u_diag[i - 1].abs().max(1.0e-12);
        let l = if constrained[i - 1] {
            0.0
        } else {
            lower[i - 1] / prev_u
        };
        let mut value = summary.operator.stiffness_diag[i] - l * upper[i - 1];
        if value.abs() < 1.0e-12 {
            value = 1.0e-12;
        }
        u_diag[i] = value;
        l_subdiag[i] = l;
    }

    for i in 0..n.saturating_sub(1) {
        if !(constrained[i] || constrained[i + 1]) {
            upper_superdiag[i] = upper[i];
        }
    }

    let inv_u_diag = u_diag
        .iter()
        .enumerate()
        .map(|(i, value)| {
            if constrained[i] {
                1.0
            } else {
                1.0 / value.abs().max(1.0e-12)
            }
        })
        .collect();

    (l_subdiag, upper_superdiag, inv_u_diag)
}
