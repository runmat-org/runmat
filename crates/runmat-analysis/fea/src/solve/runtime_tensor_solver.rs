use futures::executor::block_on;
use runmat_accelerate_api::{provider, GpuTensorHandle, HostTensorView};

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    operator::apply_k,
    solve::{linear::LinearSolveResult, preconditioner::SpdPreconditionerKind},
};

struct RuntimeTensorWorkspace {
    full_indices: Vec<u32>,
    precond_y: Option<GpuTensorHandle>,
    precond_z: Option<GpuTensorHandle>,
}

impl RuntimeTensorWorkspace {
    fn new(n: usize) -> Option<Self> {
        if n > u32::MAX as usize {
            return None;
        }
        Some(Self {
            full_indices: (0..n as u32).collect(),
            precond_y: None,
            precond_z: None,
        })
    }

    fn release(&mut self, provider: &dyn runmat_accelerate_api::AccelProvider) {
        if let Some(handle) = self.precond_y.take() {
            let _ = provider.free(&handle);
        }
        if let Some(handle) = self.precond_z.take() {
            let _ = provider.free(&handle);
        }
    }
}

pub fn solve_linear_system_runtime_tensor(
    summary: &AssemblySummary,
    preconditioner_kind: SpdPreconditionerKind,
) -> Option<LinearSolveResult> {
    solve_linear_system_runtime_tensor_with_initial_guess(summary, preconditioner_kind, None)
}

pub fn solve_linear_system_runtime_tensor_with_initial_guess(
    summary: &AssemblySummary,
    preconditioner_kind: SpdPreconditionerKind,
    initial_guess: Option<&[f64]>,
) -> Option<LinearSolveResult> {
    let provider = provider()?;
    let n = summary.dof_count;
    if n == 0 {
        return None;
    }

    let shape = [n];
    let zeros = vec![0.0; n];
    let inv_diag: Vec<f64> = (0..n)
        .map(|i| {
            if summary.operator.constrained[i] {
                1.0
            } else {
                1.0 / summary.operator.stiffness_diag[i].abs().max(1.0e-12)
            }
        })
        .collect();
    let (ilu_l_subdiag, ilu_upper_superdiag, ilu_inv_u_diag) = build_ilu0_factors(summary);
    let diag = summary.operator.stiffness_diag.clone();
    let mut upper_left = vec![0.0; n];
    let mut upper_right = vec![0.0; n];
    for i in 0..n {
        if i > 0 && !summary.operator.constrained[i - 1] && !summary.operator.constrained[i] {
            upper_left[i] = summary.operator.stiffness_upper[i - 1];
        }
        if i + 1 < n && !summary.operator.constrained[i + 1] && !summary.operator.constrained[i] {
            upper_right[i] = summary.operator.stiffness_upper[i];
        }
    }
    let constrained_mask: Vec<f64> = summary
        .operator
        .constrained
        .iter()
        .map(|&value| if value { 1.0 } else { 0.0 })
        .collect();
    let unconstrained_mask: Vec<f64> = summary
        .operator
        .constrained
        .iter()
        .map(|&value| if value { 0.0 } else { 1.0 })
        .collect();

    let prev_indices = linear_shift_indices(n, -1)?;
    let next_indices = linear_shift_indices(n, 1)?;
    let mut workspace = RuntimeTensorWorkspace::new(n)?;

    let initial_x = match initial_guess {
        Some(values) if values.len() == n => values.to_vec(),
        _ => zeros.clone(),
    };
    let mut x = provider
        .upload(&HostTensorView {
            data: &initial_x,
            shape: &shape,
        })
        .ok()?;
    let zero_h = provider
        .upload(&HostTensorView {
            data: &zeros,
            shape: &shape,
        })
        .ok()?;
    let rhs_h = provider
        .upload(&HostTensorView {
            data: &summary.operator.rhs,
            shape: &shape,
        })
        .ok()?;
    let inv = provider
        .upload(&HostTensorView {
            data: &inv_diag,
            shape: &shape,
        })
        .ok()?;
    let ilu_l_subdiag_h = provider
        .upload(&HostTensorView {
            data: &ilu_l_subdiag,
            shape: &shape,
        })
        .ok()?;
    let ilu_upper_superdiag_h = provider
        .upload(&HostTensorView {
            data: &ilu_upper_superdiag,
            shape: &shape,
        })
        .ok()?;
    let ilu_inv_u_diag_h = provider
        .upload(&HostTensorView {
            data: &ilu_inv_u_diag,
            shape: &shape,
        })
        .ok()?;
    let diag_h = provider
        .upload(&HostTensorView {
            data: &diag,
            shape: &shape,
        })
        .ok()?;
    let upper_left_h = provider
        .upload(&HostTensorView {
            data: &upper_left,
            shape: &shape,
        })
        .ok()?;
    let upper_right_h = provider
        .upload(&HostTensorView {
            data: &upper_right,
            shape: &shape,
        })
        .ok()?;
    let constrained_mask_h = provider
        .upload(&HostTensorView {
            data: &constrained_mask,
            shape: &shape,
        })
        .ok()?;
    let unconstrained_mask_h = provider
        .upload(&HostTensorView {
            data: &unconstrained_mask,
            shape: &shape,
        })
        .ok()?;

    let mut r = if initial_guess.is_some() {
        let ax = apply_k_device(
            provider,
            &x,
            &diag_h,
            &upper_left_h,
            &upper_right_h,
            &constrained_mask_h,
            &unconstrained_mask_h,
            &prev_indices,
            &next_indices,
            &shape,
        )?;
        let residual = block_on(provider.elem_sub(&rhs_h, &ax)).ok()?;
        let _ = provider.free(&ax);
        residual
    } else {
        block_on(provider.elem_add(&rhs_h, &zero_h)).ok()?
    };

    let preconditioner_ctx = PreconditionerDeviceContext {
        provider,
        inv_diag: &inv,
        ilu_l_subdiag: &ilu_l_subdiag_h,
        ilu_upper_superdiag: &ilu_upper_superdiag_h,
        ilu_inv_u_diag: &ilu_inv_u_diag_h,
        constrained_mask: &constrained_mask_h,
        unconstrained_mask: &unconstrained_mask_h,
        prev_indices: &prev_indices,
        next_indices: &next_indices,
        shape: &shape,
        zero_like: &zero_h,
    };

    let mut z =
        apply_preconditioner_device(&preconditioner_ctx, preconditioner_kind, &r, &mut workspace)?;
    // Initialize p = z without host roundtrip by adding device zero vector x.
    let mut p = block_on(provider.elem_add(&z, &x)).ok()?;

    let mut host_sync_count: u32 = 0;
    let mut rz_old = dot_handle(provider, &r, &z, &mut host_sync_count)?;
    let b_norm = summary
        .operator
        .rhs
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
        .max(1.0);
    let tol = 1.0e-8;
    let max_iters = 64;
    let mut converged = false;
    let mut iterations = 0u32;
    let mut last_rr: Option<f64> = None;
    let mut device_apply_k_count: u32 = 0;
    let mut device_apply_k_attempt_count: u32 = 0;

    for _ in 0..max_iters {
        device_apply_k_attempt_count = device_apply_k_attempt_count.saturating_add(1);
        let ap = match apply_k_device(
            provider,
            &p,
            &diag_h,
            &upper_left_h,
            &upper_right_h,
            &constrained_mask_h,
            &unconstrained_mask_h,
            &prev_indices,
            &next_indices,
            &shape,
        ) {
            Some(value) => {
                device_apply_k_count = device_apply_k_count.saturating_add(1);
                value
            }
            None => {
                host_sync_count = host_sync_count.saturating_add(1);
                let p_host = block_on(provider.download(&p)).ok()?;
                let ap_host = apply_k(&summary.operator, &p_host.data);
                provider
                    .upload(&HostTensorView {
                        data: &ap_host,
                        shape: &shape,
                    })
                    .ok()?
            }
        };

        let denom = dot_handle(provider, &p, &ap, &mut host_sync_count)?;
        if denom.abs() <= 1.0e-18 {
            let _ = provider.free(&ap);
            break;
        }

        let alpha = rz_old / denom;

        let scaled_p = provider.scalar_mul(&p, alpha).ok()?;
        let new_x = block_on(provider.elem_add(&x, &scaled_p)).ok()?;
        let _ = provider.free(&x);
        let _ = provider.free(&scaled_p);
        x = new_x;

        let scaled_ap = provider.scalar_mul(&ap, alpha).ok()?;
        let new_r = block_on(provider.elem_sub(&r, &scaled_ap)).ok()?;
        let _ = provider.free(&r);
        let _ = provider.free(&scaled_ap);
        let _ = provider.free(&ap);
        r = new_r;

        let rr = dot_handle(provider, &r, &r, &mut host_sync_count)?;
        last_rr = Some(rr);
        let residual_norm = rr.sqrt();
        iterations += 1;
        if residual_norm / b_norm <= tol {
            converged = true;
            break;
        }

        let new_z = apply_preconditioner_device(
            &preconditioner_ctx,
            preconditioner_kind,
            &r,
            &mut workspace,
        )?;
        let rz_new = dot_handle(provider, &r, &new_z, &mut host_sync_count)?;
        if rz_old.abs() <= 1.0e-18 {
            let _ = provider.free(&z);
            z = new_z;
            break;
        }
        let beta = rz_new / rz_old;

        let beta_p = provider.scalar_mul(&p, beta).ok()?;
        let new_p = block_on(provider.elem_add(&new_z, &beta_p)).ok()?;
        let _ = provider.free(&p);
        let _ = provider.free(&beta_p);
        let _ = provider.free(&z);
        p = new_p;
        z = new_z;
        rz_old = rz_new;
    }

    host_sync_count = host_sync_count.saturating_add(1);
    let x_host = block_on(provider.download(&x)).ok()?;
    let residual_norm = if let Some(rr) = last_rr {
        rr.sqrt()
    } else {
        dot_handle(provider, &r, &r, &mut host_sync_count)?.sqrt()
    };
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_SOLVER_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "solver=pcg preconditioner={} matrix_free=true backend=runtime_tensor",
            preconditioner_kind.as_str()
        ),
    }];
    if !converged {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_CG_MAX_ITERS".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: format!(
                "runtime_tensor pcg reached max iterations ({max_iters}) with residual_norm={residual_norm}"
            ),
        });
    }

    let _ = provider.free(&z);
    let _ = provider.free(&p);
    let _ = provider.free(&r);
    let _ = provider.free(&x);
    let _ = provider.free(&zero_h);
    let _ = provider.free(&inv);
    let _ = provider.free(&ilu_l_subdiag_h);
    let _ = provider.free(&ilu_upper_superdiag_h);
    let _ = provider.free(&ilu_inv_u_diag_h);
    let _ = provider.free(&diag_h);
    let _ = provider.free(&upper_left_h);
    let _ = provider.free(&upper_right_h);
    let _ = provider.free(&constrained_mask_h);
    let _ = provider.free(&unconstrained_mask_h);
    let _ = provider.free(&rhs_h);
    workspace.release(provider);

    Some(LinearSolveResult {
        iterations,
        residual_norm,
        converged,
        host_sync_count,
        solver_backend: "runtime_tensor".to_string(),
        device_apply_k_count,
        device_apply_k_attempt_count,
        solution: x_host.data,
        solver_method: "matrix_free_pcg".to_string(),
        preconditioner: preconditioner_kind.as_str().to_string(),
        diagnostics,
    })
}

struct PreconditionerDeviceContext<'a> {
    provider: &'a dyn runmat_accelerate_api::AccelProvider,
    inv_diag: &'a GpuTensorHandle,
    ilu_l_subdiag: &'a GpuTensorHandle,
    ilu_upper_superdiag: &'a GpuTensorHandle,
    ilu_inv_u_diag: &'a GpuTensorHandle,
    constrained_mask: &'a GpuTensorHandle,
    unconstrained_mask: &'a GpuTensorHandle,
    prev_indices: &'a [u32],
    next_indices: &'a [u32],
    shape: &'a [usize],
    zero_like: &'a GpuTensorHandle,
}

fn apply_preconditioner_device(
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

fn build_ilu0_factors(summary: &AssemblySummary) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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

fn apply_k_device(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    x: &GpuTensorHandle,
    diag: &GpuTensorHandle,
    upper_left: &GpuTensorHandle,
    upper_right: &GpuTensorHandle,
    constrained_mask: &GpuTensorHandle,
    unconstrained_mask: &GpuTensorHandle,
    prev_indices: &[u32],
    next_indices: &[u32],
    shape: &[usize],
) -> Option<GpuTensorHandle> {
    let x_prev = provider.gather_linear(x, prev_indices, shape).ok()?;
    let x_next = provider.gather_linear(x, next_indices, shape).ok()?;

    let diag_term = block_on(provider.elem_mul(diag, x)).ok()?;
    let left_term = block_on(provider.elem_mul(upper_left, &x_prev)).ok()?;
    let right_term = block_on(provider.elem_mul(upper_right, &x_next)).ok()?;
    let tmp = block_on(provider.elem_sub(&diag_term, &left_term)).ok()?;
    let unconstrained_value = block_on(provider.elem_sub(&tmp, &right_term)).ok()?;

    let unconstrained_part =
        block_on(provider.elem_mul(unconstrained_mask, &unconstrained_value)).ok()?;
    let constrained_part = block_on(provider.elem_mul(constrained_mask, x)).ok()?;
    let y = block_on(provider.elem_add(&unconstrained_part, &constrained_part)).ok()?;

    let _ = provider.free(&constrained_part);
    let _ = provider.free(&unconstrained_part);
    let _ = provider.free(&unconstrained_value);
    let _ = provider.free(&tmp);
    let _ = provider.free(&right_term);
    let _ = provider.free(&left_term);
    let _ = provider.free(&diag_term);
    let _ = provider.free(&x_next);
    let _ = provider.free(&x_prev);
    Some(y)
}

fn linear_shift_indices(n: usize, shift: isize) -> Option<Vec<u32>> {
    if n > u32::MAX as usize {
        return None;
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let shifted = (i as isize) + shift;
        let index = if shifted < 0 {
            0
        } else if shifted >= n as isize {
            n.saturating_sub(1)
        } else {
            shifted as usize
        };
        out.push(index as u32);
    }
    Some(out)
}

fn dot_handle(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    a: &GpuTensorHandle,
    b: &GpuTensorHandle,
    host_sync_count: &mut u32,
) -> Option<f64> {
    let mul = block_on(provider.elem_mul(a, b)).ok()?;
    let sum = block_on(provider.reduce_sum(&mul)).ok()?;
    let out = match provider.read_scalar(&sum, 0) {
        Ok(value) => Some(value),
        Err(_) => {
            *host_sync_count = host_sync_count.saturating_add(1);
            block_on(provider.download(&sum))
                .ok()
                .and_then(|host| host.data.first().copied())
        }
    };
    let _ = provider.free(&sum);
    let _ = provider.free(&mul);
    out
}

pub fn estimate_runtime_tensor_pcg_host_syncs(max_iters: u32) -> u32 {
    // With providers that support read_scalar, only final x download is required.
    // Fallback dot downloads are provider-specific and not counted by this baseline estimator.
    let _ = max_iters;
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_sync_estimator_matches_formula() {
        assert_eq!(estimate_runtime_tensor_pcg_host_syncs(0), 1);
        assert_eq!(estimate_runtime_tensor_pcg_host_syncs(1), 1);
        assert_eq!(estimate_runtime_tensor_pcg_host_syncs(64), 1);
    }
}
