use futures::executor::block_on;
use runmat_accelerate_api::{provider, GpuTensorHandle, HostTensorView};

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    operator::dense_stiffness,
    solve::{linear::LinearSolveResult, preconditioner::SpdPreconditionerKind},
};

mod operator_impl;
mod preconditioner_impl;

use operator_impl::{
    apply_k_device, apply_k_host_from_prepared, dot_handle, linear_shift_indices,
    DeviceOperatorContext,
};
use preconditioner_impl::{
    apply_preconditioner_device, build_ilu0_factors, PreconditionerDeviceContext,
};

#[derive(Debug, Clone)]
pub struct RuntimeTensorPreparedLinearSystem {
    pub(crate) dof_count: usize,
    pub(crate) shape: Vec<usize>,
    pub(crate) diag: Vec<f64>,
    pub(crate) upper_left: Vec<f64>,
    pub(crate) upper_right: Vec<f64>,
    pub(crate) inv_diag: Vec<f64>,
    pub(crate) ilu_l_subdiag: Vec<f64>,
    pub(crate) ilu_upper_superdiag: Vec<f64>,
    pub(crate) ilu_inv_u_diag: Vec<f64>,
    pub(crate) constrained_mask: Vec<f64>,
    pub(crate) unconstrained_mask: Vec<f64>,
    pub(crate) prev_indices: Vec<u32>,
    pub(crate) next_indices: Vec<u32>,
}

pub(crate) struct RuntimeTensorWorkspace {
    pub(crate) full_indices: Vec<u32>,
    pub(crate) precond_y: Option<GpuTensorHandle>,
    pub(crate) precond_z: Option<GpuTensorHandle>,
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
    solve_runtime_tensor_linear_system_internal(
        summary,
        None,
        &summary.operator.rhs,
        preconditioner_kind,
        initial_guess,
    )
}

pub fn prepare_runtime_tensor_linear_system(
    summary: &AssemblySummary,
) -> Option<RuntimeTensorPreparedLinearSystem> {
    let n = summary.dof_count;
    if n == 0 || dense_stiffness(&summary.operator).is_some() {
        return None;
    }

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

    Some(RuntimeTensorPreparedLinearSystem {
        dof_count: n,
        shape: vec![n],
        diag,
        upper_left,
        upper_right,
        inv_diag,
        ilu_l_subdiag,
        ilu_upper_superdiag,
        ilu_inv_u_diag,
        constrained_mask,
        unconstrained_mask,
        prev_indices,
        next_indices,
    })
}

pub fn solve_prepared_linear_system_runtime_tensor(
    summary: &AssemblySummary,
    prepared: &RuntimeTensorPreparedLinearSystem,
    rhs: &[f64],
    preconditioner_kind: SpdPreconditionerKind,
    initial_guess: Option<&[f64]>,
) -> Option<LinearSolveResult> {
    solve_runtime_tensor_linear_system_internal(
        summary,
        Some(prepared),
        rhs,
        preconditioner_kind,
        initial_guess,
    )
}

fn solve_runtime_tensor_linear_system_internal(
    summary: &AssemblySummary,
    prepared: Option<&RuntimeTensorPreparedLinearSystem>,
    rhs: &[f64],
    preconditioner_kind: SpdPreconditionerKind,
    initial_guess: Option<&[f64]>,
) -> Option<LinearSolveResult> {
    let provider = provider()?;
    let dof_count = prepared
        .map(|value| value.dof_count)
        .unwrap_or(summary.dof_count);
    if dof_count == 0 || rhs.len() != dof_count {
        return None;
    }

    let shape_storage = prepared
        .map(|value| value.shape.clone())
        .unwrap_or_else(|| vec![dof_count]);
    let shape = shape_storage.as_slice();
    let zeros = vec![0.0; dof_count];
    let inv_diag = prepared
        .map(|value| value.inv_diag.clone())
        .unwrap_or_else(|| {
            (0..dof_count)
                .map(|i| {
                    if summary.operator.constrained[i] {
                        1.0
                    } else {
                        1.0 / summary.operator.stiffness_diag[i].abs().max(1.0e-12)
                    }
                })
                .collect()
        });
    let (ilu_l_subdiag, ilu_upper_superdiag, ilu_inv_u_diag) = prepared
        .map(|value| {
            (
                value.ilu_l_subdiag.clone(),
                value.ilu_upper_superdiag.clone(),
                value.ilu_inv_u_diag.clone(),
            )
        })
        .unwrap_or_else(|| build_ilu0_factors(summary));
    let (diag, upper_left, upper_right) = prepared
        .map(|value| {
            (
                value.diag.clone(),
                value.upper_left.clone(),
                value.upper_right.clone(),
            )
        })
        .unwrap_or_else(|| {
            let diag = summary.operator.stiffness_diag.clone();
            let mut upper_left = vec![0.0; dof_count];
            let mut upper_right = vec![0.0; dof_count];
            for i in 0..dof_count {
                if i > 0 && !summary.operator.constrained[i - 1] && !summary.operator.constrained[i]
                {
                    upper_left[i] = summary.operator.stiffness_upper[i - 1];
                }
                if i + 1 < dof_count
                    && !summary.operator.constrained[i + 1]
                    && !summary.operator.constrained[i]
                {
                    upper_right[i] = summary.operator.stiffness_upper[i];
                }
            }
            (diag, upper_left, upper_right)
        });
    let (constrained_mask, unconstrained_mask) = prepared
        .map(|value| {
            (
                value.constrained_mask.clone(),
                value.unconstrained_mask.clone(),
            )
        })
        .unwrap_or_else(|| {
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
            (constrained_mask, unconstrained_mask)
        });
    let prev_indices = prepared
        .map(|value| value.prev_indices.clone())
        .unwrap_or(linear_shift_indices(dof_count, -1)?);
    let next_indices = prepared
        .map(|value| value.next_indices.clone())
        .unwrap_or(linear_shift_indices(dof_count, 1)?);
    let mut workspace = RuntimeTensorWorkspace::new(dof_count)?;

    let initial_x = match initial_guess {
        Some(values) if values.len() == dof_count => values.to_vec(),
        _ => zeros.clone(),
    };
    let mut x = provider
        .upload(&HostTensorView {
            data: &initial_x,
            shape,
        })
        .ok()?;
    let zero_h = provider
        .upload(&HostTensorView {
            data: &zeros,
            shape,
        })
        .ok()?;
    let rhs_h = provider.upload(&HostTensorView { data: rhs, shape }).ok()?;
    let inv = provider
        .upload(&HostTensorView {
            data: &inv_diag,
            shape,
        })
        .ok()?;
    let ilu_l_subdiag_h = provider
        .upload(&HostTensorView {
            data: &ilu_l_subdiag,
            shape,
        })
        .ok()?;
    let ilu_upper_superdiag_h = provider
        .upload(&HostTensorView {
            data: &ilu_upper_superdiag,
            shape,
        })
        .ok()?;
    let ilu_inv_u_diag_h = provider
        .upload(&HostTensorView {
            data: &ilu_inv_u_diag,
            shape,
        })
        .ok()?;
    let diag_h = provider
        .upload(&HostTensorView { data: &diag, shape })
        .ok()?;
    let upper_left_h = provider
        .upload(&HostTensorView {
            data: &upper_left,
            shape,
        })
        .ok()?;
    let upper_right_h = provider
        .upload(&HostTensorView {
            data: &upper_right,
            shape,
        })
        .ok()?;
    let constrained_mask_h = provider
        .upload(&HostTensorView {
            data: &constrained_mask,
            shape,
        })
        .ok()?;
    let unconstrained_mask_h = provider
        .upload(&HostTensorView {
            data: &unconstrained_mask,
            shape,
        })
        .ok()?;
    let device_operator = DeviceOperatorContext {
        provider,
        diag: &diag_h,
        upper_left: &upper_left_h,
        upper_right: &upper_right_h,
        constrained_mask: &constrained_mask_h,
        unconstrained_mask: &unconstrained_mask_h,
        prev_indices: &prev_indices,
        next_indices: &next_indices,
        shape,
    };

    let mut r = if initial_guess.is_some() {
        let ax = apply_k_device(&device_operator, &x)?;
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
        shape,
        zero_like: &zero_h,
    };

    let mut z =
        apply_preconditioner_device(&preconditioner_ctx, preconditioner_kind, &r, &mut workspace)?;
    let mut p = block_on(provider.elem_add(&z, &x)).ok()?;

    let mut host_sync_count: u32 = 0;
    let mut rz_old = dot_handle(provider, &r, &z, &mut host_sync_count)?;
    let b_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
    let tol = 1.0e-8;
    let max_iters = 64;
    let mut converged = false;
    let mut iterations = 0u32;
    let mut last_rr: Option<f64> = None;
    let mut device_apply_k_count: u32 = 0;
    let mut device_apply_k_attempt_count: u32 = 0;

    for _ in 0..max_iters {
        device_apply_k_attempt_count = device_apply_k_attempt_count.saturating_add(1);
        let ap = match apply_k_device(&device_operator, &p) {
            Some(value) => {
                device_apply_k_count = device_apply_k_count.saturating_add(1);
                value
            }
            None => {
                host_sync_count = host_sync_count.saturating_add(1);
                let p_host = block_on(provider.download(&p)).ok()?;
                let ap_host = apply_k_host_from_prepared(
                    &diag,
                    &upper_left,
                    &upper_right,
                    &constrained_mask,
                    &unconstrained_mask,
                    &p_host.data,
                );
                provider
                    .upload(&HostTensorView {
                        data: &ap_host,
                        shape,
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

pub fn estimate_runtime_tensor_pcg_host_syncs(max_iters: u32) -> u32 {
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
