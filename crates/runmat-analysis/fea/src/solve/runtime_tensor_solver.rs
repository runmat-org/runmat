use futures::executor::block_on;
use runmat_accelerate_api::{provider, GpuTensorHandle, HostTensorView};

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    operator::apply_k,
    solve::{linear::LinearSolveResult, preconditioner::SpdPreconditionerKind},
};

pub fn solve_linear_system_runtime_tensor(
    summary: &AssemblySummary,
    preconditioner_kind: SpdPreconditionerKind,
) -> Option<LinearSolveResult> {
    if preconditioner_kind != SpdPreconditionerKind::Jacobi {
        return None;
    }
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

    let mut x = provider
        .upload(&HostTensorView {
            data: &zeros,
            shape: &shape,
        })
        .ok()?;
    let mut r = provider
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

    let mut z = block_on(provider.elem_mul(&r, &inv)).ok()?;
    let z_host = block_on(provider.download(&z)).ok()?;
    let mut p = provider
        .upload(&HostTensorView {
            data: &z_host.data,
            shape: &shape,
        })
        .ok()?;

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

    for _ in 0..max_iters {
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
            Some(value) => value,
            None => {
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

        let new_z = block_on(provider.elem_mul(&r, &inv)).ok()?;
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

    let x_host = block_on(provider.download(&x)).ok()?;
    let residual_norm = if let Some(rr) = last_rr {
        rr.sqrt()
    } else {
        dot_handle(provider, &r, &r, &mut host_sync_count)?.sqrt()
    };
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_SOLVER_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: "solver=pcg preconditioner=jacobi matrix_free=true backend=runtime_tensor"
            .to_string(),
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
    let _ = provider.free(&inv);
    let _ = provider.free(&diag_h);
    let _ = provider.free(&upper_left_h);
    let _ = provider.free(&upper_right_h);
    let _ = provider.free(&constrained_mask_h);
    let _ = provider.free(&unconstrained_mask_h);

    Some(LinearSolveResult {
        iterations,
        residual_norm,
        converged,
        host_sync_count,
        solution: x_host.data,
        solver_method: "matrix_free_pcg".to_string(),
        preconditioner: "jacobi".to_string(),
        diagnostics,
    })
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
        Err(_) => block_on(provider.download(&sum))
            .ok()
            .and_then(|host| host.data.first().copied()),
    };
    *host_sync_count = host_sync_count.saturating_add(1);
    let _ = provider.free(&sum);
    let _ = provider.free(&mul);
    out
}
