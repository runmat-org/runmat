use futures::executor::block_on;
use runmat_accelerate_api::{provider, HostTensorView};

use crate::{
    algebra::{AssembledRealSystem, RealSparseMatrix},
    assembly::AssemblySummary,
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::{
        backend::kind::LinearAlgebraBackendKind, linear::LinearSolveResult,
        preconditioner::SpdPreconditionerKind,
    },
};

mod operator_impl;
mod preconditioner_impl;

use operator_impl::{apply_k_device, dot_handle, CsrDeviceOperatorContext, DeviceOperatorContext};
use preconditioner_impl::{apply_preconditioner_device, PreconditionerDeviceContext};

#[derive(Debug, Clone)]
pub struct RuntimeTensorPreparedLinearSystem {
    pub(crate) dof_count: usize,
    pub(crate) shape: Vec<usize>,
    pub(crate) matrix: RealSparseMatrix,
    pub(crate) inv_diag: Vec<f64>,
}

pub(crate) struct RuntimeTensorWorkspace;

impl RuntimeTensorWorkspace {
    fn new() -> Self {
        Self
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
    solve_operator_system_runtime_tensor_with_initial_guess(
        &summary.operator,
        preconditioner_kind,
        initial_guess,
    )
}

pub fn solve_operator_system_runtime_tensor_with_initial_guess(
    operator: &AssembledRealSystem,
    preconditioner_kind: SpdPreconditionerKind,
    initial_guess: Option<&[f64]>,
) -> Option<LinearSolveResult> {
    if preconditioner_kind != SpdPreconditionerKind::Jacobi {
        return None;
    }
    solve_runtime_tensor_linear_system_internal(
        operator,
        None,
        operator.load(),
        preconditioner_kind,
        initial_guess,
    )
}

pub fn prepare_runtime_tensor_linear_system(
    summary: &AssemblySummary,
) -> Option<RuntimeTensorPreparedLinearSystem> {
    prepare_runtime_tensor_operator_system(&summary.operator)
}

pub fn prepare_runtime_tensor_operator_system(
    operator: &AssembledRealSystem,
) -> Option<RuntimeTensorPreparedLinearSystem> {
    let prepared_system = operator.prepare_linear().ok()?;
    let n = prepared_system.dof_count();
    if n == 0 {
        return None;
    }

    let inv_diag: Vec<f64> = (0..n)
        .map(|i| {
            if prepared_system.constrained_dofs()[i] {
                1.0
            } else {
                1.0 / prepared_system.matrix().get(i, i).abs().max(1.0e-12)
            }
        })
        .collect();
    Some(RuntimeTensorPreparedLinearSystem {
        dof_count: n,
        shape: vec![n],
        matrix: prepared_system.matrix().clone(),
        inv_diag,
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
        &summary.operator,
        Some(prepared),
        rhs,
        preconditioner_kind,
        initial_guess,
    )
}

fn solve_runtime_tensor_linear_system_internal(
    operator: &AssembledRealSystem,
    prepared: Option<&RuntimeTensorPreparedLinearSystem>,
    rhs: &[f64],
    preconditioner_kind: SpdPreconditionerKind,
    initial_guess: Option<&[f64]>,
) -> Option<LinearSolveResult> {
    let provider = provider()?;
    let dof_count = prepared
        .map(|value| value.dof_count)
        .unwrap_or(operator.dof_count());
    if dof_count == 0 || rhs.len() != dof_count {
        return None;
    }
    let fallback_system = operator.prepare_linear().ok()?;

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
                    if fallback_system.constrained_dofs()[i] {
                        1.0
                    } else {
                        1.0 / fallback_system.matrix().get(i, i).abs().max(1.0e-12)
                    }
                })
                .collect()
        });
    let matrix = prepared
        .map(|value| value.matrix.clone())
        .unwrap_or_else(|| fallback_system.matrix().clone());
    let mut workspace = RuntimeTensorWorkspace::new();

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
    let matrix_values_h = provider
        .upload(&HostTensorView {
            data: matrix.values(),
            shape: &[matrix.values().len()],
        })
        .ok()?;
    let device_operator = DeviceOperatorContext {
        provider,
        csr: CsrDeviceOperatorContext {
            matrix: &matrix,
            values: &matrix_values_h,
        },
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
        inverse_diagonal: &inv,
    };

    let mut z =
        apply_preconditioner_device(&preconditioner_ctx, preconditioner_kind, &r, &mut workspace)?;
    let mut p = block_on(provider.elem_add(&z, &zero_h)).ok()?;

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
        if crate::is_fea_cancelled() {
            break;
        }
        device_apply_k_attempt_count = device_apply_k_attempt_count.saturating_add(1);
        let ap = apply_k_device(&device_operator, &p)?;
        device_apply_k_count = device_apply_k_count.saturating_add(1);

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
    let _ = provider.free(&matrix_values_h);
    let _ = provider.free(&rhs_h);

    Some(LinearSolveResult {
        iterations,
        residual_norm,
        converged,
        host_sync_count,
        solver_backend: LinearAlgebraBackendKind::RuntimeTensor,
        device_apply_k_count,
        device_apply_k_attempt_count,
        solution: x_host.data,
        solver_method: SolverMethod::MatrixFreePcg,
        preconditioner: SolverPreconditioner::from_spd(preconditioner_kind),
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
