use futures::executor::block_on;
use runmat_accelerate_api::{provider, HostTensorView};

use crate::{
    algebra::{
        validate_prepared_solver_compatibility, AssembledRealSystem, CoercivityCertificate,
        PreparedSolverAdmissibility, RealPreconditioner, RealSolver, RealSparseMatrix,
    },
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
    pub(crate) constrained_dofs: Vec<bool>,
    pub(crate) prepared_rhs: Vec<f64>,
    pub(crate) inv_diag: Vec<f64>,
    pub(crate) coercivity: CoercivityCertificate,
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
    let prepared = prepare_runtime_tensor_operator_system(operator)?;
    solve_runtime_tensor_linear_system_internal(
        operator,
        Some(&prepared),
        &prepared.prepared_rhs,
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
    validate_prepared_solver_compatibility(
        &prepared_system,
        RealSolver::ConjugateGradient,
        RealPreconditioner::Jacobi,
    )
    .ok()?;
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
        constrained_dofs: prepared_system.constrained_dofs().to_vec(),
        prepared_rhs: prepared_system.rhs().to_vec(),
        inv_diag,
        coercivity: prepared_system.admissibility()?.coercivity().clone(),
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
    let dof_count = prepared
        .map(|value| value.dof_count)
        .unwrap_or(operator.dof_count());
    if dof_count == 0 || rhs.len() != dof_count {
        return None;
    }
    let fallback_system = if prepared.is_none() {
        Some(operator.prepare_linear().ok()?)
    } else {
        None
    };

    let shape_storage = prepared.map_or_else(|| vec![dof_count], |value| value.shape.clone());
    let shape = shape_storage.as_slice();
    let zeros = vec![0.0; dof_count];
    let (matrix, inv_diag) = match prepared {
        Some(prepared) => {
            validate_runtime_prepared(operator, prepared, rhs)?;
            (prepared.matrix.clone(), prepared.inv_diag.clone())
        }
        None => {
            let fallback = fallback_system.as_ref()?;
            let inverse = (0..dof_count)
                .map(|index| {
                    if fallback.constrained_dofs()[index] {
                        1.0
                    } else {
                        1.0 / fallback.matrix().get(index, index).abs().max(1.0e-12)
                    }
                })
                .collect();
            (fallback.matrix().clone(), inverse)
        }
    };
    let provider = provider()?;
    let mut workspace = RuntimeTensorWorkspace::new();

    let mut initial_x = match initial_guess {
        Some(values)
            if values.len() == dof_count && values.iter().all(|value| value.is_finite()) =>
        {
            values.to_vec()
        }
        Some(_) => return None,
        None => zeros.clone(),
    };
    let constrained_dofs = prepared
        .map(|prepared| prepared.constrained_dofs.as_slice())
        .or_else(|| {
            fallback_system
                .as_ref()
                .map(|system| system.constrained_dofs())
        })?;
    for (index, constrained) in constrained_dofs.iter().copied().enumerate() {
        if constrained {
            initial_x[index] = rhs[index];
        }
    }
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

    let mut device_apply_k_count: u32 = 0;
    let mut device_apply_k_attempt_count: u32 = 0;
    let mut r = if initial_guess.is_some() || constrained_dofs.iter().any(|value| *value) {
        device_apply_k_attempt_count = device_apply_k_attempt_count.saturating_add(1);
        let ax = apply_k_device(&device_operator, &x)?;
        device_apply_k_count = device_apply_k_count.saturating_add(1);
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
    for _ in 0..max_iters {
        if crate::is_fea_cancelled() {
            break;
        }
        device_apply_k_attempt_count = device_apply_k_attempt_count.saturating_add(1);
        let ap = apply_k_device(&device_operator, &p)?;
        device_apply_k_count = device_apply_k_count.saturating_add(1);

        let denom = dot_handle(provider, &p, &ap, &mut host_sync_count)?;
        if !denom.is_finite() || denom <= 1.0e-18 {
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

fn validate_runtime_prepared(
    operator: &AssembledRealSystem,
    prepared: &RuntimeTensorPreparedLinearSystem,
    rhs: &[f64],
) -> Option<()> {
    let expected_matrix = operator
        .operators()
        .stiffness()
        .with_identity_constraints(operator.constrained_dofs())
        .ok()?;
    let expected_inverse_diagonal = (0..operator.dof_count())
        .map(|index| {
            if operator.constrained_dofs()[index] {
                1.0
            } else {
                1.0 / expected_matrix.get(index, index).abs().max(1.0e-12)
            }
        })
        .collect::<Vec<_>>();
    if prepared.dof_count != operator.dof_count()
        || prepared.shape != [operator.dof_count()]
        || prepared.constrained_dofs != operator.constrained_dofs()
        || prepared.prepared_rhs.len() != operator.dof_count()
        || prepared.prepared_rhs.iter().any(|value| !value.is_finite())
        || prepared.inv_diag != expected_inverse_diagonal
        || expected_matrix.pattern_identity().ok()? != prepared.matrix.pattern_identity().ok()?
        || expected_matrix.values_identity().ok()? != prepared.matrix.values_identity().ok()?
    {
        return None;
    }
    PreparedSolverAdmissibility::bind_existing(prepared.coercivity.clone(), &prepared.matrix, rhs)
        .ok()?;
    Some(())
}

pub fn estimate_runtime_tensor_pcg_host_syncs(max_iters: u32) -> u32 {
    let _ = max_iters;
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{
        CanonicalRealOperators, MatrixProperties, OperatorRole, RealSparseMatrix,
    };

    fn system(diagonal: [f64; 2], rhs: [f64; 2]) -> AssembledRealSystem {
        let stiffness = RealSparseMatrix::diagonal(
            diagonal.to_vec(),
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let mass = RealSparseMatrix::diagonal(
            vec![1.0; 2],
            OperatorRole::Mass,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let damping = RealSparseMatrix::diagonal(
            vec![0.0; 2],
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .unwrap();
        AssembledRealSystem::try_new(
            CanonicalRealOperators::try_new(stiffness, mass, damping).unwrap(),
            vec![false; 2],
            rhs.to_vec(),
        )
        .unwrap()
    }

    #[test]
    fn host_sync_estimator_matches_formula() {
        assert_eq!(estimate_runtime_tensor_pcg_host_syncs(0), 1);
        assert_eq!(estimate_runtime_tensor_pcg_host_syncs(1), 1);
        assert_eq!(estimate_runtime_tensor_pcg_host_syncs(64), 1);
    }

    #[test]
    fn cached_preparation_rejects_operator_and_preconditioner_substitution_before_provider_use() {
        let source = system([2.0, 3.0], [1.0, 1.0]);
        let substitute = system([2.0, 4.0], [1.0, 1.0]);
        let mut prepared = prepare_runtime_tensor_operator_system(&source).unwrap();

        assert_eq!(
            validate_runtime_prepared(&substitute, &prepared, substitute.load()),
            None
        );
        prepared.inv_diag[0] = 7.0;
        assert_eq!(
            validate_runtime_prepared(&source, &prepared, source.load()),
            None
        );
    }
}
