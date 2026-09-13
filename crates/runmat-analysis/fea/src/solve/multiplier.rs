//! Exact sparse constraint enforcement through a matrix-free saddle-point system.

use crate::{
    algebra::{apply_k, PreparedRealLinearSystem},
    assembly::AssemblySummary,
    constraints::ResolvedConstraintMatrix,
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::{
        backend::{kind::LinearAlgebraBackendKind, linear_algebra::LinearAlgebraBackend},
        krylov::{restarted_gmres, GmresPolicy},
        linear::LinearSolveResult,
    },
};

pub(super) fn solve_multiplier_constraint_system(
    summary: &AssemblySummary,
    constraints: &ResolvedConstraintMatrix,
    requested_backend: LinearAlgebraBackendKind,
    algebra: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
) -> LinearSolveResult {
    let dofs = summary.dof_count;
    let rows = constraints.row_count();
    if constraints.validate().is_err() || constraints.full_dof_count != dofs {
        return LinearSolveResult {
            iterations: 0,
            residual_norm: f64::INFINITY,
            converged: false,
            host_sync_count: 0,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            accelerator_evidence: None,
            solution: vec![0.0; dofs],
            solver_method: SolverMethod::MatrixFreeSaddlePointGmres,
            preconditioner: SolverPreconditioner::None,
            diagnostics: vec![FeaDiagnostic {
                code: "FEA_INVALID_MULTIPLIER_CONSTRAINT_SYSTEM".into(),
                severity: FeaDiagnosticSeverity::Error,
                message:
                    "multiplier constraint dimensions or canonical sparse evidence are invalid"
                        .into(),
            }],
        };
    }
    let reduction = match constraints.affine_reduction() {
        Ok(reduction) => reduction,
        Err(error) => return rejected_multiplier_system(summary, &error.to_string()),
    };
    if reduction.reduced_dof_count > 0 {
        let (reduced_operator, reduced_rhs) = match reduction.reduce_sparse_system(
            summary.operator.operators().stiffness(),
            summary.operator.load(),
        ) {
            Ok(reduced) => reduced,
            Err(error) => return rejected_multiplier_system(summary, &error.to_string()),
        };
        if let Err(error) = PreparedRealLinearSystem::try_new(
            reduced_operator,
            reduced_rhs,
            vec![false; reduction.reduced_dof_count],
        ) {
            return rejected_multiplier_system(summary, &error.to_string());
        }
    }
    let equation_scale = representative_stiffness(summary);
    let mut right_hand_side = summary.operator.load().to_vec();
    right_hand_side.extend(
        constraints
            .right_hand_side
            .iter()
            .map(|value| value * equation_scale),
    );
    let mut initial = vec![0.0; dofs + rows];
    if let Some(initial_guess) = initial_guess {
        for (target, value) in initial[..dofs].iter_mut().zip(initial_guess) {
            *target = *value;
        }
    }
    let apply = |value: &[f64]| apply_saddle(summary, constraints, value, equation_scale);
    let tolerance = 1.0e-8;
    let restart = (dofs + rows).min(32);
    let maximum_iterations = (dofs + rows).saturating_mul(8).max(256);
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_MULTIPLIER_CONSTRAINT_SYSTEM".into(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "physical_dofs={dofs} multiplier_dofs={rows} equation_scale={equation_scale}"
        ),
    }];
    if requested_backend == LinearAlgebraBackendKind::Accelerator {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_MULTIPLIER_CPU_REFERENCE".into(),
            severity: FeaDiagnosticSeverity::Warning,
            message:
                "multiplier constraint enforcement is executing on the serial CPU reference backend"
                    .into(),
        });
    }

    let result = restarted_gmres(
        &right_hand_side,
        &initial,
        GmresPolicy {
            relative_tolerance: tolerance,
            restart,
            maximum_iterations,
        },
        apply,
        |left, right| algebra.dot(left, right),
        crate::is_fea_cancelled,
    );
    let Some(result) = result else {
        return rejected_multiplier_system(summary, "matrix-free GMRES inputs are invalid");
    };
    if result.cancelled {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_SOLVER_CANCELLED".into(),
            severity: FeaDiagnosticSeverity::Warning,
            message: "saddle-point solve cancelled".into(),
        });
    }
    if result.breakdown {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_SADDLE_POINT_BREAKDOWN".into(),
            severity: FeaDiagnosticSeverity::Warning,
            message: "GMRES Hessenberg factor lost numerical rank".into(),
        });
    }
    diagnostics.push(FeaDiagnostic {
        code: "FEA_SOLVER_METHOD".into(),
        severity: FeaDiagnosticSeverity::Info,
        message: "solver=gmres space=saddle_point preconditioner=none matrix_free=true".into(),
    });
    LinearSolveResult {
        iterations: result.iterations,
        residual_norm: result.residual_norm,
        converged: result.converged,
        host_sync_count: 0,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution: result.solution[..dofs].to_vec(),
        solver_method: SolverMethod::MatrixFreeSaddlePointGmres,
        preconditioner: SolverPreconditioner::None,
        diagnostics,
    }
}

fn rejected_multiplier_system(summary: &AssemblySummary, message: &str) -> LinearSolveResult {
    LinearSolveResult {
        iterations: 0,
        residual_norm: summary
            .operator
            .load()
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt(),
        converged: false,
        host_sync_count: 0,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution: vec![0.0; summary.dof_count],
        solver_method: SolverMethod::MatrixFreeSaddlePointGmres,
        preconditioner: SolverPreconditioner::None,
        diagnostics: vec![FeaDiagnostic {
            code: "FEA_INVALID_MULTIPLIER_SOLVER_ADMISSION".into(),
            severity: FeaDiagnosticSeverity::Error,
            message: message.into(),
        }],
    }
}

fn apply_saddle(
    summary: &AssemblySummary,
    constraints: &ResolvedConstraintMatrix,
    value: &[f64],
    equation_scale: f64,
) -> Vec<f64> {
    let dofs = summary.dof_count;
    let mut physical = apply_k(&summary.operator, &value[..dofs]);
    let transpose = constraints.transpose_apply_validated(&value[dofs..]);
    for (target, contribution) in physical.iter_mut().zip(transpose) {
        *target += equation_scale * contribution;
    }
    physical.extend(
        constraints
            .apply_validated(&value[..dofs])
            .into_iter()
            .map(|entry| entry * equation_scale),
    );
    physical
}

fn representative_stiffness(summary: &AssemblySummary) -> f64 {
    summary
        .operator
        .operators()
        .stiffness()
        .diagonal_values()
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold(1.0_f64, f64::max)
}
