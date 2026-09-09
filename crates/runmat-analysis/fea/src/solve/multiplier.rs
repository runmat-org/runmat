//! Exact sparse constraint enforcement through a matrix-free saddle-point system.

use crate::{
    algebra::{apply_k, PreparedRealLinearSystem},
    assembly::AssemblySummary,
    constraints::ResolvedConstraintMatrix,
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::{
        backend::{kind::LinearAlgebraBackendKind, linear_algebra::LinearAlgebraBackend},
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
    let mut solution = vec![0.0; dofs + rows];
    if let Some(initial) = initial_guess {
        for (target, value) in solution[..dofs].iter_mut().zip(initial) {
            *target = *value;
        }
    }
    let apply = |value: &[f64]| apply_saddle(summary, constraints, value, equation_scale);
    let mut residual = subtract(&right_hand_side, &apply(&solution));
    let rhs_norm = algebra
        .dot(&right_hand_side, &right_hand_side)
        .sqrt()
        .max(1.0);
    let tolerance = 1.0e-8;
    let restart = (dofs + rows).min(32);
    let maximum_iterations = (dofs + rows).saturating_mul(8).max(256);
    let mut iterations = 0_u32;
    let mut converged = algebra.dot(&residual, &residual).sqrt() / rhs_norm <= tolerance;
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_MULTIPLIER_CONSTRAINT_SYSTEM".into(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "physical_dofs={dofs} multiplier_dofs={rows} equation_scale={equation_scale}"
        ),
    }];
    if requested_backend == LinearAlgebraBackendKind::RuntimeTensor {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_MULTIPLIER_CPU_REFERENCE".into(),
            severity: FeaDiagnosticSeverity::Warning,
            message:
                "multiplier constraint enforcement is executing on the serial CPU reference backend"
                    .into(),
        });
    }

    while !converged && (iterations as usize) < maximum_iterations {
        if crate::is_fea_cancelled() {
            diagnostics.push(FeaDiagnostic {
                code: "FEA_SOLVER_CANCELLED".into(),
                severity: FeaDiagnosticSeverity::Warning,
                message: "saddle-point solve cancelled".into(),
            });
            break;
        }
        let beta = algebra.dot(&residual, &residual).sqrt();
        if !beta.is_finite() || beta <= f64::MIN_POSITIVE {
            break;
        }
        let mut basis = vec![residual
            .iter()
            .map(|value| value / beta)
            .collect::<Vec<_>>()];
        let mut hessenberg = vec![vec![0.0; restart]; restart + 1];
        let mut cosines = vec![0.0; restart];
        let mut sines = vec![0.0; restart];
        let mut projected_rhs = vec![0.0; restart + 1];
        projected_rhs[0] = beta;
        let mut completed = 0;

        for column in 0..restart {
            if (iterations as usize) >= maximum_iterations || crate::is_fea_cancelled() {
                break;
            }
            let mut next = apply(&basis[column]);
            for row in 0..=column {
                hessenberg[row][column] = algebra.dot(&basis[row], &next);
                axpy(-hessenberg[row][column], &basis[row], &mut next);
            }
            hessenberg[column + 1][column] = algebra.dot(&next, &next).sqrt();
            if hessenberg[column + 1][column] > f64::MIN_POSITIVE {
                basis.push(
                    next.iter()
                        .map(|value| value / hessenberg[column + 1][column])
                        .collect(),
                );
            }
            for row in 0..column {
                let first = hessenberg[row][column];
                let second = hessenberg[row + 1][column];
                hessenberg[row][column] = cosines[row] * first + sines[row] * second;
                hessenberg[row + 1][column] = -sines[row] * first + cosines[row] * second;
            }
            let diagonal = hessenberg[column][column];
            let subdiagonal = hessenberg[column + 1][column];
            let norm = diagonal.hypot(subdiagonal);
            if !norm.is_finite() || norm <= f64::MIN_POSITIVE {
                break;
            }
            cosines[column] = diagonal / norm;
            sines[column] = subdiagonal / norm;
            hessenberg[column][column] = norm;
            hessenberg[column + 1][column] = 0.0;
            let projected = projected_rhs[column];
            projected_rhs[column] = cosines[column] * projected;
            projected_rhs[column + 1] = -sines[column] * projected;
            iterations += 1;
            completed = column + 1;
            if projected_rhs[column + 1].abs() / rhs_norm <= tolerance || basis.len() == column + 1
            {
                break;
            }
        }
        if completed == 0 {
            break;
        }
        let Some(coefficients) = back_substitute(&hessenberg, &projected_rhs, completed) else {
            diagnostics.push(FeaDiagnostic {
                code: "FEA_SADDLE_POINT_BREAKDOWN".into(),
                severity: FeaDiagnosticSeverity::Warning,
                message: "GMRES Hessenberg factor lost numerical rank".into(),
            });
            break;
        };
        for (coefficient, direction) in coefficients.iter().zip(&basis) {
            axpy(*coefficient, direction, &mut solution);
        }
        residual = subtract(&right_hand_side, &apply(&solution));
        converged = algebra.dot(&residual, &residual).sqrt() / rhs_norm <= tolerance;
    }

    let residual_norm = algebra.dot(&residual, &residual).sqrt();
    diagnostics.push(FeaDiagnostic {
        code: "FEA_SOLVER_METHOD".into(),
        severity: FeaDiagnosticSeverity::Info,
        message: "solver=gmres space=saddle_point preconditioner=none matrix_free=true".into(),
    });
    LinearSolveResult {
        iterations,
        residual_norm,
        converged,
        host_sync_count: 0,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        solution: solution[..dofs].to_vec(),
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

fn subtract(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter().zip(right).map(|(a, b)| a - b).collect()
}

fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += alpha * x;
    }
}

fn back_substitute(matrix: &[Vec<f64>], rhs: &[f64], size: usize) -> Option<Vec<f64>> {
    let mut solution = vec![0.0; size];
    for row in (0..size).rev() {
        let remainder = ((row + 1)..size)
            .map(|column| matrix[row][column] * solution[column])
            .sum::<f64>();
        if !matrix[row][row].is_finite() || matrix[row][row].abs() <= f64::MIN_POSITIVE {
            return None;
        }
        solution[row] = (rhs[row] - remainder) / matrix[row][row];
        if !solution[row].is_finite() {
            return None;
        }
    }
    Some(solution)
}
