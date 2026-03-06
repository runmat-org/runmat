use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalSolveResult {
    pub converged: bool,
    pub eigenvalues_hz: Vec<f64>,
    pub mode_shapes: Vec<Vec<f64>>,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub solver_method: String,
}

pub fn solve_modal_system(summary: &AssemblySummary, mode_count: usize) -> ModalSolveResult {
    if summary.dof_count == 0 || mode_count == 0 {
        return ModalSolveResult {
            converged: false,
            eigenvalues_hz: Vec::new(),
            mode_shapes: Vec::new(),
            diagnostics: vec![FeaDiagnostic {
                code: "FEA_MODAL_EMPTY_SYSTEM".to_string(),
                severity: FeaDiagnosticSeverity::Warning,
                message: "modal solve skipped because assembled system has zero DOFs or requested mode_count is zero"
                    .to_string(),
            }],
            solver_method: "diag_generalized_eigen".to_string(),
        };
    }

    let mut candidates = Vec::new();
    for i in 0..summary.operator.dof_count {
        if summary.operator.constrained[i] {
            continue;
        }
        let k = summary.operator.stiffness_diag[i].max(1.0e-12);
        let m = summary.operator.mass_diag[i].max(1.0e-12);
        let omega = (k / m).sqrt();
        let freq_hz = omega / (2.0 * std::f64::consts::PI);
        candidates.push((i, freq_hz));
    }

    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let selected: Vec<(usize, f64)> = candidates.into_iter().take(mode_count).collect();

    let mut eigenvalues_hz = Vec::with_capacity(selected.len());
    let mut mode_shapes = Vec::with_capacity(selected.len());
    for (mode_idx, (dof_index, freq_hz)) in selected.into_iter().enumerate() {
        eigenvalues_hz.push(freq_hz);

        let mut shape = vec![0.0; summary.operator.dof_count];
        shape[dof_index] = 1.0;
        if dof_index > 0 && !summary.operator.constrained[dof_index - 1] {
            shape[dof_index - 1] = -0.25;
        }
        if dof_index + 1 < shape.len() && !summary.operator.constrained[dof_index + 1] {
            shape[dof_index + 1] = 0.25;
        }
        let norm = shape.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0e-12);
        for value in &mut shape {
            *value /= norm;
        }
        let _ = mode_idx;
        mode_shapes.push(shape);
    }

    let converged = !eigenvalues_hz.is_empty();
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_MODAL_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: "solver=diag_generalized_eigen matrix_free=false".to_string(),
    }];
    diagnostics.push(FeaDiagnostic {
        code: "FEA_MODAL_CONVERGENCE".to_string(),
        severity: if converged {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "mode_count_requested={} mode_count_solved={} converged={}",
            mode_count,
            eigenvalues_hz.len(),
            converged
        ),
    });

    ModalSolveResult {
        converged,
        eigenvalues_hz,
        mode_shapes,
        diagnostics,
        solver_method: "diag_generalized_eigen".to_string(),
    }
}
