use crate::{
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    operator::{apply_m, OperatorSystem},
};

use super::math::dot;

pub(super) fn push_modal_quality_diagnostics(
    diagnostics: &mut Vec<FeaDiagnostic>,
    system: &OperatorSystem,
    eigenvalues_hz: &[f64],
    mode_shapes: &[Vec<f64>],
    residual_norms: &[f64],
) {
    if !residual_norms.is_empty() {
        let max_residual = residual_norms.iter().copied().fold(0.0_f64, f64::max);
        diagnostics.push(FeaDiagnostic {
            code: "FEA_MODAL_RESIDUAL".to_string(),
            severity: if max_residual <= 1.0e-3 {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!("max_modal_residual_norm={max_residual}"),
        });
    }

    if mode_shapes.len() >= 2 {
        let max_offdiag = modal_max_m_orthogonality_offdiag(system, mode_shapes);
        diagnostics.push(FeaDiagnostic {
            code: "FEA_MODAL_ORTHOGONALITY".to_string(),
            severity: if max_offdiag <= 1.0e-3 {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!("max_m_orthogonality_offdiag={max_offdiag}"),
        });

        let min_separation = modal_min_frequency_separation(eigenvalues_hz);
        diagnostics.push(FeaDiagnostic {
            code: "FEA_MODAL_SEPARATION".to_string(),
            severity: if min_separation >= 1.0e-3 {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!("min_relative_frequency_separation={min_separation}"),
        });
    }
}

fn modal_max_m_orthogonality_offdiag(system: &OperatorSystem, modes: &[Vec<f64>]) -> f64 {
    let mut max_offdiag = 0.0_f64;
    for i in 0..modes.len() {
        for j in 0..modes.len() {
            if i == j {
                continue;
            }
            let mphi = apply_m(system, &modes[j]);
            let value = dot(&modes[i], &mphi).abs();
            if value > max_offdiag {
                max_offdiag = value;
            }
        }
    }
    max_offdiag
}

fn modal_min_frequency_separation(freqs: &[f64]) -> f64 {
    if freqs.len() < 2 {
        return 1.0;
    }
    let mut min_sep = f64::INFINITY;
    for window in freqs.windows(2) {
        let a = window[0].abs().max(1.0e-12);
        let b = window[1].abs().max(1.0e-12);
        let sep = ((b - a).abs()) / a.max(b);
        min_sep = min_sep.min(sep);
    }
    min_sep
}
