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
        diagnostics.push(modal_cluster_diagnostic(eigenvalues_hz));
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
        let sep = relative_frequency_separation(window[0], window[1]);
        min_sep = min_sep.min(sep);
    }
    min_sep
}

fn modal_cluster_diagnostic(freqs: &[f64]) -> FeaDiagnostic {
    let metrics = modal_cluster_metrics(freqs);
    FeaDiagnostic {
        code: "FEA_MODAL_CLUSTER".to_string(),
        severity: if metrics.cluster_coverage_ratio >= 1.0 {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "adjacent_mode_pair_count={} repeated_mode_pair_count={} near_repeated_mode_pair_count={} well_separated_mode_pair_count={} min_relative_frequency_separation={} cluster_coverage_ratio={}",
            metrics.adjacent_mode_pair_count,
            metrics.repeated_mode_pair_count,
            metrics.near_repeated_mode_pair_count,
            metrics.well_separated_mode_pair_count,
            metrics.min_relative_frequency_separation,
            metrics.cluster_coverage_ratio,
        ),
    }
}

#[derive(Debug, Clone, Copy)]
struct ModalClusterMetrics {
    adjacent_mode_pair_count: usize,
    repeated_mode_pair_count: usize,
    near_repeated_mode_pair_count: usize,
    well_separated_mode_pair_count: usize,
    min_relative_frequency_separation: f64,
    cluster_coverage_ratio: f64,
}

fn modal_cluster_metrics(freqs: &[f64]) -> ModalClusterMetrics {
    if freqs.len() < 2 {
        return ModalClusterMetrics {
            adjacent_mode_pair_count: 0,
            repeated_mode_pair_count: 0,
            near_repeated_mode_pair_count: 0,
            well_separated_mode_pair_count: 0,
            min_relative_frequency_separation: 1.0,
            cluster_coverage_ratio: 1.0,
        };
    }

    let mut repeated_mode_pair_count = 0usize;
    let mut near_repeated_mode_pair_count = 0usize;
    let mut well_separated_mode_pair_count = 0usize;
    let mut min_relative_frequency_separation = f64::INFINITY;
    for window in freqs.windows(2) {
        let separation = relative_frequency_separation(window[0], window[1]);
        min_relative_frequency_separation = min_relative_frequency_separation.min(separation);
        if separation <= 1.0e-6 {
            repeated_mode_pair_count += 1;
        } else if separation <= 1.0e-3 {
            near_repeated_mode_pair_count += 1;
        } else {
            well_separated_mode_pair_count += 1;
        }
    }
    let adjacent_mode_pair_count = freqs.len() - 1;
    let classified_count =
        repeated_mode_pair_count + near_repeated_mode_pair_count + well_separated_mode_pair_count;
    let cluster_coverage_ratio = classified_count as f64 / adjacent_mode_pair_count as f64;

    ModalClusterMetrics {
        adjacent_mode_pair_count,
        repeated_mode_pair_count,
        near_repeated_mode_pair_count,
        well_separated_mode_pair_count,
        min_relative_frequency_separation,
        cluster_coverage_ratio,
    }
}

fn relative_frequency_separation(a: f64, b: f64) -> f64 {
    let a = a.abs().max(1.0e-12);
    let b = b.abs().max(1.0e-12);
    (b - a).abs() / a.max(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modal_cluster_metrics_classify_repeated_and_near_repeated_pairs() {
        let metrics = modal_cluster_metrics(&[10.0, 10.0 + 1.0e-8, 10.005, 12.0]);

        assert_eq!(metrics.adjacent_mode_pair_count, 3);
        assert_eq!(metrics.repeated_mode_pair_count, 1);
        assert_eq!(metrics.near_repeated_mode_pair_count, 1);
        assert_eq!(metrics.well_separated_mode_pair_count, 1);
        assert_eq!(metrics.cluster_coverage_ratio, 1.0);
        assert!(metrics.min_relative_frequency_separation <= 1.0e-6);
    }
}
