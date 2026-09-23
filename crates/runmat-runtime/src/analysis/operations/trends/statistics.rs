use super::AnalysisRunResult;

pub(super) fn percentile(sorted_samples: &[f64], ratio: f64) -> Option<f64> {
    if sorted_samples.is_empty() {
        return None;
    }
    let index = ((sorted_samples.len() - 1) as f64 * ratio.clamp(0.0, 1.0)).round() as usize;
    sorted_samples.get(index).copied()
}

pub(super) fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

pub(super) fn diagnostic_warning_rate(entries: &[AnalysisRunResult], code: &str) -> Option<f64> {
    let values = entries
        .iter()
        .filter_map(|run| {
            run.run
                .diagnostics
                .iter()
                .find(|diag| diag.code == code)
                .map(|diag| {
                    diag.severity
                        == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
                })
        })
        .collect::<Vec<_>>();
    if values.is_empty() {
        None
    } else {
        Some(values.iter().filter(|value| **value).count() as f64 / values.len() as f64)
    }
}
