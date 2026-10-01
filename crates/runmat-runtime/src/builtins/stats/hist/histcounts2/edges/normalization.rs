use super::super::options::HistogramNormalization;
use runmat_value::IntValue;

pub(in crate::builtins::stats::hist::histcounts2) fn apply_normalization_2d(
    counts: &[f64],
    x_edges: &[f64],
    y_edges: &[f64],
    x_exact_edges: Option<&[IntValue]>,
    y_exact_edges: Option<&[IntValue]>,
    mode: HistogramNormalization,
    observation_count: usize,
) -> Vec<f64> {
    let x_bins = x_edges.len() - 1;
    let y_bins = y_edges.len() - 1;
    let total = observation_count as f64;
    let x_widths = super::super::super::integer_bins::edge_widths(x_edges, x_exact_edges);
    let y_widths = super::super::super::integer_bins::edge_widths(y_edges, y_exact_edges);

    match mode {
        HistogramNormalization::Count => counts.to_vec(),
        HistogramNormalization::Probability => {
            if total > 0.0 {
                counts.iter().map(|&c| c / total).collect()
            } else {
                vec![0.0; counts.len()]
            }
        }
        HistogramNormalization::Percentage => {
            if total > 0.0 {
                counts.iter().map(|&c| 100.0 * c / total).collect()
            } else {
                vec![0.0; counts.len()]
            }
        }
        HistogramNormalization::CountDensity => {
            let mut out = vec![0.0; counts.len()];
            for (iy, y_width) in y_widths.iter().enumerate().take(y_bins) {
                for (ix, x_width) in x_widths.iter().enumerate().take(x_bins) {
                    let idx = ix + iy * x_bins;
                    let area = x_width * y_width;
                    out[idx] = if area > 0.0 { counts[idx] / area } else { 0.0 };
                }
            }
            out
        }
        HistogramNormalization::Pdf => {
            if total > 0.0 {
                let mut out = vec![0.0; counts.len()];
                for (iy, y_width) in y_widths.iter().enumerate().take(y_bins) {
                    for (ix, x_width) in x_widths.iter().enumerate().take(x_bins) {
                        let idx = ix + iy * x_bins;
                        let area = x_width * y_width;
                        out[idx] = if area > 0.0 {
                            counts[idx] / (total * area)
                        } else {
                            0.0
                        };
                    }
                }
                out
            } else {
                vec![0.0; counts.len()]
            }
        }
        HistogramNormalization::CumCount => cumulative_2d(counts, x_bins, y_bins, None),
        HistogramNormalization::Cdf => {
            if total > 0.0 {
                cumulative_2d(counts, x_bins, y_bins, Some(total))
            } else {
                vec![0.0; counts.len()]
            }
        }
    }
}

fn cumulative_2d(counts: &[f64], x_bins: usize, y_bins: usize, divisor: Option<f64>) -> Vec<f64> {
    let mut out = vec![0.0; counts.len()];
    for y in 0..y_bins {
        for x in 0..x_bins {
            let idx = x + y * x_bins;
            let left = if x > 0 { out[idx - 1] } else { 0.0 };
            let below = if y > 0 { out[idx - x_bins] } else { 0.0 };
            let diagonal = if x > 0 && y > 0 {
                out[idx - x_bins - 1]
            } else {
                0.0
            };
            out[idx] = counts[idx] + left + below - diagonal;
        }
    }
    if let Some(divisor) = divisor {
        for value in &mut out {
            *value /= divisor;
        }
    }
    out
}
