//! Exact geometry for the compatibility-gated integer center-vector form.
//!
//! Twice every center, midpoint and extrapolated edge fits in i128 for all
//! eight fixed-width classes. Counts never depend on rounded absolute doubles;
//! only the declared Double centers and chart widths cross that boundary.
use std::cmp::Ordering;

use runmat_value::IntegerStorage;

use super::{
    apply_normalization, build_hist_cpu_chart, hist_err, HistComputation, HistNormalization,
    RealizedBins, Tensor,
};
use crate::BuiltinResult;

#[derive(Clone)]
pub(super) struct IntegerCenterBins {
    centers: Vec<i128>,
    doubled_edges: Vec<i128>,
}

impl IntegerCenterBins {
    pub(super) fn new(storage: &IntegerStorage) -> BuiltinResult<Self> {
        let centers: Vec<_> = (0..storage.len())
            .map(|index| {
                storage
                    .value_at(index)
                    .expect("valid integer index")
                    .to_i128()
            })
            .collect();
        if centers.len() < 2 || centers.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(hist_err("hist: values must be strictly increasing"));
        }
        let spacing = centers[1] - centers[0];
        let tolerance = spacing as f64 * 1e-6;
        if centers
            .windows(2)
            .any(|pair| ((pair[1] - pair[0]) - spacing).abs() as f64 > tolerance)
        {
            return Err(hist_err("hist: bin centers must be evenly spaced"));
        }
        let mut doubled_edges = Vec::with_capacity(centers.len() + 1);
        doubled_edges.push(3 * centers[0] - centers[1]);
        doubled_edges.extend(centers.windows(2).map(|pair| pair[0] + pair[1]));
        let last = centers.len() - 1;
        doubled_edges.push(3 * centers[last] - centers[last - 1]);
        Ok(Self {
            centers,
            doubled_edges,
        })
    }

    pub(super) fn realize(&self) -> RealizedBins {
        RealizedBins {
            // This display geometry is never used for count classification.
            edges: self
                .doubled_edges
                .iter()
                .map(|&edge| edge as f64 * 0.5)
                .collect(),
            widths: self
                .doubled_edges
                .windows(2)
                .map(|pair| (pair[1] - pair[0]) as f64 * 0.5)
                .collect(),
            labels: self
                .doubled_edges
                .windows(2)
                .map(|pair| format!("[{}, {})", half_label(pair[0]), half_label(pair[1])))
                .collect(),
            centers: self.centers.iter().map(|&center| center as f64).collect(),
        }
    }

    fn integer_bin(&self, value: i128) -> usize {
        let doubled = 2 * value;
        self.doubled_edges[1..self.centers.len()].partition_point(|&edge| doubled >= edge)
    }

    fn floating_bin(&self, value: f64) -> usize {
        // Multiplication by two is exact unless it overflows; an infinity is
        // beyond every fixed-width center. NaNs retain legacy hist's last bin.
        let doubled = 2.0 * value;
        if doubled.is_nan() {
            return self.centers.len() - 1;
        }
        self.doubled_edges[1..self.centers.len()].partition_point(|&edge| {
            let whole = doubled.trunc() as i128;
            match whole.cmp(&edge) {
                Ordering::Equal => doubled.fract() >= 0.0,
                Ordering::Greater => true,
                Ordering::Less => false,
            }
        })
    }

    pub(super) fn evaluate(
        &self,
        tensor: &Tensor,
        normalization: HistNormalization,
        weights: Option<&[f64]>,
        total_weight: f64,
    ) -> BuiltinResult<HistComputation> {
        let shape = tensor.shape();
        let rows = shape.first().copied().unwrap_or(tensor.len());
        let columns = if shape.len() >= 2 {
            shape[1..].iter().product()
        } else {
            1
        };
        let matrix = columns > 1 && !(shape.len() == 2 && rows == 1);
        let columns = if matrix { columns } else { 1 };
        let rows = if matrix { rows } else { tensor.len() };
        let bins = self.realize();
        let mut all_counts = Vec::with_capacity(self.centers.len() * columns);
        let mut all_centers = Vec::with_capacity(self.centers.len() * columns);
        let mut charts = Vec::with_capacity(columns);
        for column in 0..columns {
            let mut counts = vec![0.; self.centers.len()];
            let mut column_weight = 0.;
            for row in 0..rows {
                let index = column * rows + row;
                let bin = if let Some(storage) = tensor.integer_storage() {
                    self.integer_bin(
                        storage
                            .value_at(index)
                            .expect("valid integer index")
                            .to_i128(),
                    )
                } else {
                    self.floating_bin(crate::builtins::common::tensor::tensor_value_f64(
                        tensor, index,
                    ))
                };
                let weight = weights
                    .and_then(|values| values.get(index))
                    .copied()
                    .unwrap_or(1.);
                counts[bin] += weight;
                column_weight += weight;
            }
            let total = if matrix { column_weight } else { total_weight };
            apply_normalization(&mut counts, &bins.widths, normalization, total);
            let chart = build_hist_cpu_chart(&bins, counts.clone())?;
            charts.push(if matrix {
                chart.with_group(column, columns)
            } else {
                chart
            });
            all_counts.extend_from_slice(&counts);
            all_centers.extend_from_slice(&bins.centers);
        }
        Ok(HistComputation {
            counts: all_counts,
            centers: all_centers,
            output_shape: if matrix {
                vec![self.centers.len(), columns]
            } else {
                vec![1, self.centers.len()]
            },
            charts,
        })
    }
}

fn half_label(doubled: i128) -> String {
    let sign = if doubled < 0 { "-" } else { "" };
    let magnitude = doubled.abs();
    let fraction = if magnitude % 2 == 0 { "" } else { ".5" };
    format!("{sign}{}{fraction}", magnitude / 2)
}

#[cfg(test)]
#[path = "integer_centers/tests.rs"]
mod tests;
