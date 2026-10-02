use super::{invalid, DistanceMetric};
use crate::BuiltinResult;
use nalgebra::DMatrix;
use std::cmp::Ordering;

pub(super) fn pairwise_squared_distances(
    rows: &[Vec<f64>],
    metric: DistanceMetric,
) -> BuiltinResult<Vec<f64>> {
    let n = rows.len();
    let cells = n
        .checked_mul(n)
        .ok_or_else(|| invalid("tsne: pairwise distance shape overflow"))?;
    let mut data = vec![0.0; cells];
    let prepared = match metric {
        DistanceMetric::Euclidean => PreparedMetric::Euclidean,
        DistanceMetric::Seuclidean => PreparedMetric::Seuclidean(column_std(rows)),
        DistanceMetric::Cityblock => PreparedMetric::Cityblock,
        DistanceMetric::Chebychev => PreparedMetric::Chebychev,
        DistanceMetric::Minkowski => PreparedMetric::Minkowski,
        DistanceMetric::Mahalanobis => PreparedMetric::Mahalanobis(inverse_covariance(rows)?),
        DistanceMetric::Cosine => PreparedMetric::Cosine,
        DistanceMetric::Correlation => PreparedMetric::Correlation,
        DistanceMetric::Spearman => PreparedMetric::Spearman(row_ranks(rows)),
        DistanceMetric::Hamming => PreparedMetric::Hamming,
        DistanceMetric::Jaccard => PreparedMetric::Jaccard,
    };
    for i in 0..n {
        for j in (i + 1)..n {
            let value = match &prepared {
                PreparedMetric::Euclidean | PreparedMetric::Minkowski => {
                    euclidean_sq(&rows[i], &rows[j])
                }
                PreparedMetric::Seuclidean(scale) => seuclidean_sq(&rows[i], &rows[j], scale),
                PreparedMetric::Cityblock => cityblock(&rows[i], &rows[j]).powi(2),
                PreparedMetric::Chebychev => chebychev(&rows[i], &rows[j]).powi(2),
                PreparedMetric::Mahalanobis(inverse) => mahalanobis_sq(&rows[i], &rows[j], inverse),
                PreparedMetric::Cosine => cosine_distance(&rows[i], &rows[j])?.powi(2),
                PreparedMetric::Correlation => correlation_distance(&rows[i], &rows[j])?.powi(2),
                PreparedMetric::Spearman(ranks) => {
                    correlation_distance(&ranks[i], &ranks[j])?.powi(2)
                }
                PreparedMetric::Hamming => hamming_distance(&rows[i], &rows[j]).powi(2),
                PreparedMetric::Jaccard => jaccard_distance(&rows[i], &rows[j]).powi(2),
            };
            data[i + j * n] = value;
            data[j + i * n] = value;
        }
    }
    Ok(data)
}

enum PreparedMetric {
    Euclidean,
    Seuclidean(Vec<f64>),
    Cityblock,
    Chebychev,
    Minkowski,
    Mahalanobis(DMatrix<f64>),
    Cosine,
    Correlation,
    Spearman(Vec<Vec<f64>>),
    Hamming,
    Jaccard,
}

fn euclidean_sq(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum()
}

fn seuclidean_sq(left: &[f64], right: &[f64], scale: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .zip(scale.iter())
        .map(|((a, b), s)| {
            let diff = if *s > 0.0 { (a - b) / s } else { 0.0 };
            diff * diff
        })
        .sum()
}

fn cityblock(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .sum()
}

fn chebychev(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn cosine_distance(left: &[f64], right: &[f64]) -> BuiltinResult<f64> {
    directional_distance(left, right, false)
}

fn correlation_distance(left: &[f64], right: &[f64]) -> BuiltinResult<f64> {
    directional_distance(left, right, true)
}

fn directional_distance(left: &[f64], right: &[f64], center: bool) -> BuiltinResult<f64> {
    use super::super::pairwise_metric::DirectionError;
    let Some(left) = super::super::pairwise_metric::unit_direction(left, center).map_err(
        |error| match error {
            DirectionError::Resource => invalid("tsne: direction exceeds available memory"),
        },
    )?
    else {
        return Ok(1.0);
    };
    let Some(right) = super::super::pairwise_metric::unit_direction(right, center).map_err(
        |error| match error {
            DirectionError::Resource => invalid("tsne: direction exceeds available memory"),
        },
    )?
    else {
        return Ok(1.0);
    };
    let dot = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>();
    Ok((1.0 - dot).max(0.0))
}

fn hamming_distance(left: &[f64], right: &[f64]) -> f64 {
    let mismatches = left
        .iter()
        .zip(right.iter())
        .filter(|(a, b)| (*a - *b).abs() > 0.0)
        .count();
    mismatches as f64 / left.len().max(1) as f64
}

fn jaccard_distance(left: &[f64], right: &[f64]) -> f64 {
    let mut union = 0usize;
    let mut intersection = 0usize;
    for (a, b) in left.iter().zip(right.iter()) {
        let an = *a != 0.0;
        let bn = *b != 0.0;
        if an || bn {
            union += 1;
            if an && bn {
                intersection += 1;
            }
        }
    }
    if union == 0 {
        0.0
    } else {
        1.0 - intersection as f64 / union as f64
    }
}

fn column_std(rows: &[Vec<f64>]) -> Vec<f64> {
    if rows.is_empty() {
        return Vec::new();
    }
    let cols = rows[0].len();
    (0..cols)
        .map(|col| {
            let mean = rows.iter().map(|row| row[col]).sum::<f64>() / rows.len() as f64;
            let var = rows
                .iter()
                .map(|row| {
                    let diff = row[col] - mean;
                    diff * diff
                })
                .sum::<f64>()
                / rows.len().saturating_sub(1).max(1) as f64;
            var.sqrt()
        })
        .collect()
}

fn inverse_covariance(rows: &[Vec<f64>]) -> BuiltinResult<DMatrix<f64>> {
    if rows.is_empty() {
        return Ok(DMatrix::identity(0, 0));
    }
    let p = rows[0].len();
    let mut cov: DMatrix<f64> = DMatrix::zeros(p, p);
    let means = (0..p)
        .map(|col| rows.iter().map(|row| row[col]).sum::<f64>() / rows.len() as f64)
        .collect::<Vec<_>>();
    let denom = rows.len().saturating_sub(1).max(1) as f64;
    for row in rows {
        for a in 0..p {
            for b in a..p {
                cov[(a, b)] += (row[a] - means[a]) * (row[b] - means[b]) / denom;
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            cov[(a, b)] = cov[(b, a)];
        }
    }
    for idx in 0..p {
        cov[(idx, idx)] += 1.0e-10;
    }
    cov.try_inverse()
        .ok_or_else(|| invalid("tsne: Mahalanobis covariance is singular"))
}

fn mahalanobis_sq(left: &[f64], right: &[f64], inv_cov: &DMatrix<f64>) -> f64 {
    let diff = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    let mut value = 0.0;
    for i in 0..diff.len() {
        for j in 0..diff.len() {
            value += diff[i] * inv_cov[(i, j)] * diff[j];
        }
    }
    value.max(0.0)
}

fn row_ranks(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    rows.iter().map(|row| tied_ranks(row)).collect()
}

fn tied_ranks(values: &[f64]) -> Vec<f64> {
    let mut order = values.iter().copied().enumerate().collect::<Vec<_>>();
    order.sort_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && order[end].1 == order[start].1 {
            end += 1;
        }
        let rank = (start + 1 + end) as f64 / 2.0;
        for idx in start..end {
            ranks[order[idx].0] = rank;
        }
        start = end;
    }
    ranks
}

#[cfg(test)]
mod directional_tests;
