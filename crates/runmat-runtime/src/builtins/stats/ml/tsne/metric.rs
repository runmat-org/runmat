use super::{invalid, DistanceMetric};
use crate::BuiltinResult;
use nalgebra::DMatrix;
use std::cmp::Ordering;

pub(super) async fn pairwise_squared_distances_cooperative(
    rows: &[Vec<f64>],
    metric: DistanceMetric,
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<Vec<f64>> {
    let n = rows.len();
    let cells = n
        .checked_mul(n)
        .ok_or_else(|| invalid("tsne: pairwise distance shape overflow"))?;
    let mut data = Vec::new();
    data.try_reserve_exact(cells).map_err(|source| {
        super::internal_with_source("tsne: pairwise distances exceed available memory", source)
    })?;
    for _ in 0..cells {
        data.push(0.0);
        cooperation.completed(1).await?;
    }
    let prepared = match metric {
        DistanceMetric::Euclidean => PreparedMetric::Euclidean,
        DistanceMetric::Seuclidean => {
            PreparedMetric::Seuclidean(column_std(rows, cooperation).await?)
        }
        DistanceMetric::Cityblock => PreparedMetric::Cityblock,
        DistanceMetric::Chebychev => PreparedMetric::Chebychev,
        DistanceMetric::Minkowski => PreparedMetric::Minkowski,
        DistanceMetric::Mahalanobis => {
            PreparedMetric::Mahalanobis(inverse_covariance(rows, cooperation).await?)
        }
        DistanceMetric::Cosine => PreparedMetric::Cosine,
        DistanceMetric::Correlation => PreparedMetric::Correlation,
        DistanceMetric::Spearman => PreparedMetric::Spearman(row_ranks(rows, cooperation).await?),
        DistanceMetric::Hamming => PreparedMetric::Hamming,
        DistanceMetric::Jaccard => PreparedMetric::Jaccard,
    };
    for i in 0..n {
        for j in (i + 1)..n {
            let value = match &prepared {
                PreparedMetric::Euclidean | PreparedMetric::Minkowski => {
                    euclidean_sq(&rows[i], &rows[j], cooperation).await?
                }
                PreparedMetric::Seuclidean(scale) => {
                    seuclidean_sq(&rows[i], &rows[j], scale, cooperation).await?
                }
                PreparedMetric::Cityblock => {
                    cityblock(&rows[i], &rows[j], cooperation).await?.powi(2)
                }
                PreparedMetric::Chebychev => {
                    chebychev(&rows[i], &rows[j], cooperation).await?.powi(2)
                }
                PreparedMetric::Mahalanobis(inverse) => {
                    mahalanobis_sq(&rows[i], &rows[j], inverse, cooperation).await?
                }
                PreparedMetric::Cosine => cosine_distance(&rows[i], &rows[j], cooperation)
                    .await?
                    .powi(2),
                PreparedMetric::Correlation => {
                    correlation_distance(&rows[i], &rows[j], cooperation)
                        .await?
                        .powi(2)
                }
                PreparedMetric::Spearman(ranks) => {
                    correlation_distance(&ranks[i], &ranks[j], cooperation)
                        .await?
                        .powi(2)
                }
                PreparedMetric::Hamming => hamming_distance(&rows[i], &rows[j], cooperation)
                    .await?
                    .powi(2),
                PreparedMetric::Jaccard => jaccard_distance(&rows[i], &rows[j], cooperation)
                    .await?
                    .powi(2),
            };
            data[i + j * n] = value;
            data[j + i * n] = value;
            cooperation.completed(1).await?;
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

async fn euclidean_sq(
    left: &[f64],
    right: &[f64],
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<f64> {
    let mut total = 0.0;
    for (a, b) in left.iter().zip(right) {
        let diff = a - b;
        total += diff * diff;
        cooperation.completed(1).await?;
    }
    Ok(total)
}

async fn seuclidean_sq(
    left: &[f64],
    right: &[f64],
    scale: &[f64],
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<f64> {
    let mut total = 0.0;
    for ((a, b), s) in left.iter().zip(right).zip(scale) {
        let diff = if *s > 0.0 { (a - b) / s } else { 0.0 };
        total += diff * diff;
        cooperation.completed(1).await?;
    }
    Ok(total)
}

async fn cityblock(
    left: &[f64],
    right: &[f64],
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<f64> {
    let mut total = 0.0;
    for (a, b) in left.iter().zip(right) {
        total += (a - b).abs();
        cooperation.completed(1).await?;
    }
    Ok(total)
}

async fn chebychev(
    left: &[f64],
    right: &[f64],
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<f64> {
    let mut maximum: f64 = 0.0;
    for (a, b) in left.iter().zip(right) {
        maximum = maximum.max((a - b).abs());
        cooperation.completed(1).await?;
    }
    Ok(maximum)
}

async fn cosine_distance(
    left: &[f64],
    right: &[f64],
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<f64> {
    directional_distance(left, right, false, cooperation).await
}

async fn correlation_distance(
    left: &[f64],
    right: &[f64],
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<f64> {
    directional_distance(left, right, true, cooperation).await
}

async fn directional_distance(
    left: &[f64],
    right: &[f64],
    center: bool,
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<f64> {
    let Some(left) = cooperative_direction(left, center, cooperation).await? else {
        return Ok(1.0);
    };
    let Some(right) = cooperative_direction(right, center, cooperation).await? else {
        return Ok(1.0);
    };
    let mut dot = 0.0;
    for (a, b) in left.iter().zip(&right) {
        dot += a * b;
        cooperation.completed(1).await?;
    }
    Ok((1.0 - dot).max(0.0))
}

async fn cooperative_direction(
    row: &[f64],
    center: bool,
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    use super::super::pairwise_metric::{DirectionError, DirectionProgress, DirectionWork};
    let mut work = DirectionWork::new(row, center);
    loop {
        match work.advance().map_err(|error| match error {
            DirectionError::Resource(source) => {
                super::internal_with_source("tsne: direction exceeds available memory", source)
            }
        })? {
            DirectionProgress::Pending(units) => cooperation.completed(units).await?,
            DirectionProgress::Complete(direction, units) => {
                cooperation.completed(units).await?;
                return Ok(direction);
            }
        }
    }
}

async fn hamming_distance(
    left: &[f64],
    right: &[f64],
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<f64> {
    let mut mismatches = 0usize;
    for (a, b) in left.iter().zip(right) {
        if (*a - *b).abs() > 0.0 {
            mismatches += 1;
        }
        cooperation.completed(1).await?;
    }
    Ok(mismatches as f64 / left.len().max(1) as f64)
}

async fn jaccard_distance(
    left: &[f64],
    right: &[f64],
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<f64> {
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
        cooperation.completed(1).await?;
    }
    if union == 0 {
        Ok(0.0)
    } else {
        Ok(1.0 - intersection as f64 / union as f64)
    }
}

async fn column_std(
    rows: &[Vec<f64>],
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<Vec<f64>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let cols = rows[0].len();
    let mut result = Vec::new();
    result.try_reserve_exact(cols).map_err(|source| {
        super::internal_with_source(
            "tsne: standardized-distance scales exceed available memory",
            source,
        )
    })?;
    for col in 0..cols {
        let mut total = 0.0;
        for row in rows {
            total += row[col];
            cooperation.completed(1).await?;
        }
        let mean = total / rows.len() as f64;
        let mut squared = 0.0;
        for row in rows {
            let diff = row[col] - mean;
            squared += diff * diff;
            cooperation.completed(1).await?;
        }
        let var = squared / rows.len().saturating_sub(1).max(1) as f64;
        result.push(var.sqrt());
    }
    Ok(result)
}

async fn inverse_covariance(
    rows: &[Vec<f64>],
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<DMatrix<f64>> {
    if rows.is_empty() {
        return Ok(DMatrix::identity(0, 0));
    }
    let p = rows[0].len();
    let mut cov: DMatrix<f64> = super::workspace::zero_matrix(p, p, cooperation).await?;
    let mut means = Vec::new();
    means.try_reserve_exact(p).map_err(|source| {
        super::internal_with_source("tsne: covariance means exceed available memory", source)
    })?;
    for col in 0..p {
        let mut total = 0.0;
        for row in rows {
            total += row[col];
            cooperation.completed(1).await?;
        }
        means.push(total / rows.len() as f64);
    }
    let denom = rows.len().saturating_sub(1).max(1) as f64;
    for row in rows {
        for a in 0..p {
            for b in a..p {
                cov[(a, b)] += (row[a] - means[a]) * (row[b] - means[b]) / denom;
                cooperation.completed(1).await?;
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            cov[(a, b)] = cov[(b, a)];
            cooperation.completed(1).await?;
        }
    }
    for idx in 0..p {
        cov[(idx, idx)] += 1.0e-10;
        cooperation.completed(1).await?;
    }
    cov.try_inverse()
        .ok_or_else(|| invalid("tsne: Mahalanobis covariance is singular"))
}

async fn mahalanobis_sq(
    left: &[f64],
    right: &[f64],
    inv_cov: &DMatrix<f64>,
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<f64> {
    let mut diff = Vec::new();
    diff.try_reserve_exact(left.len()).map_err(|source| {
        super::internal_with_source(
            "tsne: Mahalanobis workspace exceeds available memory",
            source,
        )
    })?;
    for (a, b) in left.iter().zip(right) {
        diff.push(a - b);
        cooperation.completed(1).await?;
    }
    let mut value = 0.0;
    for i in 0..diff.len() {
        for j in 0..diff.len() {
            value += diff[i] * inv_cov[(i, j)] * diff[j];
            cooperation.completed(1).await?;
        }
    }
    Ok(value.max(0.0))
}

async fn row_ranks(
    rows: &[Vec<f64>],
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<Vec<Vec<f64>>> {
    let mut ranks = Vec::new();
    ranks.try_reserve_exact(rows.len()).map_err(|source| {
        super::internal_with_source("tsne: rank rows exceed available memory", source)
    })?;
    for row in rows {
        ranks.push(tied_ranks(row, cooperation).await?);
        cooperation.completed(1).await?;
    }
    Ok(ranks)
}

async fn tied_ranks(
    values: &[f64],
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut order = Vec::new();
    order.try_reserve_exact(values.len()).map_err(|source| {
        super::internal_with_source("tsne: rank ordering exceeds available memory", source)
    })?;
    for (index, value) in values.iter().copied().enumerate() {
        order.push((index, value));
        cooperation.completed(1).await?;
    }
    cooperation
        .stable_sort_by(&mut order, |left, right| {
            left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal)
        })
        .await?;
    let mut ranks = Vec::new();
    ranks.try_reserve_exact(values.len()).map_err(|source| {
        super::internal_with_source("tsne: ranks exceed available memory", source)
    })?;
    for _ in values {
        ranks.push(0.0);
        cooperation.completed(1).await?;
    }
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && order[end].1 == order[start].1 {
            end += 1;
            cooperation.completed(1).await?;
        }
        let rank = (start + 1 + end) as f64 / 2.0;
        for idx in start..end {
            ranks[order[idx].0] = rank;
            cooperation.completed(1).await?;
        }
        start = end;
    }
    Ok(ranks)
}

#[cfg(test)]
mod directional_tests;
