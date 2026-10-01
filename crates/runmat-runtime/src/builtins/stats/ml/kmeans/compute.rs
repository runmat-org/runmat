use std::cmp::Ordering;

use super::{
    internal, invalid, Distance, EmptyAction, KmeansStart, Options, PreparedData, ReplicateResult,
    StartSpec, EPS, NAME,
};
use crate::builtins::common::random;
use crate::BuiltinResult;

pub(super) fn run_kmeans(
    data: &PreparedData,
    k: usize,
    options: &Options,
) -> BuiltinResult<ReplicateResult> {
    let starts = initial_centers(data, k, options)?;
    let mut best = None;
    for centers in starts {
        let result = run_replicate(data, centers, options)?;
        if best
            .as_ref()
            .map(|current: &ReplicateResult| result.objective < current.objective)
            .unwrap_or(true)
        {
            best = Some(result);
        }
    }
    best.ok_or_else(|| internal("kmeans: no replicate produced a result"))
}

fn initial_centers(
    data: &PreparedData,
    k: usize,
    options: &Options,
) -> BuiltinResult<Vec<Vec<Vec<f64>>>> {
    match &options.start {
        StartSpec::Numeric(centers) => Ok(vec![centers.clone()]),
        StartSpec::NumericPages(pages) => Ok(pages.clone()),
        _ if k > data.rows.len() && options.empty_action != EmptyAction::Drop => Err(invalid(
            "kmeans: k cannot exceed the number of complete observations unless EmptyAction is 'drop'",
        )),
        StartSpec::Named(KmeansStart::Plus) => (0..options.replicates)
            .map(|_| plus_start(data, k, options.distance))
            .collect(),
        StartSpec::Named(KmeansStart::Sample) => (0..options.replicates)
            .map(|_| sample_start(data, k))
            .collect(),
        StartSpec::Named(KmeansStart::Uniform) => (0..options.replicates)
            .map(|_| uniform_start(data, k))
            .collect(),
        StartSpec::Named(KmeansStart::Cluster) => (0..options.replicates)
            .map(|_| cluster_start(data, k, options))
            .collect(),
    }
}

fn sample_start(data: &PreparedData, k: usize) -> BuiltinResult<Vec<Vec<f64>>> {
    if k > data.rows.len() {
        return Err(invalid(
            "kmeans: Start 'sample' requires k no larger than the number of complete observations",
        ));
    }
    let mut scored = random::generate_uniform(data.rows.len(), NAME)?
        .into_iter()
        .enumerate()
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    Ok(scored
        .into_iter()
        .take(k)
        .map(|(idx, _)| data.rows[idx].clone())
        .collect())
}

fn plus_start(data: &PreparedData, k: usize, distance: Distance) -> BuiltinResult<Vec<Vec<f64>>> {
    if k > data.rows.len() {
        return Err(invalid(
            "kmeans: Start 'plus' requires k no larger than the number of complete observations",
        ));
    }
    let uniforms = random::generate_uniform(k.max(1), NAME)?;
    let first = ((uniforms[0] * data.rows.len() as f64).floor() as usize).min(data.rows.len() - 1);
    let mut centers = vec![data.rows[first].clone()];
    while centers.len() < k {
        let mut weights = Vec::with_capacity(data.rows.len());
        let mut total = 0.0;
        for row in &data.rows {
            let best = centers
                .iter()
                .map(|center| distance_value(row, center, distance))
                .fold(f64::INFINITY, f64::min);
            let weight = if best.is_finite() { best.max(0.0) } else { 0.0 };
            weights.push(weight);
            total += weight;
        }
        let next = if total <= EPS {
            first_unused_row(data, &centers, distance).unwrap_or(0)
        } else {
            let mut threshold = uniforms[centers.len()] * total;
            let mut selected = data.rows.len() - 1;
            for (idx, weight) in weights.iter().enumerate() {
                threshold -= *weight;
                if threshold <= 0.0 {
                    selected = idx;
                    break;
                }
            }
            selected
        };
        centers.push(data.rows[next].clone());
    }
    Ok(centers)
}

fn first_unused_row(
    data: &PreparedData,
    centers: &[Vec<f64>],
    distance: Distance,
) -> Option<usize> {
    data.rows.iter().position(|row| {
        centers
            .iter()
            .all(|center| distance_value(row, center, distance) > EPS)
    })
}

fn uniform_start(data: &PreparedData, k: usize) -> BuiltinResult<Vec<Vec<f64>>> {
    let cols = data.cols;
    let mut mins = vec![f64::INFINITY; cols];
    let mut maxs = vec![f64::NEG_INFINITY; cols];
    for row in &data.rows {
        for col in 0..cols {
            mins[col] = mins[col].min(row[col]);
            maxs[col] = maxs[col].max(row[col]);
        }
    }
    let uniforms = random::generate_uniform(k * cols, NAME)?;
    let mut centers = vec![vec![0.0; cols]; k];
    for center in 0..k {
        for col in 0..cols {
            let lo = mins[col];
            let hi = maxs[col];
            centers[center][col] = if hi > lo {
                lo + (hi - lo) * uniforms[center + col * k]
            } else {
                lo
            };
        }
    }
    Ok(centers)
}

fn cluster_start(data: &PreparedData, k: usize, options: &Options) -> BuiltinResult<Vec<Vec<f64>>> {
    let sample_size = data.rows.len().div_ceil(10).max(k).min(data.rows.len());
    let rows = if sample_size == data.rows.len() {
        data.rows.clone()
    } else {
        sample_start(data, sample_size)?
    };
    let subset = PreparedData {
        rows,
        original_row_count: sample_size,
        original_to_valid: (0..sample_size).map(Some).collect(),
        cols: data.cols,
    };
    if subset.rows.len() == k {
        return Ok(subset.rows);
    }
    let centers = plus_start(&subset, k, options.distance)?;
    let preliminary = Options {
        distance: options.distance,
        empty_action: EmptyAction::Singleton,
        max_iter: options.max_iter.clamp(1, 10),
        online_phase: false,
        replicates: 1,
        start: StartSpec::Numeric(centers.clone()),
    };
    run_replicate(&subset, centers, &preliminary).map(|result| result.centers)
}

fn run_replicate(
    data: &PreparedData,
    mut centers: Vec<Vec<f64>>,
    options: &Options,
) -> BuiltinResult<ReplicateResult> {
    let k = centers.len();
    let mut idx = vec![0usize; data.rows.len()];
    let iterations = if options.online_phase {
        options.max_iter.saturating_mul(2)
    } else {
        options.max_iter
    };
    for _ in 0..iterations {
        let assignment = assign_rows(&data.rows, &centers, options.distance);
        let changed = assignment.changed_from(&idx);
        idx = assignment.idx;
        let next_centers = update_centers(data, &idx, &assignment.distances, &centers, options)?;
        centers = next_centers;
        if !changed {
            break;
        }
    }
    let assignment = assign_rows(&data.rows, &centers, options.distance);
    idx = assignment.idx;
    let distances = assignment.distances;
    let sumd = sum_distances(k, &idx, &distances);
    let objective = sumd.iter().filter(|value| value.is_finite()).sum::<f64>();
    Ok(ReplicateResult {
        idx_valid: idx,
        centers,
        sumd,
        distances,
        objective,
    })
}

struct Assignment {
    idx: Vec<usize>,
    distances: Vec<Vec<f64>>,
}

impl Assignment {
    fn changed_from(&self, previous: &[usize]) -> bool {
        self.idx != previous
    }
}

fn assign_rows(rows: &[Vec<f64>], centers: &[Vec<f64>], distance: Distance) -> Assignment {
    let mut idx = Vec::with_capacity(rows.len());
    let mut all_distances = Vec::with_capacity(rows.len());
    for row in rows {
        let mut best_idx = 0usize;
        let mut best_distance = f64::INFINITY;
        let mut row_distances = Vec::with_capacity(centers.len());
        for (center_idx, center) in centers.iter().enumerate() {
            let d = distance_value(row, center, distance);
            row_distances.push(d);
            if d < best_distance {
                best_distance = d;
                best_idx = center_idx;
            }
        }
        idx.push(best_idx);
        all_distances.push(row_distances);
    }
    Assignment {
        idx,
        distances: all_distances,
    }
}

fn update_centers(
    data: &PreparedData,
    idx: &[usize],
    distances: &[Vec<f64>],
    previous: &[Vec<f64>],
    options: &Options,
) -> BuiltinResult<Vec<Vec<f64>>> {
    let k = previous.len();
    let mut members = vec![Vec::new(); k];
    for (row, cluster) in idx.iter().enumerate() {
        members[*cluster].push(row);
    }
    let mut centers = vec![vec![f64::NAN; data.cols]; k];
    let mut singleton_rows = vec![false; data.rows.len()];
    for cluster in 0..k {
        if members[cluster].is_empty() {
            match options.empty_action {
                EmptyAction::Error => {
                    return Err(invalid(
                        "kmeans: empty cluster encountered and EmptyAction is 'error'",
                    ))
                }
                EmptyAction::Drop => continue,
                EmptyAction::Singleton => {
                    let row = furthest_assigned_row(distances, idx, &singleton_rows);
                    singleton_rows[row] = true;
                    centers[cluster] = data.rows[row].clone();
                }
            }
        } else {
            centers[cluster] = centroid_for(&data.rows, &members[cluster], options.distance);
        }
    }
    Ok(centers)
}

fn furthest_assigned_row(distances: &[Vec<f64>], idx: &[usize], excluded: &[bool]) -> usize {
    let mut best_row = 0usize;
    let mut best_distance = f64::NEG_INFINITY;
    for (row, cluster) in idx.iter().enumerate() {
        if excluded.get(row).copied().unwrap_or(false) {
            continue;
        }
        let distance = distances[row][*cluster];
        if distance > best_distance {
            best_distance = distance;
            best_row = row;
        }
    }
    best_row
}

fn centroid_for(rows: &[Vec<f64>], members: &[usize], distance: Distance) -> Vec<f64> {
    let cols = rows[0].len();
    match distance {
        Distance::Cityblock | Distance::Hamming => {
            let mut center = vec![0.0; cols];
            for col in 0..cols {
                let mut values = members
                    .iter()
                    .map(|row| rows[*row][col])
                    .collect::<Vec<_>>();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                let mid = values.len() / 2;
                center[col] = if values.len() % 2 == 0 {
                    (values[mid - 1] + values[mid]) / 2.0
                } else {
                    values[mid]
                };
                if distance == Distance::Hamming {
                    center[col] = if center[col] >= 0.5 { 1.0 } else { 0.0 };
                }
            }
            center
        }
        _ => {
            let mut center = vec![0.0; cols];
            for row in members {
                for (col, value) in center.iter_mut().enumerate().take(cols) {
                    *value += rows[*row][col];
                }
            }
            for value in &mut center {
                *value /= members.len() as f64;
            }
            match distance {
                Distance::Cosine => normalize_or_nan(center),
                Distance::Correlation => {
                    let mean = center.iter().sum::<f64>() / center.len() as f64;
                    normalize_or_nan(center.into_iter().map(|value| value - mean).collect())
                }
                _ => center,
            }
        }
    }
}

fn normalize_or_nan(mut row: Vec<f64>) -> Vec<f64> {
    let norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm <= EPS {
        row.fill(f64::NAN);
    } else {
        for value in &mut row {
            *value /= norm;
        }
    }
    row
}

fn sum_distances(k: usize, idx: &[usize], distances: &[Vec<f64>]) -> Vec<f64> {
    let mut sumd = vec![0.0; k];
    for (row, cluster) in idx.iter().enumerate() {
        let value = distances[row][*cluster];
        if value.is_finite() {
            sumd[*cluster] += value;
        }
    }
    sumd
}

fn distance_value(row: &[f64], center: &[f64], distance: Distance) -> f64 {
    if row.iter().any(|v| v.is_nan()) || center.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    match distance {
        Distance::SqEuclidean => row
            .iter()
            .zip(center.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum(),
        Distance::Cityblock => row
            .iter()
            .zip(center.iter())
            .map(|(a, b)| (a - b).abs())
            .sum(),
        Distance::Cosine | Distance::Correlation => {
            1.0 - row
                .iter()
                .zip(center.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
        }
        Distance::Hamming => {
            let mismatches = row
                .iter()
                .zip(center.iter())
                .filter(|(a, b)| (*a - *b).abs() > EPS)
                .count();
            mismatches as f64 / row.len() as f64
        }
    }
}
