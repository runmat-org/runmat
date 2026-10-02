use std::cmp::Ordering;

use super::{
    internal, invalid, Distance, EmptyAction, KmeansStart, Options, PreparedData, ReplicateResult,
    StartSpec, EPS,
};
use crate::BuiltinResult;

pub(super) async fn run_kmeans_cooperative(
    data: &PreparedData,
    k: usize,
    options: &Options,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<ReplicateResult> {
    let starts = initial_centers(data, k, options, cooperation).await?;
    let mut best = None;
    for centers in starts {
        let result = run_replicate(data, centers, options, cooperation).await?;
        if best
            .as_ref()
            .map(|current: &ReplicateResult| result.objective < current.objective)
            .unwrap_or(true)
        {
            best = Some(result);
        }
        cooperation.completed(1).await?;
    }
    best.ok_or_else(|| internal("kmeans: no replicate produced a result"))
}

async fn initial_centers(
    data: &PreparedData,
    k: usize,
    options: &Options,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<Vec<Vec<f64>>>> {
    match &options.start {
        StartSpec::Numeric(centers) => Ok(vec![clone_rows(centers, cooperation).await?]),
        StartSpec::NumericPages(pages) => {
            let mut starts = Vec::new();
            starts.try_reserve_exact(pages.len()).map_err(|source| super::internal_with_source("kmeans: replicate starts exceed available memory", source))?;
            for page in pages {
                starts.push(clone_rows(page, cooperation).await?);
            }
            Ok(starts)
        }
        _ if k > data.rows.len() && options.empty_action != EmptyAction::Drop => Err(invalid(
            "kmeans: k cannot exceed the number of complete observations unless EmptyAction is 'drop'",
        )),
        StartSpec::Named(start) => {
            let mut starts = Vec::new();
            starts
                .try_reserve_exact(options.replicates)
                .map_err(|source| super::internal_with_source("kmeans: replicate starts exceed available memory", source))?;
            for _ in 0..options.replicates {
                let centers = match start {
                    KmeansStart::Plus => plus_start(data, k, options.distance, cooperation).await?,
                    KmeansStart::Sample => sample_start(data, k, cooperation).await?,
                    KmeansStart::Uniform => uniform_start(data, k, cooperation).await?,
                    KmeansStart::Cluster => cluster_start(data, k, options, cooperation).await?,
                };
                starts.push(centers);
                cooperation.completed(1).await?;
            }
            Ok(starts)
        }
    }
}

async fn sample_start(
    data: &PreparedData,
    k: usize,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<Vec<f64>>> {
    if k > data.rows.len() {
        return Err(invalid(
            "kmeans: Start 'sample' requires k no larger than the number of complete observations",
        ));
    }
    let uniforms = cooperation.uniform(data.rows.len()).await?;
    let mut scored = Vec::new();
    scored.try_reserve_exact(uniforms.len()).map_err(|source| {
        super::internal_with_source("kmeans: sample scores exceed available memory", source)
    })?;
    for (index, draw) in uniforms.into_iter().enumerate() {
        scored.push((index, draw));
        cooperation.completed(1).await?;
    }
    cooperation
        .stable_sort_by(&mut scored, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
        })
        .await?;
    let mut centers = Vec::new();
    centers.try_reserve_exact(k).map_err(|source| {
        super::internal_with_source("kmeans: sample centers exceed available memory", source)
    })?;
    for (idx, _) in scored.into_iter().take(k) {
        centers.push(clone_row(&data.rows[idx], cooperation).await?);
    }
    Ok(centers)
}

async fn plus_start(
    data: &PreparedData,
    k: usize,
    distance: Distance,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<Vec<f64>>> {
    if k > data.rows.len() {
        return Err(invalid(
            "kmeans: Start 'plus' requires k no larger than the number of complete observations",
        ));
    }
    let uniforms = cooperation.uniform(k.max(1)).await?;
    let first = ((uniforms[0] * data.rows.len() as f64).floor() as usize).min(data.rows.len() - 1);
    let mut centers = vec![clone_row(&data.rows[first], cooperation).await?];
    while centers.len() < k {
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(data.rows.len())
            .map_err(|source| {
                super::internal_with_source("kmeans: start weights exceed available memory", source)
            })?;
        let mut total = 0.0;
        for row in &data.rows {
            let mut best = f64::INFINITY;
            for center in &centers {
                best = best.min(distance_value(row, center, distance, cooperation).await?);
            }
            let weight = if best.is_finite() { best.max(0.0) } else { 0.0 };
            weights.push(weight);
            total += weight;
            cooperation.completed(1).await?;
        }
        let next = if total <= EPS {
            first_unused_row(data, &centers, distance, cooperation)
                .await?
                .unwrap_or(0)
        } else {
            let mut threshold = uniforms[centers.len()] * total;
            let mut selected = data.rows.len() - 1;
            for (idx, weight) in weights.iter().enumerate() {
                threshold -= *weight;
                cooperation.completed(1).await?;
                if threshold <= 0.0 {
                    selected = idx;
                    break;
                }
            }
            selected
        };
        centers.push(clone_row(&data.rows[next], cooperation).await?);
    }
    Ok(centers)
}

async fn first_unused_row(
    data: &PreparedData,
    centers: &[Vec<f64>],
    distance: Distance,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Option<usize>> {
    for (row_index, row) in data.rows.iter().enumerate() {
        let mut unused = true;
        for center in centers {
            let separated = distance_value(row, center, distance, cooperation).await? > EPS;
            if !separated {
                unused = false;
                break;
            }
        }
        if unused {
            return Ok(Some(row_index));
        }
    }
    Ok(None)
}

async fn uniform_start(
    data: &PreparedData,
    k: usize,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<Vec<f64>>> {
    let cols = data.cols;
    let mut mins = filled(cols, f64::INFINITY, cooperation).await?;
    let mut maxs = filled(cols, f64::NEG_INFINITY, cooperation).await?;
    for row in &data.rows {
        for col in 0..cols {
            mins[col] = mins[col].min(row[col]);
            maxs[col] = maxs[col].max(row[col]);
            cooperation.completed(1).await?;
        }
    }
    let draw_count = k
        .checked_mul(cols)
        .ok_or_else(|| invalid("kmeans: uniform start shape overflow"))?;
    let uniforms = cooperation.uniform(draw_count).await?;
    let mut centers = Vec::new();
    centers.try_reserve_exact(k).map_err(|source| {
        super::internal_with_source("kmeans: uniform centers exceed available memory", source)
    })?;
    for _ in 0..k {
        centers.push(filled(cols, 0.0, cooperation).await?);
    }
    for center in 0..k {
        for col in 0..cols {
            let lo = mins[col];
            let hi = maxs[col];
            centers[center][col] = if hi > lo {
                lo + (hi - lo) * uniforms[center + col * k]
            } else {
                lo
            };
            cooperation.completed(1).await?;
        }
    }
    Ok(centers)
}

async fn cluster_start(
    data: &PreparedData,
    k: usize,
    options: &Options,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<Vec<f64>>> {
    let sample_size = data.rows.len().div_ceil(10).max(k).min(data.rows.len());
    let rows = if sample_size == data.rows.len() {
        clone_rows(&data.rows, cooperation).await?
    } else {
        sample_start(data, sample_size, cooperation).await?
    };
    let mut original_to_valid = Vec::new();
    original_to_valid
        .try_reserve_exact(sample_size)
        .map_err(|source| {
            super::internal_with_source("kmeans: subset indices exceed available memory", source)
        })?;
    for row in 0..sample_size {
        original_to_valid.push(Some(row));
        cooperation.completed(1).await?;
    }
    let subset = PreparedData {
        rows,
        original_row_count: sample_size,
        original_to_valid,
        cols: data.cols,
    };
    if subset.rows.len() == k {
        return Ok(subset.rows);
    }
    let centers = plus_start(&subset, k, options.distance, cooperation).await?;
    let preliminary = Options {
        distance: options.distance,
        empty_action: EmptyAction::Singleton,
        max_iter: options.max_iter.clamp(1, 10),
        online_phase: false,
        replicates: 1,
        start: StartSpec::Numeric(clone_rows(&centers, cooperation).await?),
    };
    run_replicate(&subset, centers, &preliminary, cooperation)
        .await
        .map(|result| result.centers)
}

async fn run_replicate(
    data: &PreparedData,
    mut centers: Vec<Vec<f64>>,
    options: &Options,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<ReplicateResult> {
    let k = centers.len();
    let mut idx = Vec::new();
    idx.try_reserve_exact(data.rows.len()).map_err(|source| {
        super::internal_with_source("kmeans: assignments exceed available memory", source)
    })?;
    for _ in &data.rows {
        idx.push(0);
        cooperation.completed(1).await?;
    }
    let iterations = if options.online_phase {
        options.max_iter.saturating_mul(2)
    } else {
        options.max_iter
    };
    for _ in 0..iterations {
        let assignment = assign_rows(&data.rows, &centers, options.distance, cooperation).await?;
        let changed = assignment.changed_from(&idx, cooperation).await?;
        idx = assignment.idx;
        let next_centers = update_centers(
            data,
            &idx,
            &assignment.distances,
            &centers,
            options,
            cooperation,
        )
        .await?;
        centers = next_centers;
        cooperation.completed(1).await?;
        if !changed {
            break;
        }
    }
    let assignment = assign_rows(&data.rows, &centers, options.distance, cooperation).await?;
    idx = assignment.idx;
    let distances = assignment.distances;
    let sumd = sum_distances(k, &idx, &distances, cooperation).await?;
    let mut objective = 0.0;
    for value in sumd.iter().filter(|value| value.is_finite()) {
        objective += value;
        cooperation.completed(1).await?;
    }
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
    async fn changed_from(
        &self,
        previous: &[usize],
        cooperation: &mut super::cooperation::KmeansCooperation,
    ) -> BuiltinResult<bool> {
        if self.idx.len() != previous.len() {
            return Ok(true);
        }
        for (current, old) in self.idx.iter().zip(previous) {
            let changed = current != old;
            cooperation.completed(1).await?;
            if changed {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

async fn assign_rows(
    rows: &[Vec<f64>],
    centers: &[Vec<f64>],
    distance: Distance,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Assignment> {
    let mut idx = Vec::new();
    idx.try_reserve_exact(rows.len()).map_err(|source| {
        super::internal_with_source("kmeans: assignments exceed available memory", source)
    })?;
    let mut all_distances = Vec::new();
    all_distances
        .try_reserve_exact(rows.len())
        .map_err(|source| {
            super::internal_with_source("kmeans: distances exceed available memory", source)
        })?;
    for row in rows {
        let mut best_idx = 0usize;
        let mut best_distance = f64::INFINITY;
        let mut row_distances = Vec::new();
        row_distances
            .try_reserve_exact(centers.len())
            .map_err(|source| {
                super::internal_with_source("kmeans: distances exceed available memory", source)
            })?;
        for (center_idx, center) in centers.iter().enumerate() {
            let d = distance_value(row, center, distance, cooperation).await?;
            row_distances.push(d);
            if d < best_distance {
                best_distance = d;
                best_idx = center_idx;
            }
        }
        idx.push(best_idx);
        all_distances.push(row_distances);
        cooperation.completed(1).await?;
    }
    Ok(Assignment {
        idx,
        distances: all_distances,
    })
}

async fn update_centers(
    data: &PreparedData,
    idx: &[usize],
    distances: &[Vec<f64>],
    previous: &[Vec<f64>],
    options: &Options,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<Vec<f64>>> {
    let k = previous.len();
    let mut counts = filled(k, 0usize, cooperation).await?;
    for cluster in idx {
        counts[*cluster] += 1;
        cooperation.completed(1).await?;
    }
    let mut members = filled(k, Vec::<usize>::new(), cooperation).await?;
    for (member, count) in members.iter_mut().zip(counts) {
        member.try_reserve_exact(count).map_err(|source| {
            super::internal_with_source("kmeans: cluster members exceed available memory", source)
        })?;
        cooperation.completed(1).await?;
    }
    for (row, cluster) in idx.iter().enumerate() {
        members[*cluster].push(row);
        cooperation.completed(1).await?;
    }
    let mut centers = Vec::new();
    centers.try_reserve_exact(k).map_err(|source| {
        super::internal_with_source("kmeans: centers exceed available memory", source)
    })?;
    for _ in 0..k {
        centers.push(filled(data.cols, f64::NAN, cooperation).await?);
    }
    let mut singleton_rows = filled(data.rows.len(), false, cooperation).await?;
    for cluster in 0..k {
        if members[cluster].is_empty() {
            match options.empty_action {
                EmptyAction::Error => {
                    return Err(invalid(
                        "kmeans: empty cluster encountered and EmptyAction is 'error'",
                    ))
                }
                EmptyAction::Drop => {
                    cooperation.completed(1).await?;
                    continue;
                }
                EmptyAction::Singleton => {
                    let row =
                        furthest_assigned_row(distances, idx, &singleton_rows, cooperation).await?;
                    singleton_rows[row] = true;
                    centers[cluster] = clone_row(&data.rows[row], cooperation).await?;
                }
            }
        } else {
            centers[cluster] =
                centroid_for(&data.rows, &members[cluster], options.distance, cooperation).await?;
        }
        cooperation.completed(1).await?;
    }
    Ok(centers)
}

async fn furthest_assigned_row(
    distances: &[Vec<f64>],
    idx: &[usize],
    excluded: &[bool],
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<usize> {
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
        cooperation.completed(1).await?;
    }
    Ok(best_row)
}

async fn centroid_for(
    rows: &[Vec<f64>],
    members: &[usize],
    distance: Distance,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<f64>> {
    let cols = rows[0].len();
    match distance {
        Distance::Cityblock | Distance::Hamming => {
            let mut center = filled(cols, 0.0, cooperation).await?;
            for col in 0..cols {
                let mut values = Vec::new();
                values.try_reserve_exact(members.len()).map_err(|source| {
                    super::internal_with_source(
                        "kmeans: centroid samples exceed available memory",
                        source,
                    )
                })?;
                for row in members {
                    values.push(rows[*row][col]);
                    cooperation.completed(1).await?;
                }
                cooperation
                    .stable_sort_by(&mut values, |a, b| {
                        a.partial_cmp(b).unwrap_or(Ordering::Equal)
                    })
                    .await?;
                let mid = values.len() / 2;
                center[col] = if values.len() % 2 == 0 {
                    (values[mid - 1] + values[mid]) / 2.0
                } else {
                    values[mid]
                };
                if distance == Distance::Hamming {
                    center[col] = if center[col] >= 0.5 { 1.0 } else { 0.0 };
                }
                cooperation.completed(1).await?;
            }
            Ok(center)
        }
        _ => {
            let mut center = filled(cols, 0.0, cooperation).await?;
            for row in members {
                for (col, value) in center.iter_mut().enumerate().take(cols) {
                    *value += rows[*row][col];
                    cooperation.completed(1).await?;
                }
            }
            for value in &mut center {
                *value /= members.len() as f64;
                cooperation.completed(1).await?;
            }
            match distance {
                Distance::Cosine | Distance::Correlation => {
                    super::geometry::direction_or_nan(
                        &center,
                        distance == Distance::Correlation,
                        cooperation,
                    )
                    .await
                }
                _ => Ok(center),
            }
        }
    }
}

async fn clone_row(
    row: &[f64],
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(row.len()).map_err(|source| {
        super::internal_with_source("kmeans: centers exceed available memory", source)
    })?;
    for &value in row {
        copy.push(value);
        cooperation.completed(1).await?;
    }
    Ok(copy)
}

async fn clone_rows(
    rows: &[Vec<f64>],
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<Vec<f64>>> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(rows.len()).map_err(|source| {
        super::internal_with_source("kmeans: centers exceed available memory", source)
    })?;
    for row in rows {
        copy.push(clone_row(row, cooperation).await?);
    }
    Ok(copy)
}

async fn filled<T: Clone>(
    len: usize,
    value: T,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|source| {
        super::internal_with_source(
            "kmeans: clustering workspace exceeds available memory",
            source,
        )
    })?;
    for _ in 0..len {
        values.push(value.clone());
        cooperation.completed(1).await?;
    }
    Ok(values)
}

async fn sum_distances(
    k: usize,
    idx: &[usize],
    distances: &[Vec<f64>],
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut sumd = filled(k, 0.0, cooperation).await?;
    for (row, cluster) in idx.iter().enumerate() {
        let value = distances[row][*cluster];
        if value.is_finite() {
            sumd[*cluster] += value;
        }
        cooperation.completed(1).await?;
    }
    Ok(sumd)
}

async fn distance_value(
    row: &[f64],
    center: &[f64],
    distance: Distance,
    cooperation: &mut super::cooperation::KmeansCooperation,
) -> BuiltinResult<f64> {
    for value in row {
        let missing = value.is_nan();
        cooperation.completed(1).await?;
        if missing {
            return Ok(f64::NAN);
        }
    }
    for value in center {
        let missing = value.is_nan();
        cooperation.completed(1).await?;
        if missing {
            return Ok(f64::NAN);
        }
    }
    let mut sum = 0.0;
    let mut mismatches = 0usize;
    for (a, b) in row.iter().zip(center.iter()) {
        match distance {
            Distance::SqEuclidean => {
                let diff = a - b;
                sum += diff * diff;
            }
            Distance::Cityblock => sum += (a - b).abs(),
            Distance::Cosine | Distance::Correlation => sum += a * b,
            Distance::Hamming => {
                if (a - b).abs() > EPS {
                    mismatches += 1;
                }
            }
        }
        cooperation.completed(1).await?;
    }
    Ok(match distance {
        Distance::SqEuclidean | Distance::Cityblock => sum,
        Distance::Cosine | Distance::Correlation => 1.0 - sum,
        Distance::Hamming => mismatches as f64 / row.len() as f64,
    })
}
