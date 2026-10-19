//! Grading over borrowed CSR support. Scratch state contains indices and caps,
//! never copies of geometric identities, adjacency trees or metric metadata.
use super::invalid;
use crate::metric::{MetricContractError, ResolvedMetricEvaluation};
use std::{cmp::Ordering, collections::BinaryHeap};

/// Applies the same canonical size-cap propagation as the map entry point.
/// CSR rows must exactly cover unique identities and have symmetric, known,
/// non-self, non-duplicate neighbors. Identity order may be arbitrary.
pub fn grade_indexed_metric_evaluations<Identity: Ord>(
    maximum_grading_ratio: f64,
    identities: &[Identity],
    row_offsets: &[usize],
    column_indices: &[usize],
    evaluations: &mut [ResolvedMetricEvaluation],
) -> Result<(), MetricContractError> {
    if identities.len() != evaluations.len() {
        return Err(invalid(
            "metric grading graph",
            "must exactly cover the evaluations with symmetric, known, non-self adjacency",
        ));
    }
    let updates = plan(
        maximum_grading_ratio,
        identities,
        row_offsets,
        column_indices,
        |index| evaluations[index].metric.conservative_minimum_length_m(),
    )?;
    for (index, cap) in updates {
        apply_cap(&mut evaluations[index], cap)?;
    }
    Ok(())
}

pub(super) fn plan<Identity: Ord>(
    maximum_grading_ratio: f64,
    identities: &[Identity],
    row_offsets: &[usize],
    column_indices: &[usize],
    length: impl Fn(usize) -> Result<f64, MetricContractError>,
) -> Result<Vec<(usize, f64)>, MetricContractError> {
    if !maximum_grading_ratio.is_finite() || maximum_grading_ratio < 1.0 {
        return Err(invalid(
            "maximum metric grading ratio",
            "must be finite and at least one",
        ));
    }
    let ranks = validate_graph(identities, row_offsets, column_indices, identities.len())?;
    let mut caps = (0..identities.len())
        .map(length)
        .collect::<Result<Vec<_>, _>>()?;
    // Every possible relaxation is bounded below by min(cap) * ratio.
    // If that already covers the largest cap, the complete graph has no update.
    // Validate support first; a constant field cannot bless malformed adjacency.
    if caps.is_empty()
        || (caps.iter().all(|cap| cap.is_finite() && *cap > 0.0)
            && caps.iter().copied().fold(0.0_f64, f64::max)
                <= caps.iter().copied().fold(f64::INFINITY, f64::min) * maximum_grading_ratio)
    {
        return Ok(Vec::new());
    }
    let original_caps = caps.clone();
    let mut frontier = BinaryHeap::with_capacity(caps.len());
    for (index, cap) in caps.iter().copied().enumerate() {
        frontier.push(Frontier {
            cap,
            rank: ranks[index],
            index,
        });
    }
    while let Some(current) = frontier.pop() {
        if current.cap.to_bits() != caps[current.index].to_bits() {
            continue;
        }
        for &neighbor in &column_indices[row_offsets[current.index]..row_offsets[current.index + 1]]
        {
            let candidate = current.cap * maximum_grading_ratio;
            if candidate.is_finite() && candidate < caps[neighbor] {
                caps[neighbor] = candidate;
                frontier.push(Frontier {
                    cap: candidate,
                    rank: ranks[neighbor],
                    index: neighbor,
                });
            }
        }
    }
    // Preserve canonical update order, including which arithmetic error is first.
    let mut order = (0..identities.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|index| ranks[*index]);
    Ok(order
        .into_iter()
        .filter_map(|index| (caps[index] < original_caps[index]).then_some((index, caps[index])))
        .collect())
}

pub(super) fn apply_cap(
    evaluation: &mut ResolvedMetricEvaluation,
    cap: f64,
) -> Result<(), MetricContractError> {
    let grading_metric = crate::metric::MetricTensor3::isotropic_length_m(cap)?;
    let required = grading_metric.xx;
    let mut metric = evaluation.metric;
    let density = metric.maximum_row_density();
    if density < required {
        if metric.xy == 0.0
            && metric.xz == 0.0
            && metric.yz == 0.0
            && metric.xx == metric.yy
            && metric.yy == metric.zz
        {
            // Preserve the original isotropic cap exactly.
            metric.xx = required;
            metric.yy = required;
            metric.zz = required;
        } else {
            // A positive scalar multiple tightens every existing directional
            // constraint while preserving anisotropy. Filling all diagonals
            // with the finest density would erase the unconstrained directions.
            let factor = (required / density) * (1.0 + 16.0 * f64::EPSILON);
            metric.xx *= factor;
            metric.yy *= factor;
            metric.zz *= factor;
            metric.xy *= factor;
            metric.xz *= factor;
            metric.yz *= factor;
        }
    }
    metric.validate()?;
    if metric.maximum_row_density() < required {
        return Err(invalid(
            "metric grading cap",
            "cannot realize the requested density",
        ));
    }
    evaluation.metric = metric;
    evaluation.clipped_contribution_count = evaluation
        .clipped_contribution_count
        .checked_add(1)
        .ok_or_else(|| invalid("metric grading evidence", "count overflowed"))?;
    Ok(())
}

fn validate_graph<Identity: Ord>(
    identities: &[Identity],
    offsets: &[usize],
    columns: &[usize],
    evaluation_count: usize,
) -> Result<Vec<usize>, MetricContractError> {
    let failure = || {
        invalid(
            "metric grading graph",
            "must exactly cover the evaluations with symmetric, known, non-self adjacency",
        )
    };
    if identities.len() != evaluation_count
        || offsets.len() != identities.len() + 1
        || offsets.first() != Some(&0)
        || offsets.last() != Some(&columns.len())
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(failure());
    }
    let mut order = (0..identities.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|left, right| identities[*left].cmp(&identities[*right]));
    if order
        .windows(2)
        .any(|pair| identities[pair[0]] == identities[pair[1]])
    {
        return Err(failure());
    }
    let mut ranks = vec![0; identities.len()];
    for (rank, index) in order.into_iter().enumerate() {
        ranks[index] = rank;
    }
    // Sorting compact indices once avoids quadratic validation of high-valence
    // rows. The borrowed caller graph and canonical propagation order stay intact.
    let mut sorted_columns = columns.to_vec();
    for index in 0..identities.len() {
        let neighbors = &mut sorted_columns[offsets[index]..offsets[index + 1]];
        neighbors.sort_unstable();
        if neighbors
            .last()
            .is_some_and(|neighbor| *neighbor >= identities.len())
            || neighbors.binary_search(&index).is_ok()
            || neighbors.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(failure());
        }
    }
    for index in 0..identities.len() {
        for &neighbor in &sorted_columns[offsets[index]..offsets[index + 1]] {
            if sorted_columns[offsets[neighbor]..offsets[neighbor + 1]]
                .binary_search(&index)
                .is_err()
            {
                return Err(failure());
            }
        }
    }
    Ok(ranks)
}

struct Frontier {
    cap: f64,
    rank: usize,
    index: usize,
}
impl PartialEq for Frontier {
    fn eq(&self, other: &Self) -> bool {
        self.cap.to_bits() == other.cap.to_bits() && self.rank == other.rank
    }
}
impl Eq for Frontier {}
impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Frontier {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cap
            .total_cmp(&self.cap)
            .then_with(|| other.rank.cmp(&self.rank))
    }
}

#[cfg(test)]
mod tests;
