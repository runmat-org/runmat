//! Canonical adjacency from counted element-node occurrences; no per-node tree indexes.
use super::{
    storage::{check_cancelled, count_add, multiply, reserve},
    AnalysisProjectionError, ProjectedElement,
};

pub(super) fn occurrence_count(
    mut counts: impl Iterator<Item = usize>,
) -> Result<usize, AnalysisProjectionError> {
    counts.try_fold(0, |total, count| {
        Ok(count_add(total, multiply(count, count.saturating_sub(1))?)?)
    })
}

pub(super) fn derive(
    elements: &[ProjectedElement],
    nodes: usize,
) -> Result<Vec<Vec<usize>>, AnalysisProjectionError> {
    let count = occurrence_count(elements.iter().map(|element| element.node_indices.len()))?;
    let mut candidates = reserve(count)?;
    for element in elements {
        check_cancelled()?;
        for &left in &element.node_indices {
            for &right in &element.node_indices {
                if left != right {
                    candidates.push((left, right));
                }
            }
        }
    }
    check_cancelled()?;
    candidates.sort_unstable();
    candidates.dedup();
    check_cancelled()?;
    let mut adjacency = reserve(nodes)?;
    let mut first = 0;
    for node in 0..nodes {
        check_cancelled()?;
        let mut end = first;
        while end < candidates.len() && candidates[end].0 == node {
            end += 1;
        }
        let mut neighbors = reserve(end - first)?;
        neighbors.extend(candidates[first..end].iter().map(|entry| entry.1));
        adjacency.push(neighbors);
        first = end;
    }
    Ok(adjacency)
}

#[cfg(test)]
mod tests;
