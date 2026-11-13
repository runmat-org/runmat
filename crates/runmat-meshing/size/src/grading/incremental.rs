//! Exact cap updates after a graph edit. Invalidate the old shortest-path
//! dependency closure before propagating current raw caps over current edges.
//! The graph remains borrowed and unchanged; callers apply the plan atomically.
use super::invalid;
use crate::metric::MetricContractError;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

/// Before/after views of an already admitted symmetric graph. `changed` must
/// include every inserted/deleted vertex and every endpoint of a changed edge
/// or raw length. Stable handles must retain their identity across this edit.
/// A missing raw length denotes a vertex absent from the current graph.
pub trait MetricCapGraph<Key: Copy + Ord> {
    fn previous_cap(&self, key: Key) -> Option<f64>;
    fn raw_length(&self, key: Key) -> Result<Option<f64>, MetricContractError>;
    fn neighbors(&self, key: Key, previous: bool) -> Vec<Key>;
}

#[derive(Debug, PartialEq)]
pub enum MetricCapUpdateError {
    InvalidMetric(MetricContractError),
    Cancelled,
}
impl From<MetricContractError> for MetricCapUpdateError {
    fn from(value: MetricContractError) -> Self {
        Self::InvalidMetric(value)
    }
}

/// A complete update relative to the admitted previous caps. Entries can include
/// unchanged caps in the invalidated closure. Unlisted current vertices retain
/// their previous cap. Storage follows the affected closure, not total inventory.
#[derive(Debug)]
pub struct MetricCapUpdate<Key> {
    pub caps: BTreeMap<Key, f64>,
    pub invalidated_vertices: usize,
    pub neighbor_reads: usize,
}

pub fn plan_metric_cap_update<Key: Copy + Ord>(
    ratio: f64,
    graph: &impl MetricCapGraph<Key>,
    changed: impl IntoIterator<Item = Key>,
    mut is_cancelled: impl FnMut() -> bool,
) -> Result<MetricCapUpdate<Key>, MetricCapUpdateError> {
    if !ratio.is_finite() || ratio < 1.0 {
        return Err(invalid(
            "maximum metric grading ratio",
            "must be finite and at least one",
        )
        .into());
    }
    let mut invalidated = BTreeSet::new();
    let mut pending = Vec::new();
    for key in changed {
        if is_cancelled() {
            return Err(MetricCapUpdateError::Cancelled);
        }
        if invalidated.insert(key) {
            pending.push(key);
        }
    }
    let mut neighbor_reads = 0;
    // A cap may have multiple equally short supports. Invalidating all matching
    // descendants is conservative, including ratio-one cycles. Independent raw
    // sources are restored below; no stale support survives a deletion/increase.
    while let Some(key) = pending.pop() {
        if is_cancelled() {
            return Err(MetricCapUpdateError::Cancelled);
        }
        let Some(cap) = graph.previous_cap(key) else {
            continue;
        };
        valid(cap)?;
        for neighbor in graph.neighbors(key, true) {
            neighbor_reads += 1;
            if graph
                .previous_cap(neighbor)
                .is_some_and(|next| next == cap * ratio)
                && invalidated.insert(neighbor)
            {
                pending.push(neighbor);
            }
        }
    }
    let mut caps = BTreeMap::new();
    let mut frontier = BinaryHeap::new();
    for &key in &invalidated {
        if is_cancelled() {
            return Err(MetricCapUpdateError::Cancelled);
        }
        let Some(raw) = graph.raw_length(key)? else {
            continue;
        };
        valid(raw)?;
        let mut cap = raw;
        for neighbor in graph.neighbors(key, false) {
            neighbor_reads += 1;
            if !invalidated.contains(&neighbor) {
                let previous = graph.previous_cap(neighbor).ok_or_else(|| {
                    invalid(
                        "metric grading edit",
                        "new vertices and changed edges must be included in the edit",
                    )
                })?;
                valid(previous)?;
                cap = cap.min(previous * ratio);
            }
        }
        caps.insert(key, cap);
        frontier.push(Frontier { key, cap });
    }
    while let Some(current) = frontier.pop() {
        if is_cancelled() {
            return Err(MetricCapUpdateError::Cancelled);
        }
        if caps[&current.key] != current.cap {
            continue;
        }
        let candidate = current.cap * ratio;
        for neighbor in graph.neighbors(current.key, false) {
            neighbor_reads += 1;
            let previous = caps
                .get(&neighbor)
                .copied()
                .or_else(|| graph.previous_cap(neighbor))
                .ok_or_else(|| invalid("metric grading edit", "current neighbor has no cap"))?;
            valid(previous)?;
            if candidate < previous {
                caps.insert(neighbor, candidate);
                frontier.push(Frontier {
                    key: neighbor,
                    cap: candidate,
                });
            }
        }
    }
    Ok(MetricCapUpdate {
        caps,
        invalidated_vertices: invalidated.len(),
        neighbor_reads,
    })
}

fn valid(cap: f64) -> Result<(), MetricContractError> {
    if cap.is_finite() && cap > 0.0 {
        Ok(())
    } else {
        Err(invalid("metric grading cap", "must be finite and positive"))
    }
}
struct Frontier<Key> {
    key: Key,
    cap: f64,
}
impl<Key: Ord> PartialEq for Frontier<Key> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.cap == other.cap
    }
}
impl<Key: Ord> Eq for Frontier<Key> {}
impl<Key: Ord> PartialOrd for Frontier<Key> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<Key: Ord> Ord for Frontier<Key> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .cap
            .total_cmp(&self.cap)
            .then_with(|| other.key.cmp(&self.key))
    }
}
#[cfg(test)]
mod tests;
