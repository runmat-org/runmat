//! Iterative traversal over counted indexed adjacency; no hash-table workspace.
use super::*;
pub(super) fn validate_acyclic(
    sequence: &AnalysisSequence,
    edges: &Adjacency,
) -> Result<(), AnalysisValidationError> {
    let count = sequence.steps.len();
    let mut remaining = vec![0usize; count];
    for &successor in &edges.neighbors {
        remaining[successor] += 1;
    }
    let mut ready = Vec::with_capacity(count);
    for (i, &n) in remaining.iter().enumerate() {
        if n == 0 {
            ready.push(i);
        }
    }
    let mut visited = 0;
    while let Some(i) = ready.pop() {
        visited += 1;
        for &successor in edges.row(i) {
            remaining[successor] -= 1;
            if remaining[successor] == 0 {
                ready.push(successor);
            }
        }
    }
    if visited == count {
        Ok(())
    } else {
        Err(sequence_error(
            sequence,
            "dependency graph contains a cycle",
        ))
    }
}
pub(super) fn is_ancestor(predecessors: &Adjacency, candidate: usize, successor: usize) -> bool {
    if candidate == successor {
        return false;
    }
    let count = predecessors.offsets.len() - 1;
    let mut visited = vec![false; count];
    let mut pending = Vec::with_capacity(count);
    visited[successor] = true;
    pending.push(successor);
    while let Some(i) = pending.pop() {
        for &previous in predecessors.row(i) {
            if previous == candidate {
                return true;
            }
            if !visited[previous] {
                visited[previous] = true;
                pending.push(previous);
            }
        }
    }
    false
}
