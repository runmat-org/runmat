use std::collections::VecDeque;

use super::NullspaceEvidenceError;

pub(crate) fn connected_components(
    neighbors: &[Vec<usize>],
) -> Result<Vec<Vec<usize>>, NullspaceEvidenceError> {
    if neighbors
        .iter()
        .any(|adjacent| adjacent.iter().any(|node| *node >= neighbors.len()))
    {
        return Err(NullspaceEvidenceError::DimensionMismatch);
    }
    let mut seen = vec![false; neighbors.len()];
    let mut components = Vec::new();
    for root in 0..neighbors.len() {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut queue = VecDeque::from([root]);
        let mut component = Vec::new();
        while let Some(node) = queue.pop_front() {
            component.push(node);
            for neighbor in &neighbors[node] {
                if !seen[*neighbor] {
                    seen[*neighbor] = true;
                    queue.push_back(*neighbor);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    Ok(components)
}
