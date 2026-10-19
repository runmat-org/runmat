//! Counted explicit region selection using the shared sorted authored-node index.
use crate::topology::storage::{self, SortedValues, TopologyStorageError};
use runmat_analysis_core::{StructuralElementKind, StructuralModel};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExplicitTargetWorkspaceProfile {
    pub candidate_slots: usize,
    pub node_index_bytes: u64,
    pub total_requested_bytes: u64,
}

pub(crate) fn explicit_target_workspace_profile(
    structural: &StructuralModel,
    selector: &str,
) -> Result<ExplicitTargetWorkspaceProfile, TopologyStorageError> {
    storage::check_cancelled()?;
    let selected_node = explicit_node_selector(selector);
    let mut count = 0;
    for (ordinal, node) in structural.nodes.iter().enumerate() {
        if ordinal % 1024 == 0 {
            storage::check_cancelled()?;
        }
        if selected_node == Some(node.node_id) {
            count = storage::count_add(count, 1)?;
        }
    }
    for element in &structural.elements {
        storage::check_cancelled()?;
        if element.region_id == selector {
            count = storage::count_add(
                count,
                match &element.kind {
                    StructuralElementKind::Beam(beam) => beam.node_ids.len(),
                    StructuralElementKind::Shell(shell) => shell.node_ids.len(),
                },
            )?;
        }
    }
    let node_index_bytes = storage::slots::<(u32, usize)>(structural.nodes.len())?;
    let total_requested_bytes = storage::add(node_index_bytes, storage::slots::<usize>(count)?)?;
    usize::try_from(total_requested_bytes).map_err(|_| TopologyStorageError::Capacity)?;
    Ok(ExplicitTargetWorkspaceProfile {
        candidate_slots: count,
        node_index_bytes,
        total_requested_bytes,
    })
}

pub(crate) fn explicit_target_node_indices(
    structural: &StructuralModel,
    selector: &str,
) -> Result<Vec<usize>, TopologyStorageError> {
    let profile = explicit_target_workspace_profile(structural, selector)?;
    let selected_node = explicit_node_selector(selector);
    let index = SortedValues::derive(
        structural
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.node_id, i)),
        structural.nodes.len(),
    )?;
    let mut nodes = storage::reserve(profile.candidate_slots)?;
    nodes.extend(
        structural
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| selected_node == Some(n.node_id))
            .map(|(i, _)| i),
    );
    for element in structural
        .elements
        .iter()
        .filter(|e| e.region_id == selector)
    {
        storage::check_cancelled()?;
        let ids = match &element.kind {
            StructuralElementKind::Beam(beam) => beam.node_ids.as_slice(),
            StructuralElementKind::Shell(shell) => shell.node_ids.as_slice(),
        };
        for id in ids {
            if let Some(index) = index.get(id).copied() {
                nodes.push(index);
            }
        }
    }
    nodes.sort_unstable();
    nodes.dedup();
    Ok(nodes)
}

pub(crate) fn explicit_node_selector(selector: &str) -> Option<u32> {
    selector.strip_prefix("node:")?.parse().ok()
}

#[cfg(test)]
mod tests;
