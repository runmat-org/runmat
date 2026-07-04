use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::contracts::{ProtectedBoundaryComplex, TopologyEntityId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlcBoundaryComponentReport {
    pub component_count: usize,
    pub referenced_node_count: usize,
    pub min_component_node_count: usize,
    pub max_component_node_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlcShellClassificationReport {
    pub shell_nesting_classified: bool,
    pub outer_shell_count: usize,
    pub nested_shell_count: usize,
    pub max_nesting_depth: usize,
}

pub fn classify_boundary_components(plc: &ProtectedBoundaryComplex) -> PlcBoundaryComponentReport {
    let mut adjacency = BTreeMap::<TopologyEntityId, BTreeSet<TopologyEntityId>>::new();
    for facet in &plc.facets {
        for node_id in &facet.node_ids {
            adjacency.entry(node_id.clone()).or_default();
        }
        for edge_index in 0..3 {
            let left = facet.node_ids[edge_index].clone();
            let right = facet.node_ids[(edge_index + 1) % 3].clone();
            adjacency
                .entry(left.clone())
                .or_default()
                .insert(right.clone());
            adjacency.entry(right).or_default().insert(left);
        }
    }

    let mut component_count = 0_usize;
    let mut min_component_node_count = usize::MAX;
    let mut max_component_node_count = 0_usize;
    let mut visited = BTreeSet::<TopologyEntityId>::new();
    for start in adjacency.keys() {
        if visited.contains(start) {
            continue;
        }
        component_count += 1;
        let mut component_node_count = 0_usize;
        let mut stack = vec![start.clone()];
        while let Some(node_id) = stack.pop() {
            if !visited.insert(node_id.clone()) {
                continue;
            }
            component_node_count += 1;
            if let Some(neighbors) = adjacency.get(&node_id) {
                stack.extend(
                    neighbors
                        .iter()
                        .filter(|neighbor| !visited.contains(*neighbor))
                        .cloned(),
                );
            }
        }
        min_component_node_count = min_component_node_count.min(component_node_count);
        max_component_node_count = max_component_node_count.max(component_node_count);
    }

    if component_count == 0 {
        min_component_node_count = 0;
    }

    PlcBoundaryComponentReport {
        component_count,
        referenced_node_count: adjacency.len(),
        min_component_node_count,
        max_component_node_count,
    }
}

pub fn classify_shell_nesting(
    component_report: &PlcBoundaryComponentReport,
) -> PlcShellClassificationReport {
    if component_report.component_count == 1 {
        return PlcShellClassificationReport {
            shell_nesting_classified: true,
            outer_shell_count: 1,
            nested_shell_count: 0,
            max_nesting_depth: 0,
        };
    }

    PlcShellClassificationReport {
        shell_nesting_classified: false,
        outer_shell_count: 0,
        nested_shell_count: 0,
        max_nesting_depth: 0,
    }
}
