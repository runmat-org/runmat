use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::contracts::{PlcNode, ProtectedBoundaryComplex, TopologyEntityId};

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
    let components = boundary_components(plc);
    let mut min_component_node_count = usize::MAX;
    let mut max_component_node_count = 0_usize;

    for component in &components {
        let component_node_count = component.len();
        min_component_node_count = min_component_node_count.min(component_node_count);
        max_component_node_count = max_component_node_count.max(component_node_count);
    }

    if components.is_empty() {
        min_component_node_count = 0;
    }

    PlcBoundaryComponentReport {
        component_count: components.len(),
        referenced_node_count: components.iter().map(Vec::len).sum(),
        min_component_node_count,
        max_component_node_count,
    }
}

pub fn classify_shell_nesting(
    plc: &ProtectedBoundaryComplex,
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

    let Some(component_bounds) = component_bounds(plc) else {
        return unclassified_shells();
    };
    if component_bounds.len() != component_report.component_count || component_bounds.len() < 2 {
        return unclassified_shells();
    }

    let containing_shell_indices = component_bounds
        .iter()
        .enumerate()
        .filter(|(outer_index, outer_bounds)| {
            component_bounds
                .iter()
                .enumerate()
                .filter(|(inner_index, _)| inner_index != outer_index)
                .all(|(_, inner_bounds)| outer_bounds.strictly_contains(inner_bounds))
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();

    if containing_shell_indices.len() == 1 {
        return PlcShellClassificationReport {
            shell_nesting_classified: true,
            outer_shell_count: 1,
            nested_shell_count: component_report.component_count - 1,
            max_nesting_depth: 1,
        };
    }

    unclassified_shells()
}

fn unclassified_shells() -> PlcShellClassificationReport {
    PlcShellClassificationReport {
        shell_nesting_classified: false,
        outer_shell_count: 0,
        nested_shell_count: 0,
        max_nesting_depth: 0,
    }
}

fn boundary_components(plc: &ProtectedBoundaryComplex) -> Vec<Vec<TopologyEntityId>> {
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

    let mut components = Vec::<Vec<TopologyEntityId>>::new();
    let mut visited = BTreeSet::<TopologyEntityId>::new();
    for start in adjacency.keys() {
        if visited.contains(start) {
            continue;
        }
        let mut component = Vec::<TopologyEntityId>::new();
        let mut stack = vec![start.clone()];
        while let Some(node_id) = stack.pop() {
            if !visited.insert(node_id.clone()) {
                continue;
            }
            component.push(node_id.clone());
            if let Some(neighbors) = adjacency.get(&node_id) {
                stack.extend(
                    neighbors
                        .iter()
                        .filter(|neighbor| !visited.contains(*neighbor))
                        .cloned(),
                );
            }
        }
        component.sort();
        components.push(component);
    }
    components
}

fn component_bounds(plc: &ProtectedBoundaryComplex) -> Option<Vec<ComponentBounds>> {
    let node_by_id = plc
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node))
        .collect::<BTreeMap<_, _>>();
    let plc_bounds = ComponentBounds::from_nodes(plc.nodes.iter())?;
    let containment_epsilon = plc_bounds.max_extent().max(1.0) * 1.0e-9;

    boundary_components(plc)
        .iter()
        .map(|component| {
            let component_nodes = component
                .iter()
                .map(|node_id| node_by_id.get(node_id).copied())
                .collect::<Option<Vec<_>>>()?;
            ComponentBounds::from_nodes(component_nodes)
                .map(|bounds| bounds.with_containment_epsilon(containment_epsilon))
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct ComponentBounds {
    min_m: [f64; 3],
    max_m: [f64; 3],
    containment_epsilon: f64,
}

impl ComponentBounds {
    fn from_nodes<'a>(nodes: impl IntoIterator<Item = &'a PlcNode>) -> Option<Self> {
        let mut min_m = [f64::INFINITY; 3];
        let mut max_m = [f64::NEG_INFINITY; 3];
        let mut has_node = false;

        for node in nodes {
            if !node
                .coordinates_m
                .iter()
                .all(|coordinate| coordinate.is_finite())
            {
                return None;
            }
            has_node = true;
            for axis in 0..3 {
                min_m[axis] = min_m[axis].min(node.coordinates_m[axis]);
                max_m[axis] = max_m[axis].max(node.coordinates_m[axis]);
            }
        }

        has_node.then_some(Self {
            min_m,
            max_m,
            containment_epsilon: 0.0,
        })
    }

    fn with_containment_epsilon(mut self, containment_epsilon: f64) -> Self {
        self.containment_epsilon = containment_epsilon;
        self
    }

    fn max_extent(&self) -> f64 {
        (0..3)
            .map(|axis| self.max_m[axis] - self.min_m[axis])
            .fold(0.0, f64::max)
    }

    fn strictly_contains(&self, nested: &Self) -> bool {
        (0..3).all(|axis| {
            nested.min_m[axis] > self.min_m[axis] + self.containment_epsilon
                && nested.max_m[axis] < self.max_m[axis] - self.containment_epsilon
        })
    }
}
