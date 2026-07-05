use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{ProtectedBoundaryComplex, TopologyEntityId},
    quality::{
        predicate::{point_in_closed_triangle_surface, PointInClosedSurface},
        tolerance::MeshingTolerance,
    },
};

use crate::generate::TetrahedronGenerationError;

#[derive(Debug, Clone)]
pub(super) struct NestedTetrahedronShell {
    pub(super) outer_node_ids: Vec<TopologyEntityId>,
    pub(super) inner_node_ids: Vec<TopologyEntityId>,
    pub(super) outer_facet_indices: Vec<usize>,
    pub(super) inner_facet_indices: Vec<usize>,
    pub(super) outer_volume_m3: f64,
    pub(super) inner_volume_m3: f64,
}

pub(super) fn nested_tetrahedron_shell(
    plc: &ProtectedBoundaryComplex,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    tolerance: MeshingTolerance,
) -> Result<NestedTetrahedronShell, TetrahedronGenerationError> {
    let components = boundary_components(plc);
    if components.len() != 2 {
        return Err(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc);
    }
    let mut shells = components
        .into_iter()
        .map(|component| triangulated_shell(plc, coordinates_by_id, component))
        .collect::<Result<Vec<_>, _>>()?;
    shells.sort_by(|left, right| right.volume_m3.total_cmp(&left.volume_m3));
    let outer = &shells[0];
    let inner = &shells[1];
    let outer_triangles = outer
        .facet_indices
        .iter()
        .map(|facet_index| {
            let facet = &plc.facets[*facet_index];
            facet
                .node_ids
                .clone()
                .map(|node_id| coordinates_by_id[&node_id])
        })
        .collect::<Vec<_>>();
    if !inner.node_ids.iter().all(|node_id| {
        point_in_closed_triangle_surface(coordinates_by_id[node_id], &outer_triangles, tolerance)
            == PointInClosedSurface::Inside
    }) {
        return Err(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc);
    }
    Ok(NestedTetrahedronShell {
        outer_node_ids: outer.node_ids.clone(),
        inner_node_ids: inner.node_ids.clone(),
        outer_facet_indices: outer.facet_indices.clone(),
        inner_facet_indices: inner.facet_indices.clone(),
        outer_volume_m3: outer.volume_m3,
        inner_volume_m3: inner.volume_m3,
    })
}

#[derive(Debug, Clone)]
struct TriangulatedShell {
    node_ids: Vec<TopologyEntityId>,
    facet_indices: Vec<usize>,
    volume_m3: f64,
}

fn triangulated_shell(
    plc: &ProtectedBoundaryComplex,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    component: BoundaryComponent,
) -> Result<TriangulatedShell, TetrahedronGenerationError> {
    if component.node_ids.len() < 4 || component.facet_indices.len() < 4 {
        return Err(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc);
    }
    let signed_volume_m3 = component
        .facet_indices
        .iter()
        .map(|facet_index| {
            let facet = &plc.facets[*facet_index];
            let [a, b, c] = facet
                .node_ids
                .clone()
                .map(|node_id| coordinates_by_id[&node_id]);
            (a[0] * (b[1] * c[2] - b[2] * c[1])
                + a[1] * (b[2] * c[0] - b[0] * c[2])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0
        })
        .sum::<f64>();
    let volume_m3 = signed_volume_m3.abs();
    if !volume_m3.is_finite() || volume_m3 <= 0.0 {
        return Err(TetrahedronGenerationError::DegenerateNestedTetrahedronShellPlc);
    }
    Ok(TriangulatedShell {
        node_ids: component.node_ids,
        facet_indices: component.facet_indices,
        volume_m3,
    })
}

#[derive(Debug, Clone)]
struct BoundaryComponent {
    node_ids: Vec<TopologyEntityId>,
    facet_indices: Vec<usize>,
}

fn boundary_components(plc: &ProtectedBoundaryComplex) -> Vec<BoundaryComponent> {
    let mut adjacency = BTreeMap::<TopologyEntityId, BTreeSet<TopologyEntityId>>::new();
    let mut node_facets = BTreeMap::<TopologyEntityId, BTreeSet<usize>>::new();
    for (facet_index, facet) in plc.facets.iter().enumerate() {
        for node_id in &facet.node_ids {
            adjacency.entry(node_id.clone()).or_default();
            node_facets
                .entry(node_id.clone())
                .or_default()
                .insert(facet_index);
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

    let mut components = Vec::<BoundaryComponent>::new();
    let mut visited = BTreeSet::<TopologyEntityId>::new();
    for start in adjacency.keys() {
        if visited.contains(start) {
            continue;
        }
        let mut node_ids = Vec::<TopologyEntityId>::new();
        let mut facet_indices = BTreeSet::<usize>::new();
        let mut stack = vec![start.clone()];
        while let Some(node_id) = stack.pop() {
            if !visited.insert(node_id.clone()) {
                continue;
            }
            node_ids.push(node_id.clone());
            if let Some(indices) = node_facets.get(&node_id) {
                facet_indices.extend(indices.iter().copied());
            }
            if let Some(neighbors) = adjacency.get(&node_id) {
                stack.extend(
                    neighbors
                        .iter()
                        .filter(|neighbor| !visited.contains(*neighbor))
                        .cloned(),
                );
            }
        }
        node_ids.sort();
        components.push(BoundaryComponent {
            node_ids,
            facet_indices: facet_indices.into_iter().collect(),
        });
    }
    components
}
