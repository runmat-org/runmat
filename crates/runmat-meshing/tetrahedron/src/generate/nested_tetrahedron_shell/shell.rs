use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{ProtectedBoundaryComplex, TopologyEntityId},
    quality::{
        predicate::{point_in_closed_triangle_surface, solve_3x3, PointInClosedSurface},
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
        .map(|component| triangulated_shell(plc, coordinates_by_id, component, tolerance))
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
        outer_node_ids: outer.tetrahedron_node_ids.to_vec(),
        inner_node_ids: inner.tetrahedron_node_ids.to_vec(),
        outer_facet_indices: outer.facet_indices.clone(),
        inner_facet_indices: inner.facet_indices.clone(),
        outer_volume_m3: outer.volume_m3,
        inner_volume_m3: inner.volume_m3,
    })
}

#[derive(Debug, Clone)]
struct TriangulatedShell {
    node_ids: Vec<TopologyEntityId>,
    tetrahedron_node_ids: [TopologyEntityId; 4],
    facet_indices: Vec<usize>,
    volume_m3: f64,
}

fn triangulated_shell(
    plc: &ProtectedBoundaryComplex,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    component: BoundaryComponent,
    tolerance: MeshingTolerance,
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
    let tetrahedron_node_ids =
        tetrahedron_shell_corners(coordinates_by_id, &component.node_ids, volume_m3, tolerance)?;
    Ok(TriangulatedShell {
        node_ids: component.node_ids,
        tetrahedron_node_ids,
        facet_indices: component.facet_indices,
        volume_m3,
    })
}

fn tetrahedron_shell_corners(
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    node_ids: &[TopologyEntityId],
    shell_volume_m3: f64,
    tolerance: MeshingTolerance,
) -> Result<[TopologyEntityId; 4], TetrahedronGenerationError> {
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let candidate_ids = [
                        node_ids[first].clone(),
                        node_ids[second].clone(),
                        node_ids[third].clone(),
                        node_ids[fourth].clone(),
                    ];
                    let candidate_points = candidate_ids
                        .clone()
                        .map(|node_id| coordinates_by_id[&node_id]);
                    let candidate_volume_m3 = tetrahedron_volume(candidate_points);
                    if !candidate_volume_m3.is_finite()
                        || !tolerance.nearly_equal(
                            candidate_volume_m3,
                            shell_volume_m3,
                            shell_volume_m3.abs().max(1.0),
                        )
                    {
                        continue;
                    }
                    if node_ids.iter().all(|node_id| {
                        let Some(barycentric) =
                            barycentric_coordinates(coordinates_by_id[node_id], candidate_points)
                        else {
                            return false;
                        };
                        barycentric.iter().all(|value| {
                            *value >= -1.0e-8 && *value <= 1.0 + 1.0e-8 && value.is_finite()
                        }) && barycentric.iter().any(|value| value.abs() <= 1.0e-8)
                    }) {
                        return Ok(candidate_ids);
                    }
                }
            }
        }
    }
    Err(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc)
}

fn tetrahedron_volume(points: [[f64; 3]; 4]) -> f64 {
    let [a, b, c, d] = points;
    let ad = [a[0] - d[0], a[1] - d[1], a[2] - d[2]];
    let bd = [b[0] - d[0], b[1] - d[1], b[2] - d[2]];
    let cd = [c[0] - d[0], c[1] - d[1], c[2] - d[2]];
    (ad[0] * (bd[1] * cd[2] - bd[2] * cd[1]) - ad[1] * (bd[0] * cd[2] - bd[2] * cd[0])
        + ad[2] * (bd[0] * cd[1] - bd[1] * cd[0]))
        .abs()
        / 6.0
}

fn barycentric_coordinates(point: [f64; 3], tetrahedron: [[f64; 3]; 4]) -> Option<[f64; 4]> {
    let origin = tetrahedron[0];
    let matrix = [
        [
            tetrahedron[1][0] - origin[0],
            tetrahedron[2][0] - origin[0],
            tetrahedron[3][0] - origin[0],
        ],
        [
            tetrahedron[1][1] - origin[1],
            tetrahedron[2][1] - origin[1],
            tetrahedron[3][1] - origin[1],
        ],
        [
            tetrahedron[1][2] - origin[2],
            tetrahedron[2][2] - origin[2],
            tetrahedron[3][2] - origin[2],
        ],
    ];
    let rhs = [
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    ];
    let solved = solve_3x3(matrix, rhs, MeshingTolerance::default())?;
    Some([
        1.0 - solved[0] - solved[1] - solved[2],
        solved[0],
        solved[1],
        solved[2],
    ])
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
