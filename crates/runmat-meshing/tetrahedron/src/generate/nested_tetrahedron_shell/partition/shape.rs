use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{MeshingStage, TopologyEntityId},
    quality::{
        predicate::{solve_3x3, Point3},
        tolerance::MeshingTolerance,
    },
};

use crate::generate::TetrahedronGenerationError;

use super::super::shell::NestedTetrahedronShell;

const MAX_BARYCENTRIC_PARTITION_DIVISIONS: usize = 12;

#[derive(Debug, Clone)]
pub(super) struct CentralInnerTetrahedronPartition {
    pub(super) inner_lower_bounds: [f64; 4],
    pub(super) inner_scale: f64,
    pub(super) divisions: usize,
    pub(super) inner_node_ids: [TopologyEntityId; 4],
}

pub(super) fn central_inner_tetrahedron_partition(
    shell: &NestedTetrahedronShell,
    coordinates_by_node_id: &BTreeMap<TopologyEntityId, Point3>,
    outer_points: &[Point3],
) -> Result<Option<CentralInnerTetrahedronPartition>, TetrahedronGenerationError> {
    let tolerance = MeshingTolerance::default();
    let mut candidates = Vec::<(TopologyEntityId, [f64; 4])>::new();
    let mut inner_lower_bounds = [f64::INFINITY; 4];
    for node_id in &shell.inner_node_ids {
        let point = coordinates_by_node_id
            .get(node_id)
            .copied()
            .ok_or_else(|| TetrahedronGenerationError::MissingPlcNode {
                node_id: node_id.id.clone(),
            })?;
        let Some(barycentric) = barycentric_coordinates(point, outer_points, tolerance) else {
            return Ok(None);
        };
        if barycentric.iter().any(|value| !value.is_finite()) {
            return Ok(None);
        }
        candidates.push((node_id.clone(), barycentric));
        for index in 0..4 {
            inner_lower_bounds[index] = inner_lower_bounds[index].min(barycentric[index]);
        }
    }
    if inner_lower_bounds
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Ok(None);
    }
    let inner_scale = 1.0 - inner_lower_bounds.iter().sum::<f64>();
    if !inner_scale.is_finite() || inner_scale <= 0.0 {
        return Ok(None);
    }
    let divisions = (1.0 / inner_scale).round() as usize;
    if !(2..=MAX_BARYCENTRIC_PARTITION_DIVISIONS).contains(&divisions) {
        return Ok(None);
    }
    let aligned_scale = 1.0 / divisions as f64;
    if (inner_scale - aligned_scale).abs() > 1.0e-8 {
        return Ok(None);
    }
    for lower_bound in &inner_lower_bounds {
        let aligned = (lower_bound * divisions as f64).round() / divisions as f64;
        if (*lower_bound - aligned).abs() > 1.0e-8 {
            return Ok(None);
        }
    }
    let mut inner_node_ids = [(); 4].map(|_| TopologyEntityId {
        stage: MeshingStage::ProtectedBoundaryComplex,
        id: String::new(),
    });
    let mut seen = BTreeSet::<usize>::new();
    for (node_id, barycentric) in candidates {
        let elevated = (0..4)
            .filter(|index| {
                (barycentric[*index] - (inner_lower_bounds[*index] + inner_scale)).abs() <= 1.0e-8
            })
            .collect::<Vec<_>>();
        if elevated.len() != 1 {
            return Ok(None);
        }
        let elevated_index = elevated[0];
        for (index, value) in barycentric.iter().enumerate() {
            let expected = if index == elevated_index {
                inner_lower_bounds[index] + inner_scale
            } else {
                inner_lower_bounds[index]
            };
            if (*value - expected).abs() > 1.0e-8 {
                return Ok(None);
            }
        }
        if !seen.insert(elevated_index) {
            return Ok(None);
        }
        inner_node_ids[elevated_index] = node_id;
    }
    if seen.len() != 4 {
        return Ok(None);
    }
    Ok(Some(CentralInnerTetrahedronPartition {
        inner_lower_bounds,
        inner_scale,
        divisions,
        inner_node_ids,
    }))
}

fn barycentric_coordinates(
    point: Point3,
    outer_points: &[Point3],
    tolerance: MeshingTolerance,
) -> Option<[f64; 4]> {
    let origin = outer_points[0];
    let matrix = [
        [
            outer_points[1][0] - origin[0],
            outer_points[2][0] - origin[0],
            outer_points[3][0] - origin[0],
        ],
        [
            outer_points[1][1] - origin[1],
            outer_points[2][1] - origin[1],
            outer_points[3][1] - origin[1],
        ],
        [
            outer_points[1][2] - origin[2],
            outer_points[2][2] - origin[2],
            outer_points[3][2] - origin[2],
        ],
    ];
    let rhs = [
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    ];
    let solved = solve_3x3(matrix, rhs, tolerance)?;
    Some([
        1.0 - solved[0] - solved[1] - solved[2],
        solved[0],
        solved[1],
        solved[2],
    ])
}
