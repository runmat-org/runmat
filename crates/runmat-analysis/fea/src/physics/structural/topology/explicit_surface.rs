//! Canonical geometry for an authored explicit triangular-shell region.

use crate::topology::storage::{self, SortedValues, TopologyStorageError};

use runmat_analysis_core::{StructuralElementKind, StructuralModel};
use runmat_meshing_core::StableDigest;
use thiserror::Error;

use super::node_space::CanonicalStructuralNodeSpace;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExplicitShellTriangle<'a> {
    pub element_id: &'a str,
    pub node_identities: [StableDigest; 3],
    pub coordinates_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [[f64; 3]; 3]>,
    pub oriented_area_vector_m2: [f64; 3],
    pub area_m2: runmat_types::physical::Area,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExplicitShellSurfaceNode {
    pub coordinates_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
    pub nodal_measure_m2: f64,
    pub oriented_normal: [f64; 3],
}
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExplicitShellSurfaceGeometry<'a> {
    pub nodes: SortedValues<StableDigest, ExplicitShellSurfaceNode>,
    pub triangles: Vec<ExplicitShellTriangle<'a>>,
    pub area_m2: runmat_types::physical::Area,
}

/// Requested vector slots, including overlapping raw candidates and retained nodes.
/// Borrowed inputs/labels, allocator overhead and RSS are excluded.
pub(crate) fn shell_surface_workspace_bytes(
    structural: &StructuralModel,
    region_id: &str,
) -> Result<u64, TopologyStorageError> {
    storage::check_cancelled()?;
    let faces = structural
        .elements
        .iter()
        .filter(|e| e.region_id == region_id)
        .count();
    let candidates = storage::multiply(faces, 3)?;
    let index = storage::slots::<(u32, usize)>(structural.nodes.len())?;
    let raw = storage::slots::<(StableDigest, [f64; 3])>(candidates)?;
    let nodes = storage::slots::<(StableDigest, ExplicitShellSurfaceNode)>(candidates)?;
    let triangles = storage::slots::<ExplicitShellTriangle<'_>>(faces)?;
    storage::add(storage::add(index, raw)?, storage::add(nodes, triangles)?)
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub(crate) enum ExplicitShellSurfaceError {
    #[error(transparent)]
    Storage(#[from] TopologyStorageError),
    #[error("region `{region_id}` contains non-shell element `{element_id}`")]
    NonShellElement {
        region_id: String,
        element_id: String,
    },
    #[error("shell element `{element_id}` references absent node {node_id}")]
    MissingNode { element_id: String, node_id: u32 },
    #[error("shell element `{element_id}` has zero or non-finite area")]
    DegenerateTriangle { element_id: String },
    #[error("region `{region_id}` contains no explicit shell elements")]
    EmptyRegion { region_id: String },
    #[error("the oriented normal is undefined at canonical node {node_identity:?}")]
    UndefinedNodeNormal { node_identity: StableDigest },
}

pub(crate) fn explicit_shell_surface_geometry<'a>(
    structural: &'a StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    region_id: &str,
) -> Result<ExplicitShellSurfaceGeometry<'a>, ExplicitShellSurfaceError> {
    shell_surface_workspace_bytes(structural, region_id)?;
    let face_count = structural
        .elements
        .iter()
        .filter(|e| e.region_id == region_id)
        .count();
    let authored_nodes = SortedValues::derive(
        structural
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.node_id, i)),
        structural.nodes.len(),
    )?;
    let mut candidates = storage::reserve(storage::multiply(face_count, 3)?)?;
    let mut triangles = storage::reserve(face_count)?;
    let mut area_m2 = 0.0;
    for element in structural
        .elements
        .iter()
        .filter(|e| e.region_id == region_id)
    {
        storage::check_cancelled()?;
        let StructuralElementKind::Shell(shell) = &element.kind else {
            return Err(ExplicitShellSurfaceError::NonShellElement {
                region_id: region_id.into(),
                element_id: element.element_id.clone(),
            });
        };
        let mut identities = [StableDigest::ZERO; 3];
        let mut points = [[0.0; 3]; 3];
        for ((identity, point), node_id) in
            identities.iter_mut().zip(&mut points).zip(shell.node_ids)
        {
            let node = authored_nodes
                .get(&node_id)
                .and_then(|i| topology.explicit_node_indices.get(*i))
                .and_then(|i| topology.nodes.get(*i))
                .ok_or_else(|| ExplicitShellSurfaceError::MissingNode {
                    element_id: element.element_id.clone(),
                    node_id,
                })?;
            *identity = node.identity;
            *point = node.coordinates_m.canonical_value();
        }
        let area_vector = cross(
            subtract(points[1], points[0]),
            subtract(points[2], points[0]),
        );
        let double_area = magnitude(area_vector);
        if !double_area.is_finite() || double_area <= 1.0e-18 {
            return Err(ExplicitShellSurfaceError::DegenerateTriangle {
                element_id: element.element_id.clone(),
            });
        }
        let triangle_area = double_area * 0.5;
        area_m2 += triangle_area;
        candidates.extend(identities.into_iter().zip(points));
        triangles.push(ExplicitShellTriangle {
            element_id: &element.element_id,
            node_identities: identities,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(points),
            oriented_area_vector_m2: scale(area_vector, 0.5),
            area_m2: runmat_types::physical::Area::new(triangle_area),
        });
    }
    if triangles.is_empty() {
        return Err(ExplicitShellSurfaceError::EmptyRegion {
            region_id: region_id.into(),
        });
    }
    candidates.sort_unstable_by_key(|v| v.0);
    candidates.dedup_by_key(|v| v.0);
    let mut nodes = SortedValues::derive(
        candidates.iter().map(|(id, coordinates)| {
            (
                *id,
                ExplicitShellSurfaceNode {
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new(*coordinates),
                    nodal_measure_m2: 0.0,
                    oriented_normal: [0.0; 3],
                },
            )
        }),
        candidates.len(),
    )?;
    drop(candidates);
    for triangle in &triangles {
        storage::check_cancelled()?;
        let points = triangle.coordinates_m.canonical_value();
        // Recompute the original raw cross product; reversing the stored half-vector
        // would change floating-point arithmetic. Accumulate in authored face order.
        let area_vector = cross(
            subtract(points[1], points[0]),
            subtract(points[2], points[0]),
        );
        for id in triangle.node_identities {
            let node = nodes.get_mut(&id).ok_or(TopologyStorageError::Index)?;
            node.oriented_normal = if node.nodal_measure_m2 == 0.0 {
                area_vector
            } else {
                add(node.oriented_normal, area_vector)
            };
            node.nodal_measure_m2 += triangle.area_m2.canonical_value() / 3.0;
        }
    }
    for (identity, node) in nodes.entries_mut() {
        let length = magnitude(node.oriented_normal);
        if !length.is_finite() || length <= 1.0e-12 {
            return Err(ExplicitShellSurfaceError::UndefinedNodeNormal {
                node_identity: *identity,
            });
        }
        node.oriented_normal = scale(node.oriented_normal, length.recip());
    }
    Ok(ExplicitShellSurfaceGeometry {
        nodes,
        triangles,
        area_m2: runmat_types::physical::Area::new(area_m2),
    })
}

pub(crate) fn triangle_centroid(points: [[f64; 3]; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| (points[0][axis] + points[1][axis] + points[2][axis]) / 3.0)
}

pub(crate) fn triangle_area(points: [[f64; 3]; 3]) -> f64 {
    0.5 * magnitude(cross(
        subtract(points[1], points[0]),
        subtract(points[2], points[0]),
    ))
}

pub(crate) fn segment_length(left: [f64; 3], right: [f64; 3]) -> f64 {
    magnitude(subtract(right, left))
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn magnitude(vector: [f64; 3]) -> f64 {
    vector
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests;
