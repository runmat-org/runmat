//! Canonical geometry for an authored explicit triangular-shell region.

use std::collections::BTreeMap;

use runmat_analysis_core::{StructuralElementKind, StructuralModel};
use runmat_meshing_core::StableDigest;
use thiserror::Error;

use super::node_space::CanonicalStructuralNodeSpace;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExplicitShellTriangle {
    pub element_id: String,
    pub node_identities: [StableDigest; 3],
    pub coordinates_m: [[f64; 3]; 3],
    pub oriented_area_vector_m2: [f64; 3],
    pub area_m2: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExplicitShellSurfaceGeometry {
    pub coordinates_m: BTreeMap<StableDigest, [f64; 3]>,
    pub triangles: Vec<ExplicitShellTriangle>,
    pub area_m2: f64,
    pub nodal_measures_m2: BTreeMap<StableDigest, f64>,
    pub oriented_node_normals: BTreeMap<StableDigest, [f64; 3]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub(crate) enum ExplicitShellSurfaceError {
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

pub(crate) fn explicit_shell_surface_geometry(
    structural: &StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    region_id: &str,
) -> Result<ExplicitShellSurfaceGeometry, ExplicitShellSurfaceError> {
    let authored_nodes = structural
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();
    let canonical_coordinates = topology
        .nodes
        .iter()
        .map(|node| (node.identity, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut coordinates_m = BTreeMap::new();
    let mut nodal_measures_m2 = BTreeMap::<StableDigest, f64>::new();
    let mut normal_sums = BTreeMap::<StableDigest, [f64; 3]>::new();
    let mut triangles = Vec::new();
    let mut area_m2 = 0.0;

    for element in structural
        .elements
        .iter()
        .filter(|element| element.region_id == region_id)
    {
        let StructuralElementKind::Shell(shell) = &element.kind else {
            return Err(ExplicitShellSurfaceError::NonShellElement {
                region_id: region_id.into(),
                element_id: element.element_id.clone(),
            });
        };
        let mut identities = [StableDigest::ZERO; 3];
        for (slot, node_id) in identities.iter_mut().zip(shell.node_ids) {
            *slot = authored_nodes
                .get(&node_id)
                .and_then(|index| topology.explicit_node_indices.get(*index))
                .and_then(|index| topology.nodes.get(*index))
                .map(|node| node.identity)
                .ok_or_else(|| ExplicitShellSurfaceError::MissingNode {
                    element_id: element.element_id.clone(),
                    node_id,
                })?;
        }
        let mut points = [[0.0; 3]; 3];
        for ((point, identity), node_id) in points.iter_mut().zip(identities).zip(shell.node_ids) {
            *point = *canonical_coordinates.get(&identity).ok_or_else(|| {
                ExplicitShellSurfaceError::MissingNode {
                    element_id: element.element_id.clone(),
                    node_id,
                }
            })?;
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
        let nodal_measure = triangle_area / 3.0;
        area_m2 += triangle_area;
        for (identity, node_id) in identities.into_iter().zip(shell.node_ids) {
            let coordinate = canonical_coordinates
                .get(&identity)
                .copied()
                .ok_or_else(|| ExplicitShellSurfaceError::MissingNode {
                    element_id: element.element_id.clone(),
                    node_id,
                })?;
            coordinates_m.insert(identity, coordinate);
            *nodal_measures_m2.entry(identity).or_insert(0.0) += nodal_measure;
            normal_sums
                .entry(identity)
                .and_modify(|normal| *normal = add(*normal, area_vector))
                .or_insert(area_vector);
        }
        triangles.push(ExplicitShellTriangle {
            element_id: element.element_id.clone(),
            node_identities: identities,
            coordinates_m: points,
            oriented_area_vector_m2: scale(area_vector, 0.5),
            area_m2: triangle_area,
        });
    }

    if triangles.is_empty() {
        return Err(ExplicitShellSurfaceError::EmptyRegion {
            region_id: region_id.into(),
        });
    }
    let oriented_node_normals = normal_sums
        .into_iter()
        .map(|(identity, normal)| {
            let length = magnitude(normal);
            if !length.is_finite() || length <= 1.0e-12 {
                return Err(ExplicitShellSurfaceError::UndefinedNodeNormal {
                    node_identity: identity,
                });
            }
            Ok((identity, scale(normal, length.recip())))
        })
        .collect::<Result<_, _>>()?;

    Ok(ExplicitShellSurfaceGeometry {
        coordinates_m,
        triangles,
        area_m2,
        nodal_measures_m2,
        oriented_node_normals,
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
