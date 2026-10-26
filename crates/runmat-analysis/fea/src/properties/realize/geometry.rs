use std::collections::BTreeMap;

use runmat_analysis_core::{
    AuthoredOrientationDefinition, ConformalSolidAttachment, StructuralNode,
};
use runmat_meshing_core::SolverMeshArtifact;

use super::PropertyRealizationError;

#[derive(Debug, Clone, Copy)]
pub(super) enum FormulationOrientationRole {
    Beam,
    Shell,
}

pub(super) fn structural_node(
    mesh: &SolverMeshArtifact,
    mesh_node_id: u64,
    nodes: &mut Vec<StructuralNode>,
    ids: &mut BTreeMap<u64, u32>,
) -> Result<u32, PropertyRealizationError> {
    if let Some(id) = ids.get(&mesh_node_id) {
        return Ok(*id);
    }
    let source = mesh_node(mesh, mesh_node_id)?;
    let id = u32::try_from(nodes.len() + 1)
        .map_err(|_| PropertyRealizationError::NodeIdOverflow(mesh_node_id))?;
    nodes.push(StructuralNode {
        node_id: id,
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                source.coordinates_m.canonical_value(),
            ),
        solid_attachment: Some(ConformalSolidAttachment::SolverNode {
            stable_identity: source.stable_identity.0,
        }),
    });
    ids.insert(mesh_node_id, id);
    Ok(id)
}

pub(super) fn mesh_node(
    mesh: &SolverMeshArtifact,
    node_id: u64,
) -> Result<&runmat_meshing_core::SolverMeshNode, PropertyRealizationError> {
    mesh.topology
        .nodes
        .iter()
        .find(|node| node.node_id == node_id)
        .ok_or_else(|| PropertyRealizationError::InvalidMesh(format!("missing node {node_id}")))
}

pub(super) fn edge_segments(
    nodes: &[u64],
    entity: &str,
) -> Result<Vec<[u64; 2]>, PropertyRealizationError> {
    match nodes {
        [a, b] => Ok(vec![[*a, *b]]),
        [a, b, middle] => Ok(vec![[*a, *middle], [*middle, *b]]),
        _ => Err(PropertyRealizationError::UnsupportedBoundaryTopology {
            entity: entity.to_owned(),
            node_count: nodes.len(),
        }),
    }
}

pub(super) fn face_triangles(
    nodes: &[u64],
    entity: &str,
) -> Result<Vec<[u64; 3]>, PropertyRealizationError> {
    match nodes {
        [a, b, c] => Ok(vec![[*a, *b, *c]]),
        [a, b, c, ab, bc, ca] => Ok(vec![
            [*a, *ab, *ca],
            [*ab, *b, *bc],
            [*ca, *bc, *c],
            [*ab, *bc, *ca],
        ]),
        _ => Err(PropertyRealizationError::UnsupportedBoundaryTopology {
            entity: entity.to_owned(),
            node_count: nodes.len(),
        }),
    }
}

pub(super) fn orientation_axis(
    definition: Option<&AuthoredOrientationDefinition>,
    point: [f64; 3],
    default_axis: [f64; 3],
    forbidden_axis: [f64; 3],
    entity: &str,
    role: FormulationOrientationRole,
) -> Result<[f64; 3], PropertyRealizationError> {
    let axis = match definition {
        None => default_axis,
        Some(AuthoredOrientationDefinition::GlobalCartesian { .. }) => match role {
            FormulationOrientationRole::Beam => [0.0, 1.0, 0.0],
            FormulationOrientationRole::Shell => [1.0, 0.0, 0.0],
        },
        Some(AuthoredOrientationDefinition::Cartesian(value)) => match role {
            FormulationOrientationRole::Beam => value.y_axis,
            FormulationOrientationRole::Shell => value.x_axis,
        },
        Some(AuthoredOrientationDefinition::Cylindrical(value)) => {
            let radial = subtract(point, value.origin_m.canonical_value());
            let axis = normalized(value.axis);
            let projected = subtract(radial, scale(axis, dot(radial, axis)));
            if dot(projected, projected) > f64::EPSILON {
                projected
            } else {
                value.reference_direction
            }
        }
        Some(AuthoredOrientationDefinition::Spherical(value)) => {
            let radial = subtract(point, value.origin_m.canonical_value());
            if dot(radial, radial) > f64::EPSILON {
                radial
            } else {
                value.reference_direction
            }
        }
    };
    let axis_norm_squared = dot(axis, axis);
    let forbidden_norm_squared = dot(forbidden_axis, forbidden_axis);
    let cross_product = cross(axis, forbidden_axis);
    let cross_norm_squared = dot(cross_product, cross_product);
    if !axis.iter().all(|component| component.is_finite())
        || !forbidden_axis.iter().all(|component| component.is_finite())
        || axis_norm_squared <= f64::EPSILON
        || forbidden_norm_squared <= f64::EPSILON
        || cross_norm_squared <= 1.0e-24 * axis_norm_squared * forbidden_norm_squared
    {
        return Err(PropertyRealizationError::DegenerateOrientation(
            entity.to_owned(),
        ));
    }
    Ok(axis)
}

pub(super) fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(super) fn midpoint(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
        (a[2] + b[2]) / 2.0,
    ]
}

pub(super) fn centroid(points: [[f64; 3]; 3]) -> [f64; 3] {
    [
        (points[0][0] + points[1][0] + points[2][0]) / 3.0,
        (points[0][1] + points[1][1] + points[2][1]) / 3.0,
        (points[0][2] + points[1][2] + points[2][2]) / 3.0,
    ]
}

pub(super) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(super) fn least_aligned_cartesian_axis(direction: [f64; 3]) -> [f64; 3] {
    let candidates = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut selected = candidates[0];
    for candidate in candidates.into_iter().skip(1) {
        if dot(candidate, direction).abs() < dot(selected, direction).abs() {
            selected = candidate;
        }
    }
    selected
}

fn scale(value: [f64; 3], factor: f64) -> [f64; 3] {
    value.map(|component| component * factor)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}

fn normalized(value: [f64; 3]) -> [f64; 3] {
    scale(value, 1.0 / dot(value, value).sqrt())
}
