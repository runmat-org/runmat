//! True extrema of a published field, read from every stored value of the solution (every
//! node, element or integration point), never from the drawn surface. A figure's legend and
//! its reported peak come from here, so surface sampling, element averaging for display and
//! export compaction cannot hide or move the peak.

use super::{load_field, AnalysisFieldLocation, AnalysisFieldSelection, FIELD_PAGE_VALUES};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FieldExtremum {
    /// The magnitude (vector norm, complex modulus) of one stored value set.
    pub value: f64,
    /// Zero-based node, element or integration-point index in the solver's order.
    pub entity: usize,
    /// The solver node or element (an integration point's element) holding the value.
    pub mesh_entity: usize,
    /// Node coordinates, or the element's corner centroid, in metres.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position_m: Option<[f64; 3]>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FieldExtrema {
    pub minimum: FieldExtremum,
    pub maximum: FieldExtremum,
    /// `node` or `element`; element fields may hold several integration points per element.
    pub location: AnalysisFieldLocation,
    pub values_per_mesh_entity: usize,
}

/// Scan the whole field page by page. `nodes` are solver node coordinates in solver order and
/// `elements` the corner-node indices of each volume element, when the run's mesh is known.
pub fn field_extrema(
    run_id: &str,
    selection: &AnalysisFieldSelection,
    nodes: Option<&[[f64; 3]]>,
    elements: Option<&[Vec<usize>]>,
) -> Result<FieldExtrema, String> {
    let field = load_field(run_id, selection)?;
    let entities = field.value_count / field.components;
    let mesh_entities = match field.location {
        AnalysisFieldLocation::Node => nodes.map(<[_]>::len),
        _ => elements.map(<[_]>::len),
    };
    let per_entity = mesh_entities
        .filter(|&count| count > 0 && entities % count == 0)
        .map_or(1, |count| entities / count);
    let mut minimum: Option<(f64, usize)> = None;
    let mut maximum: Option<(f64, usize)> = None;
    let entities_per_page = (FIELD_PAGE_VALUES / field.components).max(1);
    let mut entity = 0;
    while entity < entities {
        let count = entities_per_page.min(entities - entity);
        let values = field.range(entity * field.components, count * field.components)?;
        for (offset, set) in values.chunks_exact(field.components).enumerate() {
            let value = if field.components == 1 {
                set[0]
            } else {
                set.iter().map(|v| v * v).sum::<f64>().sqrt()
            };
            if !value.is_finite() {
                return Err(format!("field '{}' has a non-finite value", field.label));
            }
            let index = entity + offset;
            if minimum.is_none_or(|(best, _)| value < best) {
                minimum = Some((value, index));
            }
            if maximum.is_none_or(|(best, _)| value > best) {
                maximum = Some((value, index));
            }
        }
        entity += count;
    }
    let locate = |(value, index): (f64, usize)| {
        let mesh_entity = index / per_entity;
        let position_m = match field.location {
            AnalysisFieldLocation::Node => nodes.and_then(|nodes| nodes.get(mesh_entity).copied()),
            _ => elements
                .and_then(|elements| elements.get(mesh_entity))
                .zip(nodes)
                .and_then(|(corners, nodes)| centroid(corners, nodes)),
        };
        FieldExtremum {
            value,
            entity: index,
            mesh_entity,
            position_m,
        }
    };
    let (Some(minimum), Some(maximum)) = (minimum, maximum) else {
        return Err(format!("field '{}' has no values", field.label));
    };
    Ok(FieldExtrema {
        minimum: locate(minimum),
        maximum: locate(maximum),
        location: field.location,
        values_per_mesh_entity: per_entity,
    })
}

fn centroid(corners: &[usize], nodes: &[[f64; 3]]) -> Option<[f64; 3]> {
    if corners.is_empty() {
        return None;
    }
    let mut sum = [0.0; 3];
    for &corner in corners {
        let node = nodes.get(corner)?;
        for axis in 0..3 {
            sum[axis] += node[axis];
        }
    }
    Some(sum.map(|value| value / corners.len() as f64))
}

/// Solver node coordinates and each volume element's node indices, in solver order, from the
/// run's canonical solver mesh artifact.
pub type SolverMeshEntities = (Vec<[f64; 3]>, Vec<Vec<usize>>);

pub fn solver_mesh_entities(path: &str) -> Result<SolverMeshEntities, String> {
    let mesh =
        super::super::solver_mesh_artifact::load_solver_mesh_artifact(std::path::Path::new(path))
            .map_err(|error| error.to_string())?;
    let index_by_id = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<std::collections::HashMap<_, _>>();
    let nodes = mesh
        .topology
        .nodes
        .iter()
        .map(|node| node.coordinates_m.canonical_value())
        .collect();
    let elements = mesh
        .topology
        .volume_elements
        .iter()
        .map(|element| {
            element
                .node_ids
                .iter()
                .map(|id| {
                    index_by_id
                        .get(id)
                        .copied()
                        .ok_or("element node is missing")
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((nodes, elements))
}
