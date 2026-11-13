use std::collections::BTreeMap;

use runmat_meshing_core::quality::predicate::Point3;

use super::super::{
    raw_refill_tetrahedron_with_rejection_reason,
    topology::{sorted_edge, sorted_tetrahedron_nodes},
    ConstrainedCavityNode, ConstrainedCavityRefillOptions, ConstrainedCavityRefillTetrahedron,
    ConstrainedCavityValidationError, RefillTetrahedronError,
};

pub(crate) fn split_completion_tetrahedra_for_node(
    face: [u32; 3],
    cap_node_id: u32,
    split_node: &ConstrainedCavityNode,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<Vec<ConstrainedCavityRefillTetrahedron>>, ConstrainedCavityValidationError> {
    let child_specs = [
        [face[0], face[1], split_node.node_id, cap_node_id],
        [face[1], face[2], split_node.node_id, cap_node_id],
        [face[2], face[0], split_node.node_id, cap_node_id],
    ];
    let mut child_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::with_capacity(3);
    for node_ids in child_specs {
        let points = [
            boundary_nodes[&node_ids[0]],
            boundary_nodes[&node_ids[1]],
            split_node.coordinates_m.canonical_value(),
            boundary_nodes[&cap_node_id],
        ];
        let tetrahedron =
            match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options) {
                Ok(tetrahedron) => tetrahedron,
                Err(RefillTetrahedronError::Rejected(_)) => return Ok(None),
                Err(RefillTetrahedronError::Accounting(cause)) => {
                    return Err(ConstrainedCavityValidationError::Accounting(cause))
                }
            };
        if child_tetrahedra.iter().any(|existing| {
            sorted_tetrahedron_nodes(existing.node_ids)
                == sorted_tetrahedron_nodes(tetrahedron.node_ids)
        }) {
            return Ok(None);
        }
        child_tetrahedra.push(tetrahedron);
    }
    Ok(Some(child_tetrahedra))
}

pub(crate) fn edge_split_completion_tetrahedra_for_node(
    face: [u32; 3],
    edge: [u32; 2],
    cap_node_id: u32,
    split_node: &ConstrainedCavityNode,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<Vec<ConstrainedCavityRefillTetrahedron>>, ConstrainedCavityValidationError> {
    let [a, b] = edge;
    let Some(c) = face
        .into_iter()
        .find(|node_id| *node_id != a && *node_id != b)
    else {
        return Ok(None);
    };
    let child_specs = [
        [a, split_node.node_id, c, cap_node_id],
        [split_node.node_id, b, c, cap_node_id],
    ];
    let mut child_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::with_capacity(2);
    for node_ids in child_specs {
        let points = node_ids.map(|node_id| {
            if node_id == split_node.node_id {
                split_node.coordinates_m.canonical_value()
            } else {
                boundary_nodes[&node_id]
            }
        });
        let tetrahedron =
            match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options) {
                Ok(tetrahedron) => tetrahedron,
                Err(RefillTetrahedronError::Rejected(_)) => return Ok(None),
                Err(RefillTetrahedronError::Accounting(cause)) => {
                    return Err(ConstrainedCavityValidationError::Accounting(cause))
                }
            };
        if child_tetrahedra.iter().any(|existing| {
            sorted_tetrahedron_nodes(existing.node_ids)
                == sorted_tetrahedron_nodes(tetrahedron.node_ids)
        }) {
            return Ok(None);
        }
        child_tetrahedra.push(tetrahedron);
    }
    Ok(Some(child_tetrahedra))
}

pub(crate) fn three_edge_split_completion_tetrahedra_for_node(
    face: [u32; 3],
    cap_node_id: u32,
    split_node_by_edge: &BTreeMap<[u32; 2], u32>,
    split_node_coordinates: &BTreeMap<u32, Point3>,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<Vec<ConstrainedCavityRefillTetrahedron>>, ConstrainedCavityValidationError> {
    let [a, b, c] = face;
    let Some(&ab) = split_node_by_edge.get(&sorted_edge([a, b])) else {
        return Ok(None);
    };
    let Some(&bc) = split_node_by_edge.get(&sorted_edge([b, c])) else {
        return Ok(None);
    };
    let Some(&ca) = split_node_by_edge.get(&sorted_edge([c, a])) else {
        return Ok(None);
    };
    let child_specs = [
        [a, ab, ca, cap_node_id],
        [ab, b, bc, cap_node_id],
        [ca, bc, c, cap_node_id],
        [ab, bc, ca, cap_node_id],
    ];
    let mut child_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::with_capacity(4);
    for node_ids in child_specs {
        let points = node_ids.map(|node_id| {
            split_node_coordinates
                .get(&node_id)
                .copied()
                .unwrap_or_else(|| boundary_nodes[&node_id])
        });
        let tetrahedron =
            match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options) {
                Ok(tetrahedron) => tetrahedron,
                Err(RefillTetrahedronError::Rejected(_)) => return Ok(None),
                Err(RefillTetrahedronError::Accounting(cause)) => {
                    return Err(ConstrainedCavityValidationError::Accounting(cause))
                }
            };
        if child_tetrahedra.iter().any(|existing| {
            sorted_tetrahedron_nodes(existing.node_ids)
                == sorted_tetrahedron_nodes(tetrahedron.node_ids)
        }) {
            return Ok(None);
        }
        child_tetrahedra.push(tetrahedron);
    }
    Ok(Some(child_tetrahedra))
}
