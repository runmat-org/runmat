use std::collections::BTreeMap;

use runmat_meshing_core::contracts::ProtectedBoundaryComplex;

use super::super::TetrahedronGenerationError;

pub(super) fn validate_structured_box_plc(
    plc: &ProtectedBoundaryComplex,
    bounds: [[f64; 3]; 2],
    tolerance: f64,
) -> Result<(), TetrahedronGenerationError> {
    if !plc.protected_edges.is_empty() {
        return Err(TetrahedronGenerationError::UnsupportedStructuredBoxPlc);
    }
    let [min, max] = bounds;
    let coordinates_by_id = plc
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    if !plc_nodes_are_box_corners(plc, bounds, tolerance) {
        return Err(TetrahedronGenerationError::UnsupportedStructuredBoxPlc);
    }
    let mut covered_sides = [false; 6];
    for facet in &plc.facets {
        let coordinates = facet
            .node_ids
            .iter()
            .map(|node_id| {
                coordinates_by_id.get(node_id).copied().ok_or_else(|| {
                    TetrahedronGenerationError::MissingPlcNode {
                        node_id: node_id.id.clone(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let side_index = structured_box_side_index(&coordinates, min, max, tolerance)
            .ok_or(TetrahedronGenerationError::UnsupportedStructuredBoxPlc)?;
        covered_sides[side_index] = true;
    }
    if covered_sides.iter().all(|covered| *covered) {
        Ok(())
    } else {
        Err(TetrahedronGenerationError::UnsupportedStructuredBoxPlc)
    }
}

fn plc_nodes_are_box_corners(
    plc: &ProtectedBoundaryComplex,
    bounds: [[f64; 3]; 2],
    tolerance: f64,
) -> bool {
    if plc.nodes.len() != 8 {
        return false;
    }
    let [min, max] = bounds;
    plc.nodes.iter().all(|node| {
        node.coordinates_m
            .iter()
            .enumerate()
            .all(|(axis, coordinate)| {
                (*coordinate - min[axis]).abs() <= tolerance
                    || (*coordinate - max[axis]).abs() <= tolerance
            })
    })
}

pub(super) fn structured_box_side_index(
    coordinates: &[[f64; 3]],
    min: [f64; 3],
    max: [f64; 3],
    tolerance: f64,
) -> Option<usize> {
    for axis in 0..3 {
        if coordinates
            .iter()
            .all(|point| (point[axis] - min[axis]).abs() <= tolerance)
        {
            return Some(axis * 2);
        }
        if coordinates
            .iter()
            .all(|point| (point[axis] - max[axis]).abs() <= tolerance)
        {
            return Some(axis * 2 + 1);
        }
    }
    None
}
