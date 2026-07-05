use std::collections::BTreeMap;

use runmat_meshing_core::contracts::{ProtectedBoundaryComplex, TopologyEntityId};

use super::super::TetrahedronGenerationError;

pub(in crate::generate) fn plc_coordinates_and_bounds(
    plc: &ProtectedBoundaryComplex,
) -> Result<(BTreeMap<TopologyEntityId, [f64; 3]>, [[f64; 3]; 2]), TetrahedronGenerationError> {
    let mut coordinates_by_id = BTreeMap::<TopologyEntityId, [f64; 3]>::new();
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for node in &plc.nodes {
        if node
            .coordinates_m
            .iter()
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err(TetrahedronGenerationError::NonFinitePlcNode {
                node_id: node.node_id.id.clone(),
            });
        }
        for axis in 0..3 {
            min[axis] = min[axis].min(node.coordinates_m[axis]);
            max[axis] = max[axis].max(node.coordinates_m[axis]);
        }
        coordinates_by_id.insert(node.node_id.clone(), node.coordinates_m);
    }
    if bounds_span([min, max]) <= f64::EPSILON {
        return Err(TetrahedronGenerationError::DegeneratePlcBounds);
    }
    Ok((coordinates_by_id, [min, max]))
}

pub(in crate::generate) fn plc_node_average(
    plc: &ProtectedBoundaryComplex,
) -> Result<[f64; 3], TetrahedronGenerationError> {
    let mut sum = [0.0; 3];
    for node in &plc.nodes {
        for (axis, coordinate) in node.coordinates_m.iter().enumerate() {
            sum[axis] += coordinate;
        }
    }
    let count = plc.nodes.len() as f64;
    let interior = [sum[0] / count, sum[1] / count, sum[2] / count];
    if interior.iter().all(|coordinate| coordinate.is_finite()) {
        Ok(interior)
    } else {
        Err(TetrahedronGenerationError::NonFiniteInteriorPoint)
    }
}

pub(in crate::generate) fn bounds_span(bounds: [[f64; 3]; 2]) -> f64 {
    (0..3)
        .map(|axis| bounds[1][axis] - bounds[0][axis])
        .filter(|span| span.is_finite())
        .fold(0.0_f64, f64::max)
}
