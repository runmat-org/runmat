//! Avoid optional shape searches when certified cells retain affine edge geometry.
use super::MidpointMap;
use crate::cdt::solver_topology::{
    error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
};
use runmat_geometry_core::GeometryEvaluationControl;
use runmat_meshing_core::SolverMeshTopology;

/// Caller must independently certify every element first. No coordinate is
/// changed here; this selects whether optional curved-node improvement is useful.
pub(super) fn retains_affine_edge_nodes(
    topology: &SolverMeshTopology,
    midpoints: &MidpointMap<'_>,
    control: &dyn GeometryEvaluationControl,
) -> Result<bool, DelaunaySolverTopologyError> {
    let point = |id: u64| {
        topology
            .nodes
            .get(id.checked_sub(1)? as usize)
            .filter(|node| node.node_id == id)
            .map(|node| node.coordinates_m.canonical_value())
    };
    for (edge, midpoint) in midpoints {
        control.checkpoint().map_err(error::geometry)?;
        control.consume_search_work(1).map_err(error::geometry)?;
        let [Some(left), Some(right), Some(middle)] =
            [point(edge[0]), point(edge[1]), point(*midpoint)]
        else {
            return Err(error::failure(
                DelaunaySolverTopologyErrorKind::InvalidMesh,
                "affine geometry references an absent canonical node",
            ));
        };
        if !edge_is_affine(left, right, middle) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn edge_is_affine(left: [f64; 3], right: [f64; 3], midpoint: [f64; 3]) -> bool {
    (0..3).all(|axis| {
        let expected = left[axis] * 0.5 + right[axis] * 0.5;
        // Floating evaluation/placement roundoff only: no authored geometric
        // tolerance enters this decision, and original coordinates are retained.
        let scale = left[axis]
            .abs()
            .max(right[axis].abs())
            .max(midpoint[axis].abs());
        let roundoff = 32.0 * f64::EPSILON * scale;
        expected.is_finite()
            && midpoint[axis].is_finite()
            && (midpoint[axis] - expected).abs() <= roundoff
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Floating-roundoff affine detection is invariant across the tested translations/scales
    //   and rejects material curvature and nonfinite coordinates. The production fast path retains
    //   every coordinate and requires independent positive-Jacobian certification first; this primitive
    //   test alone is not whole-study evidence.
    #[test]
    fn affine_detection_respects_scale_and_retains_material_curvature() {
        for scale in [1.0e-9_f64, 1.0, 1.0e9] {
            for shift in [0.0, 100.0 * scale] {
                let left = [shift, -scale, scale];
                let right = [shift + 2.0 * scale, scale, 3.0 * scale];
                let midpoint: [f64; 3] =
                    std::array::from_fn(|axis| left[axis] * 0.5 + right[axis] * 0.5);
                assert!(edge_is_affine(left, right, midpoint));
                let mut roundoff = midpoint;
                roundoff[0] = roundoff[0].next_up();
                assert!(edge_is_affine(left, right, roundoff));
                let mut curved = midpoint;
                curved[1] += 1.0e-3 * scale;
                assert!(!edge_is_affine(left, right, curved));
            }
        }
        assert!(!edge_is_affine([0.0; 3], [1.0; 3], [f64::NAN; 3]));
    }
}
