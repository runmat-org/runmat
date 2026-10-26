use runmat_meshing_core::SolverMeshNode;

use super::geometry::SurfaceUse;

const SEARCH_SCALES: [f64; 8] = [
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
    0.0078125,
    0.00390625,
    0.001953125,
];

pub(super) fn interior_coordinates(
    initial: [f64; 3],
    endpoints: [[f64; 3]; 2],
) -> impl Iterator<Item = [f64; 3]> + Clone {
    let edge_length = endpoints[0]
        .into_iter()
        .zip(endpoints[1])
        .map(|(left, right)| (right - left) * (right - left))
        .sum::<f64>()
        .sqrt();
    SEARCH_SCALES
        .into_iter()
        .filter(move |_| edge_length.is_finite() && edge_length != 0.0)
        .flat_map(move |scale| {
            (0..3).flat_map(move |axis| {
                [-1.0, 1.0].map(move |sign| {
                    let mut point = initial;
                    point[axis] += sign * scale * edge_length;
                    point
                })
            })
        })
}

pub(super) fn surface_parameters(
    initial: &SolverMeshNode,
    endpoints: [&SolverMeshNode; 2],
    surface_use: &SurfaceUse,
) -> impl Iterator<Item = [f64; 2]> + Clone {
    surface_basis(initial, endpoints, surface_use)
        .into_iter()
        .flat_map(|(initial, perpendicular)| {
            SEARCH_SCALES.into_iter().flat_map(move |scale| {
                [-1.0, 1.0].map(move |sign| {
                    [
                        initial[0] + sign * scale * perpendicular[0],
                        initial[1] + sign * scale * perpendicular[1],
                    ]
                })
            })
        })
}
fn surface_basis(
    initial: &SolverMeshNode,
    endpoints: [&SolverMeshNode; 2],
    surface_use: &SurfaceUse,
) -> Option<([f64; 2], [f64; 2])> {
    use super::geometry::surface_parameters as parameters;
    let (left, right) = parameters(endpoints[0], surface_use)
        .flat_map(|left| parameters(endpoints[1], surface_use).map(move |right| (left, right)))
        .min_by(|(left_a, right_a), (left_b, right_b)| {
            squared_distance(*left_a, *right_a)
                .total_cmp(&squared_distance(*left_b, *right_b))
                .then_with(|| left_a[0].total_cmp(&left_b[0]))
                .then_with(|| left_a[1].total_cmp(&left_b[1]))
                .then_with(|| right_a[0].total_cmp(&right_b[0]))
                .then_with(|| right_a[1].total_cmp(&right_b[1]))
        })?;
    let midpoint = [(left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5];
    let initial = parameters(initial, surface_use).min_by(|left, right| {
        squared_distance(*left, midpoint)
            .total_cmp(&squared_distance(*right, midpoint))
            .then_with(|| left[0].total_cmp(&right[0]))
            .then_with(|| left[1].total_cmp(&right[1]))
    })?;
    let tangent = [right[0] - left[0], right[1] - left[1]];
    let perpendicular = [-tangent[1], tangent[0]];
    (squared_distance(perpendicular, [0.0; 2]) != 0.0).then_some((initial, perpendicular))
}

fn squared_distance(left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2)
}

#[cfg(test)]
mod tests {
    use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
    use runmat_meshing_core::{SolverNodeExactParameter, StableDigest};

    use super::*;

    fn node(node_id: u64, coordinates_m: [f64; 3], uv: [f64; 2]) -> SolverMeshNode {
        SolverMeshNode {
            node_id,
            stable_identity: StableDigest::from_bytes([node_id as u8; 32]),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(coordinates_m),
            provenance: Vec::new(),
            exact_parameters: vec![SolverNodeExactParameter::Surface {
                source_face_id: face(),
                chart_id: StableDigest::from_bytes([9; 32]),
                evaluator_uv: uv,
            }],
        }
    }

    fn face() -> PersistentEntityId {
        PersistentEntityId::new(PersistentEntityKind::Face, "face:test".into(), Vec::new())
    }

    #[test]
    fn interior_search_spans_three_physical_dimensions() {
        let points =
            interior_coordinates([0.5, 0.0, 0.0], [[0.0; 3], [1.0, 0.0, 0.0]]).collect::<Vec<_>>();
        assert!(points.iter().any(|point| point[0] != 0.5));
        assert!(points.iter().any(|point| point[1] != 0.0));
        assert!(points.iter().any(|point| point[2] != 0.0));
    }

    #[test]
    fn surface_search_adds_the_independent_chart_direction() {
        let left = node(1, [0.0; 3], [0.0, 0.5]);
        let right = node(2, [1.0, 0.0, 0.0], [1.0, 0.5]);
        let initial = node(3, [0.5, 0.0, 0.0], [0.5, 0.5]);
        let use_record = SurfaceUse {
            source_face_id: &face(),
            chart_id: StableDigest::from_bytes([9; 32]),
        };
        let parameters =
            surface_parameters(&initial, [&left, &right], &use_record).collect::<Vec<_>>();
        assert!(parameters.iter().all(|uv| uv[0] == 0.5));
        assert!(parameters.iter().any(|uv| uv[1] < 0.5));
        assert!(parameters.iter().any(|uv| uv[1] > 0.5));
    }
}
