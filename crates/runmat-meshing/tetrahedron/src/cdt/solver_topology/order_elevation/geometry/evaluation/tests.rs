use runmat_geometry_core::PersistentEntityKind;
use runmat_meshing_core::StableDigest;

use super::*;

fn edge() -> PersistentEntityId {
    PersistentEntityId::new(
        PersistentEntityKind::Edge,
        "edge:closed".to_owned(),
        Vec::new(),
    )
}

fn node(parameters: &[f64], edge: &PersistentEntityId) -> SolverMeshNode {
    SolverMeshNode {
        node_id: 1,
        stable_identity: StableDigest::from_bytes([1; 32]),
        coordinates_m: [0.0; 3],
        provenance: vec![edge.clone()],
        exact_parameters: parameters
            .iter()
            .map(|parameter| SolverNodeExactParameter::Curve {
                source_edge_id: edge.clone(),
                parameter: *parameter,
            })
            .collect(),
    }
}

fn surface_node(
    parameters: &[[f64; 2]],
    face: &PersistentEntityId,
    chart: StableDigest,
) -> SolverMeshNode {
    SolverMeshNode {
        node_id: 1,
        stable_identity: StableDigest::from_bytes([1; 32]),
        coordinates_m: [0.0; 3],
        provenance: vec![face.clone()],
        exact_parameters: parameters
            .iter()
            .map(|evaluator_uv| SolverNodeExactParameter::Surface {
                source_face_id: face.clone(),
                chart_id: chart,
                evaluator_uv: *evaluator_uv,
            })
            .collect(),
    }
}

#[test]
fn closed_curve_midpoint_uses_the_nearest_parameter_images() {
    let edge = edge();
    let seam = node(&[0.0, std::f64::consts::TAU], &edge);

    assert_eq!(
        compatible_curve_parameters(&seam, &node(&[0.25], &edge), &edge).unwrap(),
        (0.0, 0.25)
    );
    assert_eq!(
        compatible_curve_parameters(&seam, &node(&[6.0], &edge), &edge).unwrap(),
        (std::f64::consts::TAU, 6.0)
    );
}

#[test]
fn periodic_surface_midpoint_uses_the_nearest_chart_images() {
    let face = PersistentEntityId::new(
        PersistentEntityKind::Face,
        "face:periodic".to_owned(),
        Vec::new(),
    );
    let chart = StableDigest::from_bytes([2; 32]);
    let surface_use = SurfaceUse {
        source_face_id: &face,
        chart_id: chart,
    };
    let seam = surface_node(&[[0.0, 0.5], [std::f64::consts::TAU, 0.5]], &face, chart);

    assert_eq!(
        compatible_surface_parameters(
            &seam,
            &surface_node(&[[6.0, 0.5]], &face, chart),
            &surface_use,
        )
        .unwrap(),
        ([std::f64::consts::TAU, 0.5], [6.0, 0.5])
    );
}
