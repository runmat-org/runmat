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
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                [0.0; 3],
            ),
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
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                [0.0; 3],
            ),
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
            [None, None],
        )
        .unwrap(),
        ([std::f64::consts::TAU, 0.5], [6.0, 0.5])
    );
}

// witness: FEA-MESH-DISCRETIZATION-001/volume_quality_convergence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: A seam node carrying only its u = 0 image pairs with a node near u = 2 pi
//   across the seam; the coax Tet10 study exercises the whole elevation.
#[test]
fn periodic_surface_midpoint_crosses_the_seam_without_a_second_seam_image() {
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
    let seam = surface_node(&[[0.0, 0.5]], &face, chart);
    let tau = std::f64::consts::TAU;
    let (left, right) = compatible_surface_parameters(
        &seam,
        &surface_node(&[[tau - 0.2, 0.5]], &face, chart),
        &surface_use,
        [Some(tau), None],
    )
    .unwrap();
    assert_eq!(left, [0.0, 0.5]);
    assert!((right[0] + 0.2).abs() < 1.0e-12 && right[1] == 0.5);
    let bounds = [
        runmat_geometry_core::ParameterRange {
            start: 0.0,
            end: tau,
        },
        runmat_geometry_core::ParameterRange {
            start: 0.0,
            end: 1.0,
        },
    ];
    let middle = wrap_into_period(interpolate2(left, right, 0.5), [Some(tau), None], bounds);
    assert!((middle[0] - (tau - 0.1)).abs() < 1.0e-12 && middle[1] == 0.5);
}
