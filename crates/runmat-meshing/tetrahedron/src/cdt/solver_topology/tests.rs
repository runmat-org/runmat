#[path = "tests/cold_projection.rs"]
mod cold_projection;
#[path = "tests/metric_authority.rs"]
mod metric_authority;

use runmat_meshing_core::{
    AlgorithmVersionSet, CancellationPolicy, CurveQualityTargets, ElementOrder,
    GeometryTolerancePolicy, MeshingCancellationSignal, MeshingQualityTargets, MeshingRequest,
    MeshingSizeLimits, NeverCancelled, SurfaceQualityTargets, VolumeQualityTargets,
    MESHING_REQUEST_SCHEMA_VERSION,
};
use runmat_meshing_size::metric::{MetricCombinationRule, MetricFieldRequest, MetricTensor3};

use super::*;

fn request(order: ElementOrder) -> MeshingRequest {
    MeshingRequest {
        schema_version: MESHING_REQUEST_SCHEMA_VERSION,
        element_order: order,
        dimension: runmat_meshing_core::MeshDimension::Solid,
        deterministic_seed: 17,
        algorithms: AlgorithmVersionSet {
            geometry: "geometry/current".into(),
            curve: "curve/current".into(),
            surface: "surface/current".into(),
            plc: "plc/current".into(),
            tetrahedron: "tetrahedron/current".into(),
            optimization: "optimization/current".into(),
            validation: "validation/current".into(),
        },
        tolerance: GeometryTolerancePolicy {
            source_tolerance_m: runmat_types::physical::Length::new(1.0e-8),
            absolute_floor_m: runmat_types::physical::Length::new(1.0e-10),
            model_relative_term: 1.0e-9,
            requested_deviation_m: runmat_types::physical::Length::new(1.0e-5),
            maximum_healing_displacement_m: runmat_types::physical::Length::new(1.0e-6),
        },
        metric: MetricFieldRequest {
            entity_supports: Vec::new(),
            located_sources: Vec::new(),
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(2.0).unwrap(),
            maximum_grading_ratio: 1.5,
            contributions: Vec::new(),
            cells_across_gap: None,
        },
        quality: MeshingQualityTargets {
            curve: CurveQualityTargets {
                maximum_chordal_deviation_m: runmat_types::physical::Length::new(1.0e-5),
                maximum_tangent_change_degrees: 5.0,
                minimum_metric_edge_length: 0.1,
                maximum_metric_edge_length: 2.0,
            },
            surface: SurfaceQualityTargets {
                minimum_metric_edge_length: 0.1,
                minimum_metric_angle_degrees: 20.0,
                maximum_physical_aspect_ratio: 10.0,
                maximum_chordal_deviation_m: runmat_types::physical::Length::new(1.0e-5),
                maximum_normal_deviation_degrees: 5.0,
            },
            volume: VolumeQualityTargets {
                maximum_radius_edge_ratio: 10.0,
                minimum_scaled_jacobian: 0.01,
                maximum_metric_edge_length: 2.0,
            },
        },
        limits: MeshingSizeLimits {
            maximum_nodes: 100,
            maximum_elements: 100,
            maximum_recursion_depth: 32,
        },
        cancellation: CancellationPolicy {
            maximum_checkpoint_latency_ms: 100,
            maximum_work_units_between_checks: 100,
        },
        size_controls: Vec::new(),
    }
}

fn volume_options() -> DelaunayVolumeMeshOptions {
    DelaunayVolumeMeshOptions {
        quality: super::super::DelaunayVolumeQualityOptions {
            maximum_metric_edge_length: 2.0,
            maximum_radius_edge_ratio: 10.0,
            minimum_metric_scaled_jacobian: 0.01,
            ..super::super::DelaunayVolumeQualityOptions::default()
        },
        ..DelaunayVolumeMeshOptions::default()
    }
}

#[test]
fn validated_tet4_projects_to_one_canonical_solver_topology() {
    let (exact_topology, exact_surface) = crate::cdt::constraints::tests::tetrahedron();
    let request = request(ElementOrder::Tet4);
    let volume_options = volume_options();
    let volume_mesh = super::super::construct_delaunay_volume_mesh(
        &exact_topology,
        &exact_surface,
        &request.metric,
        volume_options,
        &NeverCancelled,
    )
    .unwrap();
    let input = || DelaunaySolverTopologyInput {
        exact_topology: &exact_topology,
        exact_surface: &exact_surface,
        volume_mesh: &volume_mesh,
        volume_options,
        volume_metric: &request.metric,
        request: &request,
        exact_evaluation: None,
    };
    let result = build_delaunay_solver_topology(
        input(),
        DelaunaySolverTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let repeated = build_delaunay_solver_topology(
        input(),
        DelaunaySolverTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(result, repeated);
    assert_eq!(result.nodes.len(), 5);
    assert_eq!(result.volume_elements.len(), 4);
    assert_eq!(result.boundary_faces.len(), 4);
    assert_eq!(result.boundary_edges.len(), 6);
    assert_eq!(result.regions.len(), 1);
    assert!(result.conformal_interfaces.is_empty());
    assert!(result.contacts.is_empty());
    assert_eq!(
        result
            .nodes
            .iter()
            .filter(|node| node.exact_parameters.is_empty())
            .count(),
        1
    );
    assert!(result
        .nodes
        .iter()
        .filter(|node| !node.exact_parameters.is_empty())
        .all(|node| {
            node.exact_parameters
                .iter()
                .filter(|parameter| {
                    matches!(
                        parameter,
                        runmat_meshing_core::SolverNodeExactParameter::Curve { .. }
                    )
                })
                .count()
                == 3
        }));
    assert!(result.volume_elements.iter().all(|element| {
        let points = std::array::from_fn(|index| {
            result.nodes[element.node_ids[index] as usize - 1]
                .coordinates_m
                .canonical_value()
        });
        runmat_meshing_core::quality::predicate::tetrahedron_signed_volume(points) > 0.0
    }));
    for face in &result.boundary_faces {
        let element = &result.volume_elements[face.adjacent_volume_element_ids[0] as usize - 1];
        let opposite = element
            .node_ids
            .iter()
            .find(|node| !face.node_ids.contains(node))
            .unwrap();
        let point = |node_id: u64| {
            result.nodes[node_id as usize - 1]
                .coordinates_m
                .canonical_value()
        };
        assert_eq!(
            runmat_meshing_core::quality::predicate::orient3d([
                point(face.node_ids[0]),
                point(face.node_ids[1]),
                point(face.node_ids[2]),
                point(*opposite),
            ])
            .unwrap(),
            runmat_meshing_core::quality::predicate::PredicateSign::Negative
        );
    }
    runmat_meshing_core::validate_solver_mesh_topology(&result, &request).unwrap();
}

#[test]
fn projection_retains_chart_aware_exact_surface_parameters() {
    let (exact_topology, mut exact_surface) = crate::cdt::constraints::tests::tetrahedron();
    let triangle = &exact_surface.triangles[0];
    let source_face_id = triangle.source_face_id.clone();
    let chart_id = triangle.chart_id;
    let node_id = triangle.node_ids[0];
    exact_surface
        .nodes
        .iter_mut()
        .find(|node| node.node_id == node_id)
        .unwrap()
        .uses
        .push(runmat_meshing_surface::ExactFaceMeshNodeUse {
            source_face_id: source_face_id.clone(),
            chart_id,
            uv: [0.2, 0.3],
            evaluator_uv: [0.25, 0.35],
            exact_edge_parameters: Vec::new(),
        });
    let request = request(ElementOrder::Tet4);
    let volume_options = volume_options();
    let volume_mesh = super::super::construct_delaunay_volume_mesh(
        &exact_topology,
        &exact_surface,
        &request.metric,
        volume_options,
        &NeverCancelled,
    )
    .unwrap();
    let result = build_delaunay_solver_topology(
        DelaunaySolverTopologyInput {
            exact_topology: &exact_topology,
            exact_surface: &exact_surface,
            volume_mesh: &volume_mesh,
            volume_options,
            volume_metric: &request.metric,
            request: &request,
            exact_evaluation: None,
        },
        DelaunaySolverTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let node = result
        .nodes
        .iter()
        .find(|node| volume_mesh.topology.nodes[node.node_id as usize - 1].identity == node_id)
        .unwrap();
    assert!(node.exact_parameters.contains(
        &runmat_meshing_core::SolverNodeExactParameter::Surface {
            source_face_id,
            chart_id,
            evaluator_uv: [0.25, 0.35],
        }
    ));

    let mut conflicting_surface = exact_surface.clone();
    let segment = conflicting_surface
        .boundary_segments
        .iter()
        .find(|segment| segment.node_ids.contains(&node_id))
        .unwrap()
        .clone();
    conflicting_surface
        .nodes
        .iter_mut()
        .find(|node| node.node_id == node_id)
        .unwrap()
        .uses[0]
        .exact_edge_parameters
        .push(runmat_meshing_surface::ExactFaceMeshEdgeParameter {
            source_coedge_id: segment.source_coedge_id,
            source_edge_id: segment.source_edge_id,
            parameter: 0.5,
        });
    let multiple_images = build_delaunay_solver_topology(
        DelaunaySolverTopologyInput {
            exact_topology: &exact_topology,
            exact_surface: &conflicting_surface,
            volume_mesh: &volume_mesh,
            volume_options,
            volume_metric: &request.metric,
            request: &request,
            exact_evaluation: None,
        },
        DelaunaySolverTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let node = multiple_images
        .nodes
        .iter()
        .find(|node| volume_mesh.topology.nodes[node.node_id as usize - 1].identity == node_id)
        .unwrap();
    assert!(node.exact_parameters.iter().any(|parameter| matches!(
        parameter,
        runmat_meshing_core::SolverNodeExactParameter::Curve { parameter, .. }
            if parameter.to_bits() == 0.5_f64.to_bits()
    )));
}

// witness: FEA-MESH-DISCRETIZATION-001/tet10_elevation @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Tet10 elevation is deterministic and shares exact boundary midpoints; the
//   scaled-Jacobian floor on thin or curved real parts is not asserted.
#[test]
fn deterministic_tet10_elevation_shares_exact_boundary_midpoints() {
    let (exact_topology, exact_surface) = crate::cdt::constraints::tests::tetrahedron();
    let request = request(ElementOrder::Tet10);
    let volume_options = volume_options();
    let volume_mesh = super::super::construct_delaunay_volume_mesh(
        &exact_topology,
        &exact_surface,
        &request.metric,
        volume_options,
        &NeverCancelled,
    )
    .unwrap();
    let input = || DelaunaySolverTopologyInput {
        exact_topology: &exact_topology,
        exact_surface: &exact_surface,
        volume_mesh: &volume_mesh,
        volume_options,
        volume_metric: &request.metric,
        request: &request,
        exact_evaluation: Some(DelaunayExactEvaluation {
            evaluator: &super::test_evaluator::EVALUATOR,
            control: &super::test_evaluator::CONTROL,
        }),
    };
    let topology = build_delaunay_solver_topology(
        input(),
        DelaunaySolverTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let repeated = build_delaunay_solver_topology(
        input(),
        DelaunaySolverTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(topology, repeated);
    assert_eq!(topology.nodes.len(), 15);
    assert_eq!(topology.volume_elements.len(), 4);
    assert!(topology
        .volume_elements
        .iter()
        .all(|element| { element.order == ElementOrder::Tet10 && element.node_ids.len() == 10 }));
    assert!(topology.boundary_faces.iter().all(|face| {
        face.order == runmat_meshing_core::BoundaryTriangleOrder::Tri6 && face.node_ids.len() == 6
    }));
    assert!(topology.boundary_edges.iter().all(|edge| {
        edge.order == runmat_meshing_core::BoundaryEdgeOrder::Line3 && edge.node_ids.len() == 3
    }));
    for element in &topology.volume_elements {
        for (local_edge, endpoints) in runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS
            .iter()
            .enumerate()
        {
            let left = topology.nodes[element.node_ids[endpoints[0]] as usize - 1]
                .coordinates_m
                .canonical_value();
            let right = topology.nodes[element.node_ids[endpoints[1]] as usize - 1]
                .coordinates_m
                .canonical_value();
            let midpoint = &topology.nodes[element.node_ids[4 + local_edge] as usize - 1];
            assert_eq!(
                midpoint.coordinates_m.canonical_value(),
                [
                    left[0] * 0.5 + right[0] * 0.5,
                    left[1] * 0.5 + right[1] * 0.5,
                    left[2] * 0.5 + right[2] * 0.5,
                ]
            );
            if !midpoint.exact_parameters.is_empty() {
                assert_eq!(midpoint.exact_parameters.len(), 3);
                assert!(matches!(
                    midpoint.exact_parameters[0],
                    runmat_meshing_core::SolverNodeExactParameter::Curve { .. }
                ));
            }
        }
    }
    runmat_meshing_core::validate_solver_mesh_topology(&topology, &request).unwrap();

    let mut limited_request = request.clone();
    limited_request.limits.maximum_nodes = 9;
    let limited = DelaunaySolverTopologyInput {
        request: &limited_request,
        ..input()
    };
    assert_eq!(
        build_delaunay_solver_topology(
            limited,
            DelaunaySolverTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunaySolverTopologyErrorKind::ResourceLimit
    );
}

// witness: FEA-MESH-DISCRETIZATION-001/volume_quality_convergence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Curve midpoints whose pcurve images a trim classifier reports outside (the
//   pin fitting v1 STEP) are admitted by curve/surface agreement; concave-trim surface
//   fallback is exercised by the site CAD runs.
#[test]
fn tet10_curve_midpoints_do_not_trim_classify_their_own_pcurve_images() {
    let (exact_topology, exact_surface) = crate::cdt::constraints::tests::tetrahedron();
    let request = request(ElementOrder::Tet10);
    let volume_options = volume_options();
    let volume_mesh = super::super::construct_delaunay_volume_mesh(
        &exact_topology,
        &exact_surface,
        &request.metric,
        volume_options,
        &NeverCancelled,
    )
    .unwrap();
    let topology = build_delaunay_solver_topology(
        DelaunaySolverTopologyInput {
            exact_topology: &exact_topology,
            exact_surface: &exact_surface,
            volume_mesh: &volume_mesh,
            volume_options,
            volume_metric: &request.metric,
            request: &request,
            exact_evaluation: Some(DelaunayExactEvaluation {
                evaluator: &super::test_evaluator::BOUNDARY_OUTSIDE_EVALUATOR,
                control: &super::test_evaluator::CONTROL,
            }),
        },
        DelaunaySolverTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    let curve_midpoints = topology
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.exact_parameters.first(),
                Some(runmat_meshing_core::SolverNodeExactParameter::Curve { .. })
            ) && node.exact_parameters.len() == 3
        })
        .count();
    assert_eq!(curve_midpoints, 6);
    runmat_meshing_core::validate_solver_mesh_topology(&topology, &request).unwrap();
}

#[test]
fn curved_tet10_nodes_optimize_deterministically() {
    let (exact_topology, exact_surface) = crate::cdt::constraints::tests::tetrahedron();
    let request = request(ElementOrder::Tet10);
    let volume_options = volume_options();
    let volume_mesh = super::super::construct_delaunay_volume_mesh(
        &exact_topology,
        &exact_surface,
        &request.metric,
        volume_options,
        &NeverCancelled,
    )
    .unwrap();
    let input = || DelaunaySolverTopologyInput {
        exact_topology: &exact_topology,
        exact_surface: &exact_surface,
        volume_mesh: &volume_mesh,
        volume_options,
        volume_metric: &request.metric,
        request: &request,
        exact_evaluation: Some(DelaunayExactEvaluation {
            evaluator: &super::test_evaluator::WARPED_EVALUATOR,
            control: &super::test_evaluator::CONTROL,
        }),
    };
    let topology = build_delaunay_solver_topology(
        input(),
        DelaunaySolverTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let repeated = build_delaunay_solver_topology(
        input(),
        DelaunaySolverTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(topology, repeated);

    let element = &topology.volume_elements[0];
    for (local_edge, endpoints) in runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS
        .iter()
        .enumerate()
    {
        let left = topology.nodes[element.node_ids[endpoints[0]] as usize - 1]
            .coordinates_m
            .canonical_value();
        let right = topology.nodes[element.node_ids[endpoints[1]] as usize - 1]
            .coordinates_m
            .canonical_value();
        let node = &topology.nodes[element.node_ids[4 + local_edge] as usize - 1];
        let target: [f64; 3] = std::array::from_fn(|axis| left[axis] * 0.5 + right[axis] * 0.5);
        let edge_length_squared = left
            .into_iter()
            .zip(right)
            .map(|(left, right)| (left - right) * (left - right))
            .sum::<f64>();
        let optimized_distance_squared = node
            .coordinates_m
            .canonical_value()
            .into_iter()
            .zip(target)
            .map(|(left, right)| (left - right) * (left - right))
            .sum::<f64>();
        assert!(optimized_distance_squared < edge_length_squared / 64.0);
        if let Some(runmat_meshing_core::SolverNodeExactParameter::Curve { parameter, .. }) =
            node.exact_parameters.first()
        {
            assert_ne!(parameter.to_bits(), 0.5_f64.to_bits());
        } else {
            assert!(node.exact_parameters.is_empty());
        }
    }
    runmat_meshing_core::validate_solver_mesh_topology(&topology, &request).unwrap();
}

#[test]
fn exact_interface_and_contact_classification_preserve_typed_sides() {
    use runmat_geometry_core::{
        ExactContactPair, ExactSharedInterface, PersistentEntityId, PersistentEntityKind,
        TopologicalOrientation,
    };

    let (mut topology, _) = crate::cdt::constraints::tests::tetrahedron();
    let face_a = topology.faces[0].id.clone();
    let face_b = topology.faces[1].id.clone();
    let region_a = topology.regions[0].id.clone();
    let region_b = PersistentEntityId::new(
        PersistentEntityKind::Region,
        "region:b".into(),
        vec!["root".into()],
    );
    topology.interfaces = vec![ExactSharedInterface {
        face_id: face_a.clone(),
        side_a_region_id: region_a.clone(),
        side_b_region_id: region_b.clone(),
        side_a_orientation: TopologicalOrientation::Forward,
        side_b_orientation: TopologicalOrientation::Reversed,
    }];
    let index = super::classification::ClassificationIndex::new(&topology, None).unwrap();
    let mut regions = vec![region_a.clone(), region_b];
    regions.sort();
    let interface = index
        .classify(std::slice::from_ref(&face_a), &regions)
        .unwrap();
    assert_eq!(
        interface.role,
        runmat_meshing_core::BoundaryFaceRole::ConformalInterface
    );
    assert_eq!(interface.outward_region_id, &region_a);

    drop(index);
    topology.interfaces.clear();
    let contact_id = PersistentEntityId::new(
        PersistentEntityKind::Contact,
        "contact".into(),
        vec!["root".into()],
    );
    topology.contacts = vec![ExactContactPair {
        id: contact_id.clone(),
        side_a_face_ids: vec![face_a.clone()],
        side_b_face_ids: vec![face_b.clone()],
        pairing_schema_version: 1,
        pairing_contract_digest: [1; 32],
    }];
    let index = super::classification::ClassificationIndex::new(&topology, None).unwrap();
    let primary_entities = [face_a, contact_id.clone()];
    let primary = index
        .classify(&primary_entities, std::slice::from_ref(&region_a))
        .unwrap();
    let secondary_entities = [face_b, contact_id];
    let secondary = index
        .classify(&secondary_entities, std::slice::from_ref(&region_a))
        .unwrap();
    assert_eq!(
        primary.role,
        runmat_meshing_core::BoundaryFaceRole::ContactPrimary
    );
    assert_eq!(
        secondary.role,
        runmat_meshing_core::BoundaryFaceRole::ContactSecondary
    );
}

#[test]
fn projection_rejects_unsupported_order_and_resource_limit() {
    let (exact_topology, exact_surface) = crate::cdt::constraints::tests::tetrahedron();
    let linear_request = request(ElementOrder::Tet4);
    let volume_options = volume_options();
    let volume_mesh = super::super::construct_delaunay_volume_mesh(
        &exact_topology,
        &exact_surface,
        &linear_request.metric,
        volume_options,
        &NeverCancelled,
    )
    .unwrap();
    let quadratic_request = request(ElementOrder::Tet10);
    let input = |request| DelaunaySolverTopologyInput {
        exact_topology: &exact_topology,
        exact_surface: &exact_surface,
        volume_mesh: &volume_mesh,
        volume_options,
        volume_metric: &linear_request.metric,
        request,
        exact_evaluation: None,
    };
    assert_eq!(
        build_delaunay_solver_topology(
            input(&quadratic_request),
            DelaunaySolverTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunaySolverTopologyErrorKind::InvalidOptions
    );
    assert_eq!(
        build_delaunay_solver_topology(
            input(&linear_request),
            DelaunaySolverTopologyOptions {
                maximum_boundary_faces: 3,
                ..DelaunaySolverTopologyOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunaySolverTopologyErrorKind::ResourceLimit
    );
}

struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn projection_preserves_cancellation() {
    let (exact_topology, exact_surface) = crate::cdt::constraints::tests::tetrahedron();
    let request = request(ElementOrder::Tet4);
    let volume_options = volume_options();
    let volume_mesh = super::super::construct_delaunay_volume_mesh(
        &exact_topology,
        &exact_surface,
        &request.metric,
        volume_options,
        &NeverCancelled,
    )
    .unwrap();
    let result = build_delaunay_solver_topology(
        DelaunaySolverTopologyInput {
            exact_topology: &exact_topology,
            exact_surface: &exact_surface,
            volume_mesh: &volume_mesh,
            volume_options,
            volume_metric: &request.metric,
            request: &request,
            exact_evaluation: None,
        },
        DelaunaySolverTopologyOptions::default(),
        &Cancelled,
    );
    assert_eq!(
        result.unwrap_err().kind,
        DelaunaySolverTopologyErrorKind::Cancelled
    );
}
