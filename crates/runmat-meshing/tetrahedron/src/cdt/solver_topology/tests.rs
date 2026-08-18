use runmat_meshing_core::{
    AlgorithmVersionSet, CancellationPolicy, CurveQualityTargets, ElementOrder,
    GeometryTolerancePolicy, MeshingCancellationSignal, MeshingQualityTargets, MeshingRequest,
    MeshingResourceBudget, NeverCancelled, SurfaceQualityTargets, VolumeQualityTargets,
    MESHING_REQUEST_SCHEMA_VERSION,
};
use runmat_meshing_size::metric::{MetricCombinationRule, MetricFieldRequest, MetricTensor3};

use super::*;

fn request(order: ElementOrder) -> MeshingRequest {
    MeshingRequest {
        schema_version: MESHING_REQUEST_SCHEMA_VERSION,
        element_order: order,
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
            source_tolerance_m: 1.0e-8,
            absolute_floor_m: 1.0e-10,
            model_relative_term: 1.0e-9,
            requested_deviation_m: 1.0e-5,
            maximum_healing_displacement_m: 1.0e-6,
        },
        metric: MetricFieldRequest {
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(2.0).unwrap(),
            maximum_grading_ratio: 1.5,
            contributions: Vec::new(),
        },
        quality: MeshingQualityTargets {
            curve: CurveQualityTargets {
                maximum_chordal_deviation_m: 1.0e-5,
                maximum_tangent_change_degrees: 5.0,
                minimum_metric_edge_length: 0.1,
                maximum_metric_edge_length: 2.0,
            },
            surface: SurfaceQualityTargets {
                minimum_metric_angle_degrees: 20.0,
                maximum_physical_aspect_ratio: 10.0,
                maximum_chordal_deviation_m: 1.0e-5,
                maximum_normal_deviation_degrees: 5.0,
            },
            volume: VolumeQualityTargets {
                maximum_radius_edge_ratio: 10.0,
                minimum_scaled_jacobian: 0.01,
                maximum_metric_edge_length: 2.0,
            },
        },
        resources: MeshingResourceBudget {
            maximum_nodes: 100,
            maximum_elements: 100,
            maximum_memory_bytes: 1_000_000,
            maximum_scratch_bytes: 1_000_000,
            maximum_wall_time_ms: 10_000,
            maximum_artifact_bytes: 1_000_000,
            maximum_search_work: 10_000,
            maximum_recursion_depth: 32,
            maximum_iterations: 10_000,
        },
        cancellation: CancellationPolicy {
            maximum_checkpoint_latency_ms: 100,
            maximum_work_units_between_checks: 100,
        },
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
    let region_materials = [DelaunayRegionMaterial {
        region_id: exact_topology.regions[0].id.clone(),
        material_id: "steel".into(),
    }];
    let input = || DelaunaySolverTopologyInput {
        exact_topology: &exact_topology,
        exact_surface: &exact_surface,
        volume_mesh: &volume_mesh,
        volume_options,
        request: &request,
        region_materials: &region_materials,
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
    assert_eq!(result.nodes.len(), 4);
    assert_eq!(result.volume_elements.len(), 1);
    assert_eq!(result.boundary_faces.len(), 4);
    assert_eq!(result.boundary_edges.len(), 6);
    assert_eq!(result.regions.len(), 1);
    assert!(result.material_interfaces.is_empty());
    assert!(result.contacts.is_empty());
    assert!(result.nodes.iter().all(|node| !node.provenance.is_empty()));
    assert!(result.nodes.iter().all(|node| {
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
    for face in &result.boundary_faces {
        let element = &result.volume_elements[face.adjacent_volume_element_ids[0] as usize - 1];
        let opposite = element
            .node_ids
            .iter()
            .find(|node| !face.node_ids.contains(node))
            .unwrap();
        let point = |node_id: u64| result.nodes[node_id as usize - 1].coordinates_m;
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
    let materials = [DelaunayRegionMaterial {
        region_id: exact_topology.regions[0].id.clone(),
        material_id: "steel".into(),
    }];
    let result = build_delaunay_solver_topology(
        DelaunaySolverTopologyInput {
            exact_topology: &exact_topology,
            exact_surface: &exact_surface,
            volume_mesh: &volume_mesh,
            volume_options,
            request: &request,
            region_materials: &materials,
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
    assert_eq!(
        build_delaunay_solver_topology(
            DelaunaySolverTopologyInput {
                exact_topology: &exact_topology,
                exact_surface: &conflicting_surface,
                volume_mesh: &volume_mesh,
                volume_options,
                request: &request,
                region_materials: &materials,
            },
            DelaunaySolverTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunaySolverTopologyErrorKind::InvalidMesh
    );
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
    let region_b = PersistentEntityId {
        kind: PersistentEntityKind::Region,
        source_topology_id: "region:b".into(),
        assembly_path: vec!["root".into()],
    };
    topology.interfaces = vec![ExactSharedInterface {
        face_id: face_a.clone(),
        side_a_region_id: region_a.clone(),
        side_b_region_id: region_b.clone(),
        side_a_orientation: TopologicalOrientation::Forward,
        side_b_orientation: TopologicalOrientation::Reversed,
    }];
    let index = super::classification::ClassificationIndex::new(&topology);
    let mut regions = vec![region_a.clone(), region_b];
    regions.sort();
    let interface = index
        .classify(std::slice::from_ref(&face_a), &regions)
        .unwrap();
    assert_eq!(
        interface.role,
        runmat_meshing_core::BoundaryFaceRole::MaterialInterface
    );
    assert_eq!(interface.outward_region_id, region_a);

    topology.interfaces.clear();
    let contact_id = PersistentEntityId {
        kind: PersistentEntityKind::Contact,
        source_topology_id: "contact".into(),
        assembly_path: vec!["root".into()],
    };
    topology.contacts = vec![ExactContactPair {
        id: contact_id.clone(),
        side_a_face_ids: vec![face_a.clone()],
        side_b_face_ids: vec![face_b.clone()],
        pairing_schema_version: 1,
        pairing_contract_digest: [1; 32],
    }];
    let index = super::classification::ClassificationIndex::new(&topology);
    let primary = index
        .classify(
            &[face_a, contact_id.clone()],
            std::slice::from_ref(&region_a),
        )
        .unwrap();
    let secondary = index
        .classify(&[face_b, contact_id], std::slice::from_ref(&region_a))
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
fn projection_rejects_unsupported_order_materials_and_resource_limit() {
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
    let no_materials = DelaunaySolverTopologyInput {
        exact_topology: &exact_topology,
        exact_surface: &exact_surface,
        volume_mesh: &volume_mesh,
        volume_options,
        request: &linear_request,
        region_materials: &[],
    };
    assert_eq!(
        build_delaunay_solver_topology(
            no_materials,
            DelaunaySolverTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunaySolverTopologyErrorKind::InvalidMaterials
    );

    let quadratic_request = request(ElementOrder::Tet10);
    let materials = [DelaunayRegionMaterial {
        region_id: exact_topology.regions[0].id.clone(),
        material_id: "steel".into(),
    }];
    let input = |request| DelaunaySolverTopologyInput {
        exact_topology: &exact_topology,
        exact_surface: &exact_surface,
        volume_mesh: &volume_mesh,
        volume_options,
        request,
        region_materials: &materials,
    };
    assert_eq!(
        build_delaunay_solver_topology(
            input(&quadratic_request),
            DelaunaySolverTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunaySolverTopologyErrorKind::UnsupportedOrder
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
    let materials = [DelaunayRegionMaterial {
        region_id: exact_topology.regions[0].id.clone(),
        material_id: "steel".into(),
    }];
    let result = build_delaunay_solver_topology(
        DelaunaySolverTopologyInput {
            exact_topology: &exact_topology,
            exact_surface: &exact_surface,
            volume_mesh: &volume_mesh,
            volume_options,
            request: &request,
            region_materials: &materials,
        },
        DelaunaySolverTopologyOptions::default(),
        &Cancelled,
    );
    assert_eq!(
        result.unwrap_err().kind,
        DelaunaySolverTopologyErrorKind::Cancelled
    );
}
