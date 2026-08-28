use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{
    solver_volume_element_identity, AlgorithmVersionSet, CancellationPolicy,
    CanonicalMeshingContract, CurveQualityTargets, ElementOrder, FieldTopologyLocation,
    FieldTopologyMap, GeometryRevisionRef, GeometryTolerancePolicy, MeshNeighbor, MeshRegion,
    MeshingCancellationSignal, MeshingQualityTargets, MeshingRequest, MeshingResourceBudget,
    NeverCancelled, SolverMeshAdaptationKind, SolverMeshAdaptationLineage, SolverMeshArtifact,
    SolverMeshNode, SolverMeshTopology, SolverTransferMethod, SolverVolumeElement, StableDigest,
    SurfaceQualityTargets, VolumeQualityTargets, ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION,
    MESHING_REQUEST_SCHEMA_VERSION,
};
use runmat_meshing_size::metric::{MetricCombinationRule, MetricFieldRequest, MetricTensor3};

use super::*;
use crate::cdt::{
    assign_delaunay_volume_regions, build_delaunay_volume_topology, coarsen_marked_delaunay_volume,
    evaluate_delaunay_volume_quality, refine_marked_delaunay_volume,
    DelaunayAdaptiveRefinementMark, DelaunayTopologyOptions, DelaunayVolumeNode,
    DelaunayVolumeProvenance, DelaunayVolumeQualityOptions,
};

fn region() -> PersistentEntityId {
    PersistentEntityId {
        kind: PersistentEntityKind::Region,
        source_topology_id: "solid".into(),
        assembly_path: Vec::new(),
    }
}

fn topology() -> DelaunayVolumeTopology {
    let nodes = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    .into_iter()
    .enumerate()
    .map(|(index, coordinates_m)| DelaunayVolumeNode {
        identity: StableDigest::from_bytes([(index + 1) as u8; 32]),
        coordinates_m,
    })
    .collect();
    let topology = build_delaunay_volume_topology(
        nodes,
        vec![[0, 1, 2, 3]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assign_delaunay_volume_regions(
        topology,
        vec![region()],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

fn request() -> MeshingRequest {
    MeshingRequest {
        schema_version: MESHING_REQUEST_SCHEMA_VERSION,
        element_order: ElementOrder::Tet4,
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
        metric: metric(),
        quality: MeshingQualityTargets {
            curve: CurveQualityTargets {
                maximum_chordal_deviation_m: 1.0e-5,
                maximum_tangent_change_degrees: 5.0,
                minimum_metric_edge_length: 0.1,
                maximum_metric_edge_length: 10.0,
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
                maximum_metric_edge_length: 10.0,
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

fn metric() -> MetricFieldRequest {
    MetricFieldRequest {
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(1.0).unwrap(),
        maximum_grading_ratio: 1.5,
        contributions: Vec::new(),
    }
}

fn artifact(topology: &DelaunayVolumeTopology) -> SolverMeshArtifact {
    let region = region();
    let nodes = topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| SolverMeshNode {
            node_id: index as u64 + 1,
            stable_identity: node.identity,
            coordinates_m: node.coordinates_m,
            provenance: vec![region.clone()],
            exact_parameters: Vec::new(),
        })
        .collect::<Vec<_>>();
    let volume_elements = topology
        .tetrahedra
        .iter()
        .enumerate()
        .map(|(index, tetrahedron)| SolverVolumeElement {
            element_id: index as u64 + 1,
            stable_identity: solver_volume_element_identity(
                tetrahedron
                    .vertex_indices
                    .map(|vertex| topology.nodes[vertex as usize].identity),
            ),
            order: ElementOrder::Tet4,
            node_ids: tetrahedron
                .vertex_indices
                .map(|vertex| vertex as u64 + 1)
                .into(),
            region_id: region.clone(),
            material_id: "steel".into(),
            provenance: vec![region.clone()],
        })
        .collect::<Vec<_>>();
    let neighbors = topology
        .tetrahedra
        .iter()
        .enumerate()
        .flat_map(|(index, tetrahedron)| {
            tetrahedron
                .neighbors
                .iter()
                .enumerate()
                .map(move |(face, adjacent)| MeshNeighbor {
                    element_id: index as u64 + 1,
                    local_face_index: face as u8,
                    adjacent_element_id: adjacent.map(|value| value as u64 + 1),
                })
        })
        .collect();
    let node_ids = (1..=nodes.len() as u64).collect::<Vec<_>>();
    let element_ids = (1..=volume_elements.len() as u64).collect::<Vec<_>>();
    let mut artifact = SolverMeshArtifact {
        schema_version: ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION,
        canonical_digest: StableDigest::ZERO,
        root_stage_manifest_digest: StableDigest::from_bytes([90; 32]),
        geometry: GeometryRevisionRef {
            source_digest: StableDigest::from_bytes([91; 32]),
            geometry_revision: 1,
            persistent_mapping_version: 1,
        },
        resolved_request: request(),
        topology: SolverMeshTopology {
            nodes,
            volume_elements,
            neighbors,
            boundary_faces: Vec::new(),
            boundary_edges: Vec::new(),
            regions: vec![MeshRegion {
                region_id: region,
                material_id: "steel".into(),
                element_ids: element_ids.clone(),
            }],
            material_interfaces: Vec::new(),
            contacts: Vec::new(),
            field_topologies: vec![
                FieldTopologyMap {
                    topology_id: "nodes".into(),
                    location: FieldTopologyLocation::Node,
                    ordered_entity_ids: node_ids,
                },
                FieldTopologyMap {
                    topology_id: "elements".into(),
                    location: FieldTopologyLocation::VolumeElement,
                    ordered_entity_ids: element_ids,
                },
                FieldTopologyMap {
                    topology_id: "boundary_faces".into(),
                    location: FieldTopologyLocation::BoundaryFace,
                    ordered_entity_ids: Vec::new(),
                },
                FieldTopologyMap {
                    topology_id: "boundary_edges".into(),
                    location: FieldTopologyLocation::BoundaryEdge,
                    ordered_entity_ids: Vec::new(),
                },
            ],
        },
    };
    artifact.seal_canonical_digest().unwrap();
    artifact
}

fn adaptation() -> (
    DelaunayVolumeTopology,
    DelaunayVolumeProvenance,
    MetricFieldRequest,
    DelaunayVolumeQualityOptions,
    super::DelaunayAdaptiveRefinementResult,
) {
    let topology = topology();
    let provenance = DelaunayVolumeProvenance {
        nodes: Vec::new(),
        segments: Vec::new(),
        facets: Vec::new(),
    };
    let metric = metric();
    let quality_options = DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: 10.0,
        maximum_radius_edge_ratio: 10.0,
        ..DelaunayVolumeQualityOptions::default()
    };
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        &metric,
        &provenance,
        quality_options,
        &NeverCancelled,
    )
    .unwrap();
    let mark = DelaunayAdaptiveRefinementMark {
        node_identities: quality.tetrahedra[0].node_identities,
        indicator_value: 1.0,
    };
    let refinement = refine_marked_delaunay_volume(
        DelaunayVolumeRefinementInput {
            topology: &topology,
            metric_request: &metric,
            provenance: &provenance,
            quality: &quality,
            quality_options,
        },
        &[mark],
        DelaunayAdaptiveRefinementOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    (topology, provenance, metric, quality_options, refinement)
}

#[test]
fn adaptive_lineage_builds_canonical_refinement_and_coarsening_records() {
    let (topology, provenance, metric, quality_options, refinement) = adaptation();
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        &metric,
        &provenance,
        quality_options,
        &NeverCancelled,
    )
    .unwrap();
    let input = DelaunayVolumeRefinementInput {
        topology: &topology,
        metric_request: &metric,
        provenance: &provenance,
        quality: &quality,
        quality_options,
    };
    let source = artifact(&topology);
    let refined = artifact(&refinement.topology);
    let adaptation = build_refinement_solver_adaptation(
        input,
        &refinement,
        &source,
        &refined,
        DelaunayAdaptiveRefinementOptions::default(),
        DelaunayAdaptiveTransferOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let transfer = &adaptation.transfer_map;
    assert_eq!(transfer.node_transfers.len(), 1);
    assert_eq!(transfer.volume_element_transfers.len(), 4);
    assert_eq!(
        transfer.node_transfers[0].method,
        SolverTransferMethod::BarycentricInterpolation
    );
    assert!(transfer.node_transfers[0]
        .sources
        .iter()
        .all(|source| source.weight == 0.25));
    assert!(transfer
        .volume_element_transfers
        .iter()
        .all(|entry| entry.method == SolverTransferMethod::CentroidProjection));
    transfer.validate_against(&source, &refined).unwrap();
    adaptation
        .lineage
        .validate_against(&source, &refined, transfer)
        .unwrap();
    assert_eq!(
        adaptation.lineage.kind,
        SolverMeshAdaptationKind::HRefinement
    );
    assert_eq!(adaptation.lineage.marks.len(), 1);
    assert_eq!(adaptation.lineage.mutations.len(), 1);
    let encoded = adaptation.lineage.canonical_encode().unwrap();
    assert_eq!(
        SolverMeshAdaptationLineage::canonical_decode(&encoded).unwrap(),
        adaptation.lineage
    );
    let mut tampered = adaptation.lineage.clone();
    tampered.mutations[0].created_cells.pop();
    assert!(tampered
        .validate_against(&source, &refined, transfer)
        .is_err());

    let inserted = transfer.node_transfers[0].target_stable_identity;
    let coarsening = coarsen_marked_delaunay_volume(
        input,
        &refinement,
        &[inserted],
        DelaunayAdaptiveCoarseningOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let restored = artifact(&coarsening.topology);
    let reverse = build_coarsening_solver_adaptation(
        DelaunayAdaptiveCoarseningInput {
            original: input,
            refinement: &refinement,
            removal_node_identities: &[inserted],
            coarsening: &coarsening,
            source_artifact: &refined,
            target_artifact: &restored,
        },
        DelaunayAdaptiveCoarseningOptions::default(),
        DelaunayAdaptiveTransferOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert!(reverse.transfer_map.node_transfers.is_empty());
    assert_eq!(reverse.transfer_map.volume_element_transfers.len(), 1);
    reverse
        .transfer_map
        .validate_against(&refined, &restored)
        .unwrap();
    reverse
        .lineage
        .validate_against(&refined, &restored, &reverse.transfer_map)
        .unwrap();
    assert_eq!(reverse.lineage.kind, SolverMeshAdaptationKind::HCoarsening);
    assert_eq!(
        reverse.lineage.requested_removal_node_identities,
        vec![inserted]
    );
    assert_eq!(
        reverse.lineage.mutations[0].removed_cells,
        adaptation.lineage.mutations[0].created_cells
    );
}

#[test]
fn adaptive_transfer_rejects_mismatches_limits_cancellation_and_invalid_options() {
    let (topology, provenance, metric, quality_options, refinement) = adaptation();
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        &metric,
        &provenance,
        quality_options,
        &NeverCancelled,
    )
    .unwrap();
    let input = DelaunayVolumeRefinementInput {
        topology: &topology,
        metric_request: &metric,
        provenance: &provenance,
        quality: &quality,
        quality_options,
    };
    let source = artifact(&topology);
    let refined = artifact(&refinement.topology);
    assert_eq!(
        build_refinement_solver_adaptation(
            input,
            &refinement,
            &source,
            &source,
            DelaunayAdaptiveRefinementOptions::default(),
            DelaunayAdaptiveTransferOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveTransferErrorKind::InvalidArtifact
    );
    let mut displaced = source.clone();
    displaced.topology.nodes[0].coordinates_m[0] = 0.125;
    displaced.seal_canonical_digest().unwrap();
    assert_eq!(
        build_refinement_solver_adaptation(
            input,
            &refinement,
            &displaced,
            &refined,
            DelaunayAdaptiveRefinementOptions::default(),
            DelaunayAdaptiveTransferOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveTransferErrorKind::InvalidArtifact
    );
    assert_eq!(
        build_refinement_solver_adaptation(
            input,
            &refinement,
            &source,
            &refined,
            DelaunayAdaptiveRefinementOptions::default(),
            DelaunayAdaptiveTransferOptions {
                maximum_point_location_predicates: 1,
                ..DelaunayAdaptiveTransferOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveTransferErrorKind::ResourceLimit
    );
    assert_eq!(
        build_refinement_solver_adaptation(
            input,
            &refinement,
            &source,
            &refined,
            DelaunayAdaptiveRefinementOptions {
                cancellation_check_interval: 0,
                ..DelaunayAdaptiveRefinementOptions::default()
            },
            DelaunayAdaptiveTransferOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveTransferErrorKind::InvalidOptions
    );
    assert_eq!(
        build_refinement_solver_adaptation(
            input,
            &refinement,
            &source,
            &refined,
            DelaunayAdaptiveRefinementOptions::default(),
            DelaunayAdaptiveTransferOptions::default(),
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveTransferErrorKind::Cancelled
    );
}

struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}
