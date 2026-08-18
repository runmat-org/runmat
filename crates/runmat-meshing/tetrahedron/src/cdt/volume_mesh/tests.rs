use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled};
use runmat_meshing_size::metric::{MetricCombinationRule, MetricFieldRequest, MetricTensor3};

use super::*;

fn metric_request() -> MetricFieldRequest {
    MetricFieldRequest {
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(2.0).unwrap(),
        maximum_grading_ratio: 1.5,
        contributions: Vec::new(),
    }
}

#[test]
fn exact_surface_constructs_one_validated_general_volume_mesh() {
    let (topology, surface) = crate::cdt::constraints::tests::tetrahedron();
    let options = DelaunayVolumeMeshOptions {
        quality: DelaunayVolumeQualityOptions {
            maximum_metric_edge_length: 2.0,
            maximum_radius_edge_ratio: 10.0,
            minimum_metric_scaled_jacobian: 0.01,
            ..DelaunayVolumeQualityOptions::default()
        },
        ..DelaunayVolumeMeshOptions::default()
    };
    let result = construct_delaunay_volume_mesh(
        &topology,
        &surface,
        &metric_request(),
        options,
        &NeverCancelled,
    )
    .unwrap();
    let repeated = construct_delaunay_volume_mesh(
        &topology,
        &surface,
        &metric_request(),
        options,
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(result, repeated);
    assert_eq!(result.topology.tetrahedra.len(), 1);
    assert_eq!(result.provenance.nodes.len(), 4);
    assert_eq!(result.provenance.segments.len(), 6);
    assert_eq!(result.provenance.facets.len(), 4);
    assert!(result.mutations.is_empty());
    validate_delaunay_volume_mesh(
        &topology,
        &surface,
        &metric_request(),
        &result,
        options,
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn final_validation_rejects_lineage_tampering() {
    let (topology, surface) = crate::cdt::constraints::tests::tetrahedron();
    let mut options = DelaunayVolumeMeshOptions::default();
    options.quality.maximum_metric_edge_length = 2.0;
    options.quality.maximum_radius_edge_ratio = 10.0;
    options.quality.minimum_metric_scaled_jacobian = 0.01;
    let mut result = construct_delaunay_volume_mesh(
        &topology,
        &surface,
        &metric_request(),
        options,
        &NeverCancelled,
    )
    .unwrap();
    result.provenance.facets[0].entity_ids[0].source_topology_id = "stale-face".to_owned();

    assert_eq!(
        validate_delaunay_volume_mesh(
            &topology,
            &surface,
            &metric_request(),
            &result,
            options,
            &NeverCancelled,
        )
        .unwrap_err()
        .stage,
        DelaunayVolumeMeshStage::Provenance
    );
}

struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn unified_volume_mesh_maps_options_resources_and_cancellation() {
    let (topology, surface) = crate::cdt::constraints::tests::tetrahedron();
    let inconsistent = DelaunayVolumeMeshOptions {
        provenance: DelaunayVolumeProvenanceOptions {
            maximum_node_bindings: 1,
            ..DelaunayVolumeProvenanceOptions::default()
        },
        ..DelaunayVolumeMeshOptions::default()
    };
    assert_eq!(
        construct_delaunay_volume_mesh(
            &topology,
            &surface,
            &metric_request(),
            inconsistent,
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeMeshErrorKind::InvalidOptions
    );

    let bounded = DelaunayVolumeMeshOptions {
        constraints: DelaunayConstraintOptions {
            maximum_nodes: 3,
            ..DelaunayConstraintOptions::default()
        },
        ..DelaunayVolumeMeshOptions::default()
    };
    assert_eq!(
        construct_delaunay_volume_mesh(
            &topology,
            &surface,
            &metric_request(),
            bounded,
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeMeshErrorKind::ResourceLimit
    );
    assert_eq!(
        construct_delaunay_volume_mesh(
            &topology,
            &surface,
            &metric_request(),
            DelaunayVolumeMeshOptions::default(),
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeMeshErrorKind::Cancelled
    );
}
