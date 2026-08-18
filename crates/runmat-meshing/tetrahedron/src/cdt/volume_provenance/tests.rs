use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled, StableDigest};

use super::*;
use crate::cdt::{
    assign_delaunay_volume_regions, build_delaunay_volume_topology, DelaunayTopologyOptions,
    DelaunayVolumeNode,
};

fn entity(kind: PersistentEntityKind, value: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: value.to_owned(),
        assembly_path: Vec::new(),
    }
}

fn region(value: &str) -> PersistentEntityId {
    entity(PersistentEntityKind::Region, value)
}

fn topology() -> DelaunayVolumeTopology {
    let nodes = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0],
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
        vec![[0, 1, 2, 3], [1, 4, 2, 3]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assign_delaunay_volume_regions(
        topology,
        vec![region("inner"), region("outer")],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

fn provenance() -> DelaunayVolumeProvenance {
    DelaunayVolumeProvenance {
        nodes: vec![DelaunayNodeProvenance {
            node_identity: StableDigest::from_bytes([2; 32]),
            entity_ids: vec![entity(PersistentEntityKind::Vertex, "corner")],
        }],
        segments: vec![DelaunaySegmentProvenance {
            node_identities: [
                StableDigest::from_bytes([2; 32]),
                StableDigest::from_bytes([3; 32]),
            ],
            entity_ids: vec![entity(PersistentEntityKind::Edge, "feature")],
            edge_parameters: [0.0, 1.0],
        }],
        facets: vec![
            DelaunayFacetProvenance {
                node_identities: [
                    StableDigest::from_bytes([2; 32]),
                    StableDigest::from_bytes([3; 32]),
                    StableDigest::from_bytes([4; 32]),
                ],
                chart_id: StableDigest::from_bytes([40; 32]),
                entity_ids: vec![entity(PersistentEntityKind::Face, "interface")],
                region_ids: vec![region("inner"), region("outer")],
            },
            DelaunayFacetProvenance {
                node_identities: [
                    StableDigest::from_bytes([3; 32]),
                    StableDigest::from_bytes([4; 32]),
                    StableDigest::from_bytes([5; 32]),
                ],
                chart_id: StableDigest::from_bytes([41; 32]),
                entity_ids: vec![entity(PersistentEntityKind::Face, "outer-boundary")],
                region_ids: vec![region("outer")],
            },
        ],
    }
}

#[test]
fn derives_exact_metric_context_from_persistent_simplex_incidence() {
    let topology = topology();
    let provenance = provenance();
    let first = derive_delaunay_volume_metric_contexts(
        &topology,
        &provenance,
        DelaunayVolumeProvenanceOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let second = derive_delaunay_volume_metric_contexts(
        &topology,
        &provenance,
        DelaunayVolumeProvenanceOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(first, second);
    assert_eq!(first.len(), 2);
    assert_eq!(
        first[0].incident_entity_ids,
        vec![
            entity(PersistentEntityKind::Face, "interface"),
            entity(PersistentEntityKind::Edge, "feature"),
            entity(PersistentEntityKind::Vertex, "corner"),
            region("inner"),
        ]
    );
    assert_eq!(
        first[1].incident_entity_ids,
        vec![
            entity(PersistentEntityKind::Face, "interface"),
            entity(PersistentEntityKind::Face, "outer-boundary"),
            entity(PersistentEntityKind::Edge, "feature"),
            entity(PersistentEntityKind::Vertex, "corner"),
            region("outer"),
        ]
    );
}

#[test]
fn rejects_noncanonical_or_false_simplex_provenance() {
    let topology = topology();
    let mut noncanonical = provenance();
    noncanonical.segments[0].node_identities.swap(0, 1);
    assert_eq!(
        validate_delaunay_volume_provenance(
            &topology,
            &noncanonical,
            DelaunayVolumeProvenanceOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeProvenanceErrorKind::InvalidProvenance
    );

    let mut false_regions = provenance();
    false_regions.facets[0].region_ids = vec![region("inner")];
    assert_eq!(
        validate_delaunay_volume_provenance(
            &topology,
            &false_regions,
            DelaunayVolumeProvenanceOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeProvenanceErrorKind::InvalidProvenance
    );

    let mut wrong_dimension = provenance();
    wrong_dimension.nodes[0].entity_ids = vec![entity(PersistentEntityKind::Edge, "wrong")];
    assert_eq!(
        validate_delaunay_volume_provenance(
            &topology,
            &wrong_dimension,
            DelaunayVolumeProvenanceOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeProvenanceErrorKind::InvalidProvenance
    );

    let mut invalid_parameters = provenance();
    invalid_parameters.segments[0].edge_parameters[1] = f64::NAN;
    assert_eq!(
        validate_delaunay_volume_provenance(
            &topology,
            &invalid_parameters,
            DelaunayVolumeProvenanceOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeProvenanceErrorKind::InvalidProvenance
    );

    let mut missing_chart = provenance();
    missing_chart.facets[0].chart_id = StableDigest::ZERO;
    assert_eq!(
        validate_delaunay_volume_provenance(
            &topology,
            &missing_chart,
            DelaunayVolumeProvenanceOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeProvenanceErrorKind::InvalidProvenance
    );
}

struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn enforces_real_inventory_limits_and_cancellation() {
    let topology = topology();
    let provenance = provenance();
    assert_eq!(
        validate_delaunay_volume_provenance(
            &topology,
            &provenance,
            DelaunayVolumeProvenanceOptions {
                maximum_facet_bindings: 1,
                ..DelaunayVolumeProvenanceOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeProvenanceErrorKind::ResourceLimit
    );
    assert_eq!(
        validate_delaunay_volume_provenance(
            &topology,
            &provenance,
            DelaunayVolumeProvenanceOptions::default(),
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeProvenanceErrorKind::Cancelled
    );
    assert_eq!(
        validate_delaunay_volume_provenance(
            &topology,
            &provenance,
            DelaunayVolumeProvenanceOptions {
                cancellation_check_interval: 0,
                ..DelaunayVolumeProvenanceOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeProvenanceErrorKind::InvalidOptions
    );
}
